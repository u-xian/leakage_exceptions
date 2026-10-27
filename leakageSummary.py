"""
Step 6 - conclusion built from the outputs of steps 1-5.

    1. generateData.py    -> the CDRs                      (generate_cdrs)
    2. analyzeLeakage.py  -> total leak, % of revenue      (summarize_leakage)
    3. analyzeGroups.py   -> stand-out groups              (standouts)
    4. chartLeakage.py    -> step / spikes / weekly slots  (detect_step, detect_spikes, detect_hour_spikes,
                             matched to events.csv          detect_weekly, events_near)
    5. compareCalls.py    -> decision-tree rules           (fit_tree, leaf_rules)

Nothing about the causes is hard-coded here: every cause is a detection from
step 4, attributed to CDRs by its detected scope (zone / date / weekday-hour),
matched to the change log for the root cause, corroborated by the step-5 rules,
and given a fix template chosen by the matched event type.

Usage:
    python leakageSummary.py -i cdrs.csv [-e events.csv] [--as-of 2026-10-31] [--rebill-days 60]
"""
import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analyzeGroups import WEEKDAYS, add_time_dimensions, standouts
from analyzeLeakage import summarize_leakage
from chartLeakage import (daily_series, detect_hour_spikes, detect_spikes, detect_step, detect_weekly,
                          events_near, load_events)
from compareCalls import FEATURES, fit_tree, leaf_rules, prepare as tree_prepare
from generateData import RATE_PER_MIN, generate_cdrs

# ---------------------------------------------------------------- fix templates by matched event type

FIX = {
    "tariff_change": "Reload the {zone} tariff with the correct rates and re-rate {zone} CDRs from {start:%Y-%m-%d}; "
                     "add a pre-load validation that blocks any rate dropping > 10% without approval",
    "software_patch": "Re-rate CDRs in the {window} window; add a rating smoke-test to the deployment runbook and "
                      "hold CDRs in the mediation queue while the rating engine is unavailable",
    "failed_file_load": "Fix the {window} batch job; make the loader atomic so CDRs queue instead of passing through "
                        "unrated; alert on batch failure; re-rate the affected CDRs",
    "unknown": "No matching entry in the change log: pull rating-engine logs for {window}, re-rate the affected CDRs",
    "background": "Case-by-case review of the largest records; keep the random-error rate under watch",
}
CHECK_FOR = {"under_rated": "effective_rate_by_zone", "unrated": "unrated_usage_hourly"}


@dataclass
class Cause:
    key: str
    kind: str                 # step | weekly | spike | background
    description: str
    mask: pd.Series           # which leaking CDRs belong to it
    leak_type: str = ""
    event_type: str = "unknown"
    event: str = "no matching event"
    evidence: str = ""
    rules: list = field(default_factory=list)
    window: str = ""
    zone: str = "all zones"
    start: pd.Timestamp | None = None
    recovery_rate: float = 1.0

    @property
    def fix(self) -> str:
        return FIX[self.event_type if self.event_type in FIX else "unknown"].format(zone=self.zone, start=self.start, window=self.window)

    @property
    def daily_check(self) -> str:
        if self.kind == "weekly":
            return "weekly_slot_recurrence"
        if self.kind == "background":
            return "daily_leak_pct_vs_baseline"
        return CHECK_FOR.get(self.leak_type, "daily_leak_pct_vs_baseline")


# ---------------------------------------------------------------- derive causes from steps 2-5


def leak_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Leak type from charged vs tariff (not from the synthetic label)."""
    df = add_time_dimensions(df)
    df["tariff_charge"] = df["billed_minutes"] * df["zone"].map(RATE_PER_MIN)
    df["leak_amt"] = (df["tariff_charge"] - df["charged"]).round(2)
    df["leak_type"] = np.select([(df["charged"] == 0) & (df["tariff_charge"] > 0),
                                 (df["charged"] > 0) & (df["charged"] < df["tariff_charge"])],
                                ["unrated", "under_rated"], default="")
    return df


def majority_type(sub: pd.DataFrame) -> str:
    t = sub.loc[sub["leak_type"] != "", "leak_type"]
    return t.mode().iat[0] if len(t) else ""


def derive_causes(df: pd.DataFrame, events: pd.DataFrame) -> list[Cause]:
    leaking = df["leak_type"] != ""
    causes: list[Cause] = []
    ts = df["call_start"]

    # --- step 4a: weekly pattern -> recurring batch fault
    weekly = detect_weekly(df)
    for wd, grp in weekly.reset_index().groupby("weekday", observed=True):
        hours = sorted(grp["hour"])
        m = leaking & (df["weekday"] == wd) & df["hour"].isin(hours)
        ev = events[events["ts"].dt.day_name().str[:3] == wd]
        ev = ev[ev["ts"].dt.hour.between(min(hours) - 1, max(hours))]
        c = Cause(f"weekly_{wd}_{min(hours):02d}", "weekly",
                  f"{wd} {min(hours):02d}:00-{max(hours) + 1:02d}:00 calls leaking every week", m,
                  leak_type=majority_type(df[m]), window=f"{wd} {min(hours):02d}:00",
                  evidence=f"step 4: weekday x hour slot {grp['ratio'].max():.1f}x overall leak rate, hit in "
                           f"{int(grp['weeks_hit'].max())} weeks")
        if not ev.empty:
            c.event_type, c.event = ev["event_type"].iat[0], f"{len(ev)}x {ev['description'].iat[0]}"
        causes.append(c)
    explained = pd.Series(False, index=df.index)
    for c in causes:
        explained |= c.mask

    # --- step 4b: spikes -> one-off incidents (skip hours already explained by a weekly cause)
    daily = daily_series(df)
    spikes, hour_spikes = detect_spikes(daily), detect_hour_spikes(df)
    for d in spikes.index:
        hrs = sorted(h for (dd, h) in hour_spikes.index if dd == d.date())
        if not hrs:
            continue
        m = leaking & (df["day"] == d.date()) & df["hour"].isin(hrs) & ~explained
        if m.sum() < 5:
            continue                                  # the spike was the weekly fault, already covered
        ev = events_near(events, d + pd.Timedelta(hours=hrs[0]), 36)
        c = Cause(f"spike_{d:%m%d}", "spike", f"{d:%Y-%m-%d} {hrs[0]:02d}:00-{hrs[-1] + 1:02d}:00 incident", m,
                  leak_type=majority_type(df[m]), window=f"{d:%Y-%m-%d} {hrs[0]:02d}:00-{hrs[-1] + 1:02d}:00",
                  start=d, evidence=f"step 4: daily spike z={spikes.loc[d, 'z']}, hours at "
                                    f"{hour_spikes.loc[[(d.date(), h) for h in hrs], 'leak_pct'].mean():.0f}% leak")
        if not ev.empty:
            c.event_type, c.event = ev["event_type"].iat[-1], ev["description"].iat[-1]
        causes.append(c)
        explained |= m

    # --- step 4c: step change -> persistent fault from a date, in the zone that moved
    step = detect_step(daily)
    if step and abs(step["t"]) > 4:
        before, after = df[ts < step["date"]], df[ts >= step["date"]]
        shift = (100 * after.groupby("zone")["leak_amt"].sum() / after.groupby("zone")["tariff_charge"].sum()
                 - 100 * before.groupby("zone")["leak_amt"].sum() / before.groupby("zone")["tariff_charge"].sum())
        zone = shift.idxmax()
        m = leaking & (ts >= step["date"]) & (df["zone"] == zone) & ~explained
        lt = majority_type(df[m])
        m &= df["leak_type"] == lt
        ev = events_near(events, step["date"], 36)
        c = Cause(f"step_{zone}", "step", f"{zone} calls {lt.replace('_', '-')} since {step['date']:%Y-%m-%d}", m,
                  leak_type=lt, zone=zone, start=step["date"], window=f"{zone} from {step['date']:%Y-%m-%d}",
                  evidence=f"step 4: daily leak {step['before']:.1f}% -> {step['after']:.1f}% (t={step['t']:.1f}); "
                           f"{zone} moved +{shift.max():.1f} pts, other zones {shift.drop(zone).max():+.1f} at most")
        if not ev.empty:
            c.event_type, c.event = ev["event_type"].iat[-1], ev["description"].iat[-1]
        causes.append(c)
        explained |= m

    # --- remainder
    causes.append(Cause("background", "background", "Unexplained / random leakage", leaking & ~explained,
                        event_type="background", event="n/a", recovery_rate=0.0,
                        evidence="step 3: no stand-out group; step 5: no rule covers these calls"))

    # --- step 5: corroborate with decision-tree rules (rule covers the cause's calls)
    tdf = tree_prepare(df)
    clf = fit_tree(tdf, "issue" if "issue" in tdf else "is_leaking")
    rules = leaf_rules(clf, tdf, "issue" if "issue" in tdf else "is_leaking")
    leaf = pd.Series(clf.apply(tdf[FEATURES]), index=df.index)
    for c in causes:
        if c.kind == "background":
            continue
        for _, r in rules.iterrows():
            covered = leaf == r["leaf"]
            if (covered & c.mask).sum() / max(c.mask.sum(), 1) >= 0.3:
                c.rules.append(f"{r['rule']} -> {r['predicts']} ({r['rate_%']}%, lift {r['lift']})")
    return causes


# ---------------------------------------------------------------- summary


def classify(df: pd.DataFrame, events_path: str | None = "events.csv"):
    df = leak_flags(df)
    causes = derive_causes(df, load_events(events_path))
    df["cause"] = ""
    for c in reversed(causes):                        # earlier (more specific) causes win
        df.loc[c.mask, "cause"] = c.key
    return df, causes


def summarize(df: pd.DataFrame, causes: list[Cause], as_of: pd.Timestamp, rebill_days: int) -> pd.DataFrame:
    total_expected = df["tariff_charge"].sum()
    in_window = (as_of - df["call_start"]).dt.days <= rebill_days
    rows = []
    for c in causes:
        sub = df[df["cause"] == c.key]
        win = sub.loc[in_window[sub.index], "leak_amt"].sum()
        rows.append({"cause": c.key, "description": c.description, "leaking_calls": len(sub), "leaked": sub["leak_amt"].sum(),
                     "pct_of_revenue": 100 * sub["leak_amt"].sum() / total_expected, "in_rebill_window": win,
                     "recovery_rate": c.recovery_rate, "recoverable": win * c.recovery_rate,
                     "first_seen": sub["call_start"].min(), "last_seen": sub["call_start"].max(), "daily_check": c.daily_check})
    return pd.DataFrame(rows).sort_values("leaked", ascending=False).reset_index(drop=True)


def print_report(df: pd.DataFrame, events_path: str, as_of: pd.Timestamp, rebill_days: int):
    df, causes = classify(df, events_path)
    by_key = {c.key: c for c in causes}
    s = summarize(df, causes, as_of, rebill_days)
    tot = summarize_leakage(df.assign(leak=df["leak_amt"], expected_charge=df["tariff_charge"]))   # step 2
    groups = standouts(df.assign(leak=df["leak_amt"], expected_charge=df["tariff_charge"]), tot["leaked_pct"], 5)  # step 3

    print("=" * 78); print("REVENUE LEAKAGE - CONCLUSION FROM STEPS 1-5"); print("=" * 78)
    print(f"Step 1  data      : {len(df):,} CDRs, {df['call_start'].min():%Y-%m-%d} to {df['call_start'].max():%Y-%m-%d}")
    print(f"Step 2  totals    : expected {tot['expected_revenue']:,.2f} | leak {tot['leaked_amount']:,.2f} "
          f"({tot['leaked_pct']:.2f}% of revenue) in {tot['leaking_calls']:,} calls")
    print(f"Step 3  stand-outs: " + "; ".join(f"{r.dimension}={r.group} lift {r.lift}" for r in groups.itertuples()))
    print(f"Step 4  detections: {sum(c.kind != 'background' for c in causes)} cause(s) - "
          + ", ".join(f"{c.kind} [{c.event_type}]" for c in causes if c.kind != "background"))
    print(f"Step 5  rules     : {sum(len(c.rules) for c in causes)} tree rule(s) corroborate the detected causes")
    print(f"\nRecoverable       : {s['recoverable'].sum():,.2f} of {s['leaked'].sum():,.2f} "
          f"(re-rate / re-bill within {rebill_days} days as of {as_of:%Y-%m-%d}); "
          f"review / write-off {s['leaked'].sum() - s['recoverable'].sum():,.2f}")

    print("\n--- Causes ranked by leaked amount ---")
    view = s[["cause", "leaking_calls", "leaked", "pct_of_revenue", "recoverable", "first_seen", "last_seen"]].copy()
    view["share_%"] = (100 * view["leaked"] / s["leaked"].sum()).round(1)
    print(view.round(2).to_string(index=False))

    for i, r in s.iterrows():
        c = by_key[r["cause"]]
        print(f"\n{i + 1}. {c.description}")
        print(f"   Leak        : {r['leaked']:,.2f} ({r['leaking_calls']:,} calls, {c.leak_type or 'mixed'}) -> recoverable {r['recoverable']:,.2f}")
        print(f"   Evidence    : {c.evidence}")
        for rule in c.rules:
            print(f"   Tree rule   : {rule}")
        print(f"   Root cause  : [{c.event_type}] {c.event}")
        print(f"   Fix         : {c.fix}")
        print(f"   Daily check : {c.daily_check}")
    return df, causes, s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-e", "--events", default="events.csv")
    parser.add_argument("-n", "--number-calls", type=int, default=20000)
    parser.add_argument("--scenario", action="store_true")
    parser.add_argument("--as-of", help="date recoverability is assessed at (default: day after last call)")
    parser.add_argument("--rebill-days", type=int, default=60)
    args = parser.parse_args()

    if args.input:
        df = pd.read_csv(args.input, parse_dates=["call_start", "call_end"], keep_default_na=False)
    else:
        df = generate_cdrs(args.number_calls, 42, args.scenario)
    as_of = pd.Timestamp(args.as_of) if args.as_of else df["call_start"].max().normalize() + pd.Timedelta(days=1)
    with pd.option_context("display.width", 220, "display.max_columns", None, "display.max_colwidth", 100):
        print_report(df, args.events, as_of, args.rebill_days)


if __name__ == "__main__":
    main()
