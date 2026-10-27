"""
Chart the leak per day / per hour and match anomalies with known events.

Detects three signatures in the daily / hourly leak-rate series:
  * sudden start  - a step change (leak rate before vs after a date)
  * spike         - a single day or day-hour far above its neighbours
  * weekly pattern- one weekday x hour slot consistently above the rest
and matches each finding against an operational event log (events.csv).

Usage:
     python chartLeakage.py -i cdrs.csv -e events.csv -o leakage_charts
    python chartLeakage.py -n 20000 --scenario        # generate in-memory
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyzeGroups import WEEKDAYS, add_time_dimensions
from generateData import generate_cdrs

# ---------------------------------------------------------------- series


def daily_series(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("day")
    out = pd.DataFrame({"expected": g["expected_charge"].sum(), "leaked": g["leak"].sum(),
                        "leaking_calls": g["leak"].apply(lambda s: int((s > 0).sum()))})
    out.index = pd.to_datetime(out.index)
    out["leak_pct"] = 100 * out["leaked"] / out["expected"]
    return out


def hourly_series(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("hour")
    out = pd.DataFrame({"expected": g["expected_charge"].sum(), "leaked": g["leak"].sum()})
    out["leak_pct"] = 100 * out["leaked"] / out["expected"]
    return out


def pivot_pct(df: pd.DataFrame, rows: str, cols: str) -> pd.DataFrame:
    exp = df.pivot_table(index=rows, columns=cols, values="expected_charge", aggfunc="sum", observed=True)
    lk = df.pivot_table(index=rows, columns=cols, values="leak", aggfunc="sum", observed=True)
    return (100 * lk / exp).reindex(columns=range(24)) if cols == "hour" else 100 * lk / exp


# ---------------------------------------------------------------- detection


def detect_step(daily: pd.DataFrame, min_days: int = 3) -> dict | None:
    """Best single split of the daily leak% series into before/after (largest t-statistic)."""
    y = daily["leak_pct"].to_numpy()
    best = None
    for k in range(min_days, len(y) - min_days + 1):
        a, b = y[:k], y[k:]
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        t = (b.mean() - a.mean()) / se if se else 0
        if best is None or abs(t) > abs(best["t"]):
            best = {"date": daily.index[k], "before": a.mean(), "after": b.mean(), "t": t}
    return best


def robust_z(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median() or 1e-9
    return (s - med) / (1.4826 * mad)


def detect_spikes(daily: pd.DataFrame, z_thresh: float = 3.5) -> pd.DataFrame:
    """Days whose leak% is far above the median (robust z-score), after removing any step."""
    s = daily["leak_pct"].copy()
    step = detect_step(daily)
    if step and abs(step["t"]) > 4:                     # de-trend around the step so the step itself is not a 'spike'
        s[s.index >= step["date"]] -= step["after"] - step["before"]
    z = robust_z(s)
    out = daily.assign(z=z.round(2))[z > z_thresh]
    return out.sort_values("z", ascending=False)


def detect_hour_spikes(df: pd.DataFrame, z_thresh: float = 3.5, min_calls: int = 5) -> pd.DataFrame:
    """Day x hour cells far above the typical cell (robust z on leak%)."""
    g = df.groupby(["day", "hour"])
    cell = pd.DataFrame({"calls": g.size(), "expected": g["expected_charge"].sum(), "leaked": g["leak"].sum()})
    cell["leak_pct"] = 100 * cell["leaked"] / cell["expected"]
    cell = cell[cell["calls"] >= min_calls]
    cell["z"] = robust_z(cell["leak_pct"]).round(2)
    return cell[cell["z"] > z_thresh].sort_values("z", ascending=False)


def detect_weekly(df: pd.DataFrame, ratio_thresh: float = 3.0, min_calls: int = 20, min_pct: float = 25.0) -> pd.DataFrame:
    """Weekday x hour slots whose leak% is >= ratio_thresh x the overall leak% AND >= min_pct
    (a batch fault leaves a large share of the slot unrated; 3x a tiny base rate is still noise)."""
    g = df.groupby(["weekday", "hour"], observed=True)
    slot = pd.DataFrame({"calls": g.size(), "expected": g["expected_charge"].sum(), "leaked": g["leak"].sum(),
                         "weeks_hit": g["day"].apply(lambda d: d[df.loc[d.index, "leak"] > 0].nunique())})
    slot["leak_pct"] = 100 * slot["leaked"] / slot["expected"]
    overall = 100 * df["leak"].sum() / df["expected_charge"].sum()
    slot["ratio"] = (slot["leak_pct"] / overall).round(2)
    return slot[(slot["calls"] >= min_calls) & (slot["ratio"] >= ratio_thresh) & (slot["leak_pct"] >= min_pct)].sort_values("ratio", ascending=False)


# ---------------------------------------------------------------- event matching


def load_events(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["ts", "event_type", "description"])
    ev = pd.read_csv(path)
    ev["ts"] = pd.to_datetime(ev["date"].astype(str) + " " + ev.get("time", "00:00").astype(str))
    return ev[["ts", "event_type", "description"]].sort_values("ts")


def events_near(events: pd.DataFrame, ts: pd.Timestamp, window_h: float = 24) -> pd.DataFrame:
    """Events within `window_h` hours *before* the anomaly (a cause precedes its effect), plus same day."""
    lo = ts - pd.Timedelta(hours=window_h)
    return events[(events["ts"] >= lo) & (events["ts"] <= ts + pd.Timedelta(hours=1))]


def fmt_events(ev: pd.DataFrame) -> str:
    if ev.empty:
        return "no known event in window"
    return "; ".join(f"{r.ts:%Y-%m-%d %H:%M} {r.event_type}: {r.description}" for r in ev.itertuples())


# ---------------------------------------------------------------- charts


def plot(df: pd.DataFrame, daily: pd.DataFrame, hourly: pd.DataFrame, events: pd.DataFrame,
         step: dict | None, spikes: pd.DataFrame, out: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    colors = {"tariff_change": "tab:red", "software_patch": "tab:purple", "failed_file_load": "tab:orange"}

    # 1. leak per day
    ax = axes[0, 0]
    ax.bar(daily.index, daily["leak_pct"], color="tab:blue", alpha=0.8, label="leak % of expected revenue")
    ax.plot(daily.index, daily["leak_pct"].rolling(7, center=True, min_periods=3).mean(), color="k", lw=1.5, label="7-day mean")
    for r in events.itertuples():
        ax.axvline(r.ts, color=colors.get(r.event_type, "gray"), ls="--", lw=1.2,
                   label=r.event_type if r.event_type not in ax.get_legend_handles_labels()[1] else None)
    if step and abs(step["t"]) > 4:
        ax.axvline(step["date"], color="tab:red", lw=3, alpha=0.3)
        ax.annotate(f"step: {step['before']:.1f}% -> {step['after']:.1f}%", (step["date"], daily["leak_pct"].max()),
                    xytext=(5, -15), textcoords="offset points", color="tab:red")
    for d in spikes.index:
        ax.annotate("spike", (d, daily.loc[d, "leak_pct"]), xytext=(0, 5), textcoords="offset points", ha="center", color="tab:red")
    ax.set_title("Leak per day (% of expected revenue)"); ax.legend(fontsize=8); ax.tick_params(axis="x", rotation=45)

    # 2. leak per hour of day
    ax = axes[0, 1]
    ax.bar(hourly.index, hourly["leak_pct"], color="tab:green", alpha=0.8)
    ax.axhline(100 * df["leak"].sum() / df["expected_charge"].sum(), color="k", ls=":", label="overall")
    ax.set_xticks(range(24)); ax.set_xlabel("hour of day"); ax.set_title("Leak per hour of day (% of expected revenue)"); ax.legend()

    # 3. weekday x hour heatmap
    ax = axes[1, 0]
    wk = pivot_pct(df, "weekday", "hour").reindex(WEEKDAYS)
    im = ax.imshow(wk.to_numpy(), aspect="auto", cmap="Reds")
    ax.set_yticks(range(7), WEEKDAYS); ax.set_xticks(range(24)); ax.set_xlabel("hour"); ax.set_title("Weekday x hour leak % (weekly pattern)")
    fig.colorbar(im, ax=ax, fraction=0.03)

    # 4. day x hour heatmap
    ax = axes[1, 1]
    dh = pivot_pct(df, "day", "hour")
    im = ax.imshow(dh.to_numpy(), aspect="auto", cmap="Reds")
    ax.set_yticks(range(len(dh.index)), [str(d)[5:] for d in dh.index], fontsize=7)
    ax.set_xticks(range(24)); ax.set_xlabel("hour"); ax.set_title("Day x hour leak % (spikes / sudden start)")
    fig.colorbar(im, ax=ax, fraction=0.03)

    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


# ---------------------------------------------------------------- report


def print_report(df: pd.DataFrame, events: pd.DataFrame, out: str) -> None:
    df = add_time_dimensions(df)
    daily, hourly = daily_series(df), hourly_series(df)
    step, spikes = detect_step(daily), detect_spikes(daily)
    hour_spikes, weekly = detect_hour_spikes(df), detect_weekly(df)

    print("=" * 70); print("LEAKAGE TIMELINE ANALYSIS"); print("=" * 70)
    print(f"Overall leak: {100 * df['leak'].sum() / df['expected_charge'].sum():.2f}% of expected revenue, "
          f"{len(daily)} days, {len(events)} known events loaded")

    print("\n--- 1. Sudden start (step change in daily leak %) ---")
    if step and abs(step["t"]) > 4:
        print(f"Step on {step['date']:%Y-%m-%d}: {step['before']:.2f}% before -> {step['after']:.2f}% after (t={step['t']:.1f})")
        after = df[df["call_start"] >= step["date"]]; before = df[df["call_start"] < step["date"]]
        by_zone = pd.DataFrame({"before_pct": 100 * before.groupby("zone")["leak"].sum() / before.groupby("zone")["expected_charge"].sum(),
                                "after_pct": 100 * after.groupby("zone")["leak"].sum() / after.groupby("zone")["expected_charge"].sum()}).round(2)
        print(by_zone.to_string())
        print("  matches ->", fmt_events(events_near(events, step["date"], 36)))
    else:
        print(f"No significant step (best candidate {step['date']:%Y-%m-%d}, t={step['t']:.1f})" if step else "n/a")

    print("\n--- 2. Spikes (days with robust z > 3.5) ---")
    if spikes.empty:
        print("No daily spikes")
    for d, r in spikes.iterrows():
        print(f"{d:%Y-%m-%d}: {r.leak_pct:.1f}% leaked, {int(r.leaking_calls)} leaking calls (z={r.z})")
        hrs = hour_spikes.loc[hour_spikes.index.get_level_values("day") == d.date()]
        if not hrs.empty:
            print("   hours: " + ", ".join(f"{h:02d}:00 ({p:.0f}%)" for (_, h), p in hrs["leak_pct"].items()))
        print("   matches ->", fmt_events(events_near(events, d + pd.Timedelta(hours=23), 36)))

    print("\n--- 3. Weekly pattern (weekday x hour slots >= 3x overall leak %) ---")
    if weekly.empty:
        print("No recurring weekly slot")
    else:
        print(weekly[["calls", "leak_pct", "ratio", "weeks_hit"]].round(1).to_string())
        for (wd, h) in weekly.index:
            same = events[(events["ts"].dt.day_name().str[:3] == wd)]
            print(f"   {wd} {h:02d}:00 matches -> {fmt_events(same) if not same.empty else 'no recurring event on this weekday'}")

    print("\n--- 4. Every known event: leak % in the 24h before vs 24h after ---")
    for r in events.itertuples():
        b = df[(df["call_start"] >= r.ts - pd.Timedelta(hours=24)) & (df["call_start"] < r.ts)]
        a = df[(df["call_start"] >= r.ts) & (df["call_start"] < r.ts + pd.Timedelta(hours=24))]
        pb = 100 * b["leak"].sum() / b["expected_charge"].sum() if len(b) else float("nan")
        pa = 100 * a["leak"].sum() / a["expected_charge"].sum() if len(a) else float("nan")
        flag = "  <-- impact" if pa > 2 * pb else ""
        print(f"{r.ts:%Y-%m-%d %H:%M} {r.event_type:17s} before {pb:5.1f}%  after {pa:5.1f}%{flag}   {r.description}")

    plot(df, daily, hourly, events, step, spikes, out)
    print(f"\nCharts written to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chart leakage over time and match with known events.")
    parser.add_argument("-i", "--input", help="CSV produced by generateData.py")
    parser.add_argument("-e", "--events", default="events.csv", help="known-events CSV (date,time,event_type,description)")
    parser.add_argument("-o", "--output", default="leakage_charts.png")
    parser.add_argument("-n", "--number-calls", type=int, default=20000)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("--scenario", action="store_true")
    args = parser.parse_args()

    if args.input:
        df = pd.read_csv(args.input, parse_dates=["call_start", "call_end"], keep_default_na=False)
    else:
        df = generate_cdrs(args.number_calls, args.seed, args.scenario)

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print_report(df, load_events(args.events), args.output)


if __name__ == "__main__":
    main()
