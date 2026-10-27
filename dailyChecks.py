"""
Daily automatic leakage checks with alerts, persisted to PostgreSQL.

One generic check per root cause found in the analysis (no hard-coded dates):

  unrated_usage_hourly        share of CDRs charged 0 per hour         -> rating outage / failed file load
  effective_rate_by_zone      share of CDRs charged below tariff, per zone -> wrong tariff loaded
  daily_leak_pct_vs_baseline  today's leak % vs median of previous 7 days -> anything new / drift
  weekly_slot_recurrence      an hour that alerted today AND the same weekday-hour in earlier weeks -> recurring batch fault

Results (OK and ALERT) are upserted into leakage_daily_checks; alerts are also
printed / handed to send_alert() where email / Slack / PagerDuty can be wired.

Usage:
    python dailyChecks.py -i cdrs.csv --date 2026-10-17          # one day
    python dailyChecks.py -i cdrs.csv --backfill                  # every day in the file
    python dailyChecks.py -i cdrs.csv --date yesterday --no-db    # dry run
Schedule (cron, 01:05 every day, after the nightly CDR load):
    5 1 * * *  cd /opt/leakage && python dailyChecks.py -i /data/cdrs_latest.csv --date yesterday

Connection: env LEAKAGE_DSN, default postgresql://postgres@localhost:5432/leakage
"""
import argparse
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from generateData import RATE_PER_MIN, generate_cdrs
from leakageSummary import classify, summarize

DSN = os.environ.get("LEAKAGE_DSN", "postgresql://postgres@localhost:5432/leakage")

THRESHOLDS = {
    "unrated_usage_hourly": 0.30,        # >= 30% of an hour's calls charged 0
    "effective_rate_by_zone": 0.15,      # >= 15% of a zone's calls charged below tariff (background is ~3%)
    "daily_leak_pct_vs_baseline": 2.0,   # today's leak % >= 2x the 7-day median (and >= 3 points above it)
    "weekly_slot_recurrence": 2,         # same weekday-hour alerted in >= 2 of the last 4 weeks
}
MIN_CALLS = 5

DDL = """
CREATE TABLE IF NOT EXISTS leakage_daily_checks (
    check_date     date        NOT NULL,
    check_name     text        NOT NULL,
    scope          text        NOT NULL,
    metric         text        NOT NULL,
    value          numeric(14,4),
    threshold      numeric(14,4),
    status         text        NOT NULL CHECK (status IN ('OK', 'ALERT', 'NO_BASELINE')),
    calls          integer,
    leaked_amount  numeric(14,2),
    message        text,
    run_at         timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (check_date, check_name, scope)
);
CREATE INDEX IF NOT EXISTS leakage_daily_checks_alert_idx ON leakage_daily_checks (status, check_date);
CREATE OR REPLACE VIEW leakage_alerts AS
    SELECT * FROM leakage_daily_checks WHERE status = 'ALERT' ORDER BY check_date DESC, leaked_amount DESC;

CREATE TABLE IF NOT EXISTS leakage_causes (
    as_of            date        NOT NULL,
    cause            text        NOT NULL,
    description      text,
    leaking_calls    integer,
    leaked           numeric(14,2),
    pct_of_revenue   numeric(8,4),
    in_rebill_window numeric(14,2),
    recovery_rate    numeric(5,2),
    recoverable      numeric(14,2),
    first_seen       timestamp,
    last_seen        timestamp,
    daily_check      text,
    run_at           timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (as_of, cause)
);
"""


# ---------------------------------------------------------------- checks


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tariff_charge"] = df["billed_minutes"] * df["call_type"].map(RATE_PER_MIN)
    df["leak_amt"] = df["tariff_charge"] - df["charged"]
    df["unrated"] = (df["charged"] == 0) & (df["tariff_charge"] > 0)
    df["under_rated"] = (df["charged"] > 0) & (df["charged"] < df["tariff_charge"])
    df["day"] = df["call_start"].dt.normalize()
    df["hour"] = df["call_start"].dt.hour
    return df


def row(check_date, name, scope, metric, value, thr, status, calls, leaked, msg) -> dict:
    return {"check_date": check_date.date(), "check_name": name, "scope": scope, "metric": metric,
            "value": None if value is None else round(float(value), 4), "threshold": thr, "status": status,
            "calls": int(calls), "leaked_amount": round(float(leaked), 2), "message": msg}


def check_unrated_usage_hourly(today: pd.DataFrame, check_date) -> list[dict]:
    thr, out = THRESHOLDS["unrated_usage_hourly"], []
    for h, g in today.groupby("hour"):
        share = g["unrated"].mean() if len(g) else 0.0
        alert = len(g) >= MIN_CALLS and share >= thr
        out.append(row(check_date, "unrated_usage_hourly", f"hour={h:02d}", "unrated_share", share, thr,
                       "ALERT" if alert else "OK", len(g), g.loc[g["unrated"], "leak_amt"].sum(),
                       f"{int(g['unrated'].sum())}/{len(g)} calls at {h:02d}:00 left unrated" if alert else ""))
    return out


def check_effective_rate_by_zone(today: pd.DataFrame, check_date) -> list[dict]:
    thr, out = THRESHOLDS["effective_rate_by_zone"], []
    for z, g in today.groupby("call_type"):
        rated = g[g["charged"] > 0]
        share = rated["under_rated"].mean() if len(rated) else 0.0
        eff = (rated["charged"] / rated["billed_minutes"]).median() if len(rated) else None
        alert = len(rated) >= MIN_CALLS and share >= thr
        msg = (f"{share:.0%} of {z} calls charged below tariff {RATE_PER_MIN[z]}/min "
               f"(median effective rate {eff:.1f}/min) - check tariff table") if alert else ""
        out.append(row(check_date, "effective_rate_by_zone", f"zone={z}", "under_rated_share", share, thr,
                       "ALERT" if alert else "OK", len(rated), rated.loc[rated["under_rated"], "leak_amt"].sum(), msg))
    return out


def check_daily_leak_vs_baseline(df: pd.DataFrame, today: pd.DataFrame, check_date) -> list[dict]:
    thr = THRESHOLDS["daily_leak_pct_vs_baseline"]
    pct = 100 * today["leak_amt"].sum() / today["tariff_charge"].sum() if today["tariff_charge"].sum() else 0.0
    hist = df[(df["day"] < check_date) & (df["day"] >= check_date - pd.Timedelta(days=7))]
    daily = hist.groupby("day").apply(lambda g: 100 * g["leak_amt"].sum() / g["tariff_charge"].sum(), include_groups=False)
    if len(daily) < 3:
        return [row(check_date, "daily_leak_pct_vs_baseline", "all", "leak_pct", pct, None, "NO_BASELINE",
                    len(today), today["leak_amt"].sum(), f"leak {pct:.1f}% - fewer than 3 prior days for a baseline")]
    base = daily.median()
    alert = pct >= thr * base and pct >= base + 3
    return [row(check_date, "daily_leak_pct_vs_baseline", "all", "leak_pct_vs_7d_median", pct / base if base else None, thr,
                "ALERT" if alert else "OK", len(today), today["leak_amt"].sum(),
                f"leak {pct:.1f}% of revenue vs 7-day median {base:.1f}%" if alert else "")]


def check_weekly_recurrence(df: pd.DataFrame, check_date, hourly_results: list[dict]) -> list[dict]:
    """For each hour that alerted today, test the same weekday-hour in the previous 4 weeks."""
    thr, out = THRESHOLDS["weekly_slot_recurrence"], []
    for r in hourly_results:
        if r["status"] != "ALERT":
            continue
        h = int(r["scope"].split("=")[1])
        hits, weeks = [], 0
        for w in range(1, 5):
            d = check_date - pd.Timedelta(weeks=w)
            g = df[(df["day"] == d) & (df["hour"] == h)]
            if len(g) >= MIN_CALLS:
                weeks += 1
                if g["unrated"].mean() >= THRESHOLDS["unrated_usage_hourly"]:
                    hits.append(d.strftime("%m-%d"))
        status = "ALERT" if len(hits) >= thr - 1 else "OK"       # today + (thr-1) earlier weeks
        wd = check_date.strftime("%a")
        out.append(row(check_date, "weekly_slot_recurrence", f"weekday={wd} hour={h:02d}", "prior_weeks_hit", len(hits) + 1, thr,
                       status, r["calls"], r["leaked_amount"],
                       f"RECURRING: {wd} {h:02d}:00 unrated in {len(hits) + 1} of last {weeks + 1} weeks ({', '.join(hits)}, today) "
                       f"- scheduled job / batch fault" if status == "ALERT" else f"first occurrence in {weeks + 1} weeks"))
    return out


def run_checks(df: pd.DataFrame, check_date: pd.Timestamp) -> list[dict]:
    today = df[df["day"] == check_date]
    if today.empty:
        return []
    hourly = check_unrated_usage_hourly(today, check_date)
    return (hourly + check_effective_rate_by_zone(today, check_date)
            + check_daily_leak_vs_baseline(df, today, check_date) + check_weekly_recurrence(df, check_date, hourly))


# ---------------------------------------------------------------- alerts + db


def send_alert(r: dict) -> None:
    """Alert sink. Replace / extend with email, Slack webhook, PagerDuty, ticket creation..."""
    print(f"  ALERT {r['check_date']} {r['check_name']:<28} {r['scope']:<28} leaked {r['leaked_amount']:>10,.2f}  {r['message']}")


def load_results(conn, results: list[dict]) -> None:
    cols = ["check_date", "check_name", "scope", "metric", "value", "threshold", "status", "calls", "leaked_amount", "message"]
    sql = f"""INSERT INTO leakage_daily_checks ({', '.join(cols)}, run_at) VALUES %s
              ON CONFLICT (check_date, check_name, scope) DO UPDATE SET
              {', '.join(f'{c} = EXCLUDED.{c}' for c in cols[3:])}, run_at = EXCLUDED.run_at"""
    now = datetime.now(timezone.utc)
    with conn.cursor() as cur:
        execute_values(cur, sql, [tuple(r[c] for c in cols) + (now,) for r in results])


def load_causes(conn, summary: pd.DataFrame, as_of) -> None:
    cols = ["cause", "description", "leaking_calls", "leaked", "pct_of_revenue", "in_rebill_window",
            "recovery_rate", "recoverable", "first_seen", "last_seen", "daily_check"]
    sql = f"""INSERT INTO leakage_causes (as_of, {', '.join(cols)}, run_at) VALUES %s
              ON CONFLICT (as_of, cause) DO UPDATE SET
              {', '.join(f'{c} = EXCLUDED.{c}' for c in cols[1:])}, run_at = EXCLUDED.run_at"""
    now = datetime.now(timezone.utc)
    rows = [(as_of.date(),) + tuple(r[c].to_pydatetime() if isinstance(r[c], pd.Timestamp) else
                                    (None if pd.isna(r[c]) else r[c].item() if hasattr(r[c], "item") else r[c])
                                    for c in cols) + (now,) for _, r in summary.iterrows()]
    with conn.cursor() as cur:
        execute_values(cur, sql, rows)


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-n", "--number-calls", type=int, default=20000)
    parser.add_argument("--scenario", action="store_true")
    parser.add_argument("--date", default="yesterday", help="YYYY-MM-DD or 'yesterday'")
    parser.add_argument("--backfill", action="store_true", help="run every day present in the data")
    parser.add_argument("--causes", action="store_true", help="also (re)load the leakage_causes summary table")
    parser.add_argument("-e", "--events", default="events.csv", help="change log used for root-cause matching")
    parser.add_argument("--no-db", action="store_true")
    args = parser.parse_args()

    if args.input:
        raw = pd.read_csv(args.input, parse_dates=["call_start", "call_end"], keep_default_na=False)
    else:
        raw = generate_cdrs(args.number_calls, 42, args.scenario)
    df = prepare(raw)

    if args.backfill:
        dates = sorted(df["day"].unique())
    elif args.date == "yesterday":
        dates = [pd.Timestamp(datetime.now().date() - timedelta(days=1))]
    else:
        dates = [pd.Timestamp(args.date)]

    results = []
    for d in dates:
        results.extend(run_checks(df, pd.Timestamp(d)))
    alerts = [r for r in results if r["status"] == "ALERT"]

    print(f"Ran checks for {len(dates)} day(s): {len(results)} results, {len(alerts)} alerts")
    for r in alerts:
        send_alert(r)

    if args.no_db:
        return
    conn = psycopg2.connect(DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(DDL)
        if results:
            load_results(conn, results)
        if args.causes:
            cdf, causes = classify(raw, args.events)
            as_of = cdf["call_start"].max().normalize() + pd.Timedelta(days=1)
            load_causes(conn, summarize(cdf, causes, as_of, 60), as_of)
        conn.commit()
        print(f"Loaded {len(results)} check results{' + causes summary' if args.causes else ''} into PostgreSQL ({DSN})")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
