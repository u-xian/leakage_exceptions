"""
Run the whole leakage pipeline in one go (steps 1-7) and write every output
to one folder:

    <outdir>/cdrs.csv               step 1  CDRs used by all later steps
    <outdir>/leakage_charts.png     step 4  charts
    <outdir>/causes.csv             step 6  causes / recoverable amounts
    <outdir>/daily_checks.csv       step 7  every daily check result
    <outdir>/report.txt             everything printed, in order

Usage:
    python runAll.py --scenario                        # generate 20k CDRs with planted incidents, run all
    python runAll.py -i my_cdrs.csv -e changes.csv     # run on real data
    python runAll.py --scenario --no-db                # skip PostgreSQL
    python runAll.py --scenario --steps 1,2,6          # only some steps (1 is always run)
"""
import argparse
import contextlib
import os
import sys
import time
from datetime import timezone, datetime

import pandas as pd

import analyzeGroups
import analyzeLeakage
import chartLeakage
import compareCalls
import dailyChecks
import leakageSummary
from generateData import generate_cdrs

STEPS = {
    1: "generateData    - CDRs",
    2: "analyzeLeakage  - count leaking calls, leaked amount, % of revenue",
    3: "analyzeGroups   - leak by zone / issue / day / hour / weekday",
    4: "chartLeakage    - charts, sudden start / spike / weekly pattern, event matching",
    5: "compareCalls    - leaking vs normal calls, decision-tree rules",
    6: "leakageSummary  - causes, recoverable amount, fix per cause",
    7: "dailyChecks     - daily checks + alerts, loaded into PostgreSQL",
}


class Tee:
    """Print to the console and to report.txt at the same time."""
    def __init__(self, path):
        self.file, self.stdout = open(path, "w"), sys.stdout

    def write(self, s):
        self.stdout.write(s); self.file.write(s)

    def flush(self):
        self.stdout.flush(); self.file.flush()

    def close(self):
        self.file.close()


def banner(n: int) -> None:
    print("\n" + "#" * 78 + f"\n# STEP {n}: {STEPS[n]}\n" + "#" * 78)


def main() -> None:
    p = argparse.ArgumentParser(description="Run the full revenue-leakage pipeline.")
    p.add_argument("-i", "--input", help="existing CDR csv (default: generate)")
    p.add_argument("-n", "--number-calls", type=int, default=20000)
    p.add_argument("-s", "--seed", type=int, default=42)
    p.add_argument("--scenario", action="store_true", help="plant the incidents listed in events.csv when generating")
    p.add_argument("-e", "--events", default="events.csv", help="change / incident log for root-cause matching")
    p.add_argument("-o", "--outdir", default="output")
    p.add_argument("--steps", default="1,2,3,4,5,6,7", help="comma-separated steps to run")
    p.add_argument("--rebill-days", type=int, default=60)
    p.add_argument("--no-db", action="store_true", help="run step 7 checks but do not write to PostgreSQL")
    args = p.parse_args()

    steps = {1} | {int(x) for x in args.steps.split(",") if x.strip()}
    os.makedirs(args.outdir, exist_ok=True)
    tee = Tee(os.path.join(args.outdir, "report.txt"))
    t0 = time.time()

    with contextlib.redirect_stdout(tee), pd.option_context("display.width", 220, "display.max_columns", None,
                                                            "display.max_colwidth", 100):
        print(f"Revenue leakage pipeline - {datetime.now():%Y-%m-%d %H:%M:%S} - steps {sorted(steps)}")

        # ---- 1
        banner(1)
        if args.input:
            df = pd.read_csv(args.input, parse_dates=["call_start", "call_end"], keep_default_na=False)
            print(f"Loaded {len(df):,} CDRs from {args.input}")
        else:
            df = generate_cdrs(args.number_calls, args.seed, args.scenario)
            print(f"Generated {len(df):,} CDRs (seed {args.seed}, scenario={'on' if args.scenario else 'off'})")
        cdr_path = os.path.join(args.outdir, "cdrs.csv")
        df.to_csv(cdr_path, index=False)
        print(f"{df['call_start'].min():%Y-%m-%d} .. {df['call_start'].max():%Y-%m-%d}  ->  {cdr_path}")

        # ---- 2
        if 2 in steps:
            banner(2); analyzeLeakage.print_report(df)

        # ---- 3
        if 3 in steps:
            banner(3); analyzeGroups.print_report(df, top=10)

        # ---- 4
        events = chartLeakage.load_events(args.events if os.path.exists(args.events) else None)
        if 4 in steps:
            banner(4)
            chartLeakage.print_report(df, events, os.path.join(args.outdir, "leakage_charts.png"))

        # ---- 5
        if 5 in steps:
            banner(5); compareCalls.print_report(df, max_depth=6)

        # ---- 6
        causes, summary = None, None
        if 6 in steps or 7 in steps:
            banner(6)
            as_of = df["call_start"].max().normalize() + pd.Timedelta(days=1)
            cdf, causes, summary = leakageSummary.print_report(
                df, args.events if os.path.exists(args.events) else None, as_of, args.rebill_days)
            summary.to_csv(os.path.join(args.outdir, "causes.csv"), index=False)

        # ---- 7
        if 7 in steps:
            banner(7)
            dc = dailyChecks.prepare(df)
            results = []
            for d in sorted(dc["day"].unique()):
                results.extend(dailyChecks.run_checks(dc, pd.Timestamp(d)))
            alerts = [r for r in results if r["status"] == "ALERT"]
            print(f"Ran checks for {dc['day'].nunique()} day(s): {len(results)} results, {len(alerts)} alerts")
            for r in alerts:
                dailyChecks.send_alert(r)
            pd.DataFrame(results).to_csv(os.path.join(args.outdir, "daily_checks.csv"), index=False)
            if args.no_db:
                print("PostgreSQL load skipped (--no-db)")
            else:
                conn = dailyChecks.psycopg2.connect(dailyChecks.DSN)
                try:
                    with conn.cursor() as cur:
                        cur.execute(dailyChecks.DDL)
                    dailyChecks.load_results(conn, results)
                    dailyChecks.load_causes(conn, summary, as_of)
                    conn.commit()
                    print(f"Loaded {len(results)} check results + {len(summary)} causes into PostgreSQL ({dailyChecks.DSN})")
                finally:
                    conn.close()

        print(f"\nDone in {time.time() - t0:.1f}s - outputs in {os.path.abspath(args.outdir)}/")
    tee.close()


if __name__ == "__main__":
    main()
