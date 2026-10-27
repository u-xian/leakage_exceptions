"""
Revenue-leakage analysis on the synthetic CDRs.

Counts the leaking calls, sums the leaked amount and expresses it as a
percentage of the total expected revenue, with breakdowns by issue type
and call type.

Usage:
    python analyzeLeakage.py                # generate 1000 CDRs and analyse
    python analyzeLeakage.py -n 50000       # bigger sample
    python analyzeLeakage.py -i cdrs.csv    # analyse an existing CSV
"""
import argparse

import pandas as pd

from generateData import generate_cdrs


def summarize_leakage(df: pd.DataFrame) -> dict:
    """Headline figures for the whole dataset."""
    leaks = df[df["leak"] > 0]
    expected_total = df["expected_charge"].sum()
    leaked_total = leaks["leak"].sum()
    return {
        "total_calls": len(df),
        "leaking_calls": len(leaks),
        "leaking_calls_pct": 100 * len(leaks) / len(df) if len(df) else 0.0,
        "expected_revenue": expected_total,
        "charged_revenue": df["charged"].sum(),
        "leaked_amount": leaked_total,
        "leaked_pct": 100 * leaked_total / expected_total if expected_total else 0.0,
    }


def leakage_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Leaking calls / leaked amount / % of that group's expected revenue, per `column`."""
    grouped = df.groupby(column)
    out = pd.DataFrame({
        "calls": grouped.size(),
        "leaking_calls": grouped["leak"].apply(lambda s: int((s > 0).sum())),
        "expected_revenue": grouped["expected_charge"].sum(),
        "leaked_amount": grouped["leak"].sum(),
    })
    out["leaked_pct"] = (100 * out["leaked_amount"] / out["expected_revenue"]).round(2)
    return out.sort_values("leaked_amount", ascending=False)


def print_report(df: pd.DataFrame) -> None:
    s = summarize_leakage(df)

    print("=" * 60)
    print("REVENUE LEAKAGE REPORT")
    print("=" * 60)
    print(f"Total calls             : {s['total_calls']:>12,}")
    print(f"Leaking calls           : {s['leaking_calls']:>12,}  ({s['leaking_calls_pct']:.2f}% of calls)")
    print(f"Expected revenue        : {s['expected_revenue']:>12,.2f}")
    print(f"Actually charged        : {s['charged_revenue']:>12,.2f}")
    print(f"Leaked amount           : {s['leaked_amount']:>12,.2f}")
    print(f"Revenue lost            : {s['leaked_pct']:>11.2f}%")

    leaks = df[df["leak"] > 0]
    print("\n--- Leakage by issue type ---")
    by_issue = leakage_by(leaks, "issue")[["leaking_calls", "leaked_amount"]]
    by_issue["share_of_leak_pct"] = (100 * by_issue["leaked_amount"] / s["leaked_amount"]).round(2)
    print(by_issue.to_string())

    print("\n--- Leakage by call type ---")
    print(leakage_by(df, "call_type").to_string())

    print("\n--- Top 5 leaking calls ---")
    cols = ["call_id", "a_number", "call_start", "call_type", "expected_charge", "charged", "leak", "issue"]
    print(leaks.nlargest(5, "leak")[cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse revenue leakage in CDRs.")
    parser.add_argument("-i", "--input", help="CSV produced by generateData.py (default: generate in-memory)")
    parser.add_argument("-n", "--number-calls", type=int, default=1000)
    parser.add_argument("-s", "--seed", type=int, default=42)
    args = parser.parse_args()

    if args.input:
        df = pd.read_csv(args.input, parse_dates=["call_start", "call_end"], keep_default_na=False)
    else:
        df = generate_cdrs(args.number_calls, args.seed)

    print_report(df)


if __name__ == "__main__":
    main()
