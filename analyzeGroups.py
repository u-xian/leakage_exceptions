"""
Where does the leakage come from?

Groups the leak by zone (call_type), issue type, day, hour and weekday,
plus a few combined views, and ranks the groups by leaked amount.
A "lift" column compares each group's loss rate with the overall loss
rate, so a group that stands out (lift >> 1) is a first clue to the cause.

Usage:
    python analyzeGroups.py                # generate 1000 CDRs and analyse
    python analyzeGroups.py -n 50000
    python analyzeGroups.py -i cdrs.csv
"""
import argparse

import pandas as pd

from analyzeLeakage import leakage_by, summarize_leakage
from generateData import generate_cdrs

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def add_time_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """Add zone / day / hour / weekday columns derived from existing data."""
    df = df.copy()
    df["zone"] = df["call_type"]                      # call_type is the rating zone
    df["day"] = df["call_start"].dt.date
    df["hour"] = df["call_start"].dt.hour
    df["weekday"] = pd.Categorical(
        df["call_start"].dt.day_name().str[:3], categories=WEEKDAYS, ordered=True
    )
    return df


def leakage_table(df: pd.DataFrame, keys, overall_pct: float, top: int | None = None) -> pd.DataFrame:
    """Per-group leakage with a lift column (group loss rate / overall loss rate)."""
    t = leakage_by(df, keys)
    t["lift"] = (t["leaked_pct"] / overall_pct).round(2) if overall_pct else 0.0
    return t.head(top) if top else t


def print_section(title: str, table: pd.DataFrame) -> None:
    print(f"\n--- {title} ---")
    print(table.to_string())


def print_report(df: pd.DataFrame, top: int = 10) -> None:
    df = add_time_dimensions(df)
    s = summarize_leakage(df)
    overall = s["leaked_pct"]

    print("=" * 70)
    print(f"LEAKAGE BY GROUP  ({s['total_calls']:,} calls, "
          f"{s['leaking_calls']:,} leaking, {s['leaked_amount']:,.2f} lost = {overall:.2f}% of revenue)")
    print("=" * 70)
    print("lift = group's leaked % of revenue / overall leaked %  (1.0 = average)")

    # single dimensions
    print_section("By zone", leakage_table(df, "zone", overall))
    print_section("By issue type (leaking calls only)",
                  leakage_by(df[df["leak"] > 0], "issue")[["leaking_calls", "leaked_amount"]])
    print_section("By weekday", leakage_table(df, "weekday", overall).sort_index())
    print_section("By hour of day", leakage_table(df, "hour", overall).sort_index())
    print_section(f"By day (top {top} by leaked amount)", leakage_table(df, "day", overall, top))

    # combined dimensions - where a single cause usually shows up
    print_section(f"By zone x issue", leakage_by(df[df["leak"] > 0], ["zone", "issue"])[["leaking_calls", "leaked_amount"]])
    print_section(f"By zone x weekday x hour (top {top})", leakage_table(df, ["zone", "weekday", "hour"], overall, top))
    print_section(f"By day x hour (top {top})", leakage_table(df, ["day", "hour"], overall, top))

    # what stands out most: groups with meaningful volume ranked by lift
    print_section("Groups that stand out (lift, min 20 calls)", standouts(df, overall, top))


def standouts(df: pd.DataFrame, overall: float, top: int = 10, min_calls: int = 20) -> pd.DataFrame:
    """Rank every single-dimension group by lift, ignoring tiny groups."""
    frames = []
    for dim in ["zone", "weekday", "hour", "day"]:
        t = leakage_table(df, dim, overall).reset_index()
        t["dimension"] = dim
        t["group"] = t[dim].astype(str)
        frames.append(t[["dimension", "group", "calls", "leaking_calls", "leaked_amount", "leaked_pct", "lift"]])
    allg = pd.concat(frames, ignore_index=True)
    allg = allg[allg["calls"] >= min_calls]
    return allg.sort_values("lift", ascending=False).head(top).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Group revenue leakage by zone / issue / time.")
    parser.add_argument("-i", "--input", help="CSV produced by generateData.py (default: generate in-memory)")
    parser.add_argument("-n", "--number-calls", type=int, default=1000)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-t", "--top", type=int, default=10)
    args = parser.parse_args()

    if args.input:
        df = pd.read_csv(args.input, parse_dates=["call_start", "call_end"], keep_default_na=False)
    else:
        df = generate_cdrs(args.number_calls, args.seed)

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print_report(df, args.top)


if __name__ == "__main__":
    main()
