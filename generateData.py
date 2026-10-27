"""
Generate synthetic Call Detail Records (CDRs) in a Pandas DataFrame.

A small share of records is deliberately mis-billed (unrated or wrongly rated)
so the data can be used to test revenue-leakage detection.

Usage:
    python generateData.py                 # 1000 calls, printed to stdout
    python generateData.py -n 50000 -o cdrs.csv
"""
import argparse
import math
import random
from datetime import datetime, timedelta

import pandas as pd

FIRST_DAY = datetime(2026, 10, 1)
DAYS = 30

RATE_PER_MIN = {"on_net": 10, "x_net": 15, "international": 100}

# Probability of each billing issue (remaining share is billed correctly)
P_UNRATED = 0.03     # call not charged at all
P_WRONG_RATE = 0.03  # charged at 60% of the correct rate

COLUMNS = [
    "call_id", "a_number", "b_number", "call_start", "call_end", "call_type",
    "duration_sec", "billed_minutes", "expected_charge", "charged", "leak", "issue",
]


def generate_phone_number() -> str:
    return f"+1-{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"


def generate_cdr(call_id: int) -> dict:
    """Build a single CDR as a dict."""
    call_type = random.choice(list(RATE_PER_MIN))
    seconds = random.randint(5, 900)
    billed_minutes = math.ceil(seconds / 60)          # per-minute rounding, as billed

    start = FIRST_DAY + timedelta(seconds=random.randint(0, DAYS * 86400 - 1))
    end = start + timedelta(seconds=seconds)

    expected = billed_minutes * RATE_PER_MIN[call_type]

    charged, issue = expected, ""
    r = random.random()
    if r < P_UNRATED:
        charged, issue = 0, "unrated_usage"
    elif r < P_UNRATED + P_WRONG_RATE:
        charged, issue = round(expected * 0.6, 2), "wrong_rate"

    return {
        "call_id": call_id,
        "a_number": generate_phone_number(),
        "b_number": generate_phone_number(),
        "call_start": start,
        "call_end": end,
        "call_type": call_type,
        "duration_sec": seconds,
        "billed_minutes": billed_minutes,
        "expected_charge": expected,
        "charged": charged,
        "leak": round(expected - charged, 2),
        "issue": issue,
    }


def generate_cdrs(number_calls: int = 1000, seed: int | None = 42) -> pd.DataFrame:
    """Return a DataFrame of `number_calls` CDRs, sorted by call_start."""
    if seed is not None:
        random.seed(seed)
    records = [generate_cdr(i) for i in range(1, number_calls + 1)]
    df = pd.DataFrame.from_records(records, columns=COLUMNS)
    return df.sort_values("call_start").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic CDRs.")
    parser.add_argument("-n", "--number-calls", type=int, default=1000)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-o", "--output", help="optional CSV path to write")
    args = parser.parse_args()

    df = generate_cdrs(args.number_calls, args.seed)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Wrote {len(df)} CDRs to {args.output}")
    else:
        with pd.option_context("display.width", 200, "display.max_columns", None):
            print(df.head(10))

    leaks = df[df["leak"] > 0]
    print(f"\n{len(df)} CDRs, {len(leaks)} with leakage "
          f"(total leak = {leaks['leak'].sum():.2f})")
    print(leaks["issue"].value_counts().to_string())


if __name__ == "__main__":
    main()
