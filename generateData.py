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

# --scenario: planted incidents, matching the operational log in events.csv
TARIFF_CHANGE_DATE = datetime(2026, 10, 9)          # new international tariff loaded wrong -> wrong_rate from here on
PATCH_WINDOW = (datetime(2026, 10, 17, 2), datetime(2026, 10, 17, 5))   # rating engine down after patch -> unrated
WEEKLY_LOAD_FAIL = {"weekday": 6, "hours": (1, 3)}   # Sunday 01:00-03:00 rating-file load fails -> unrated

COLUMNS = [
    "call_id", "a_number", "b_number", "call_start", "call_end", "call_type",
    "duration_sec", "billed_minutes", "expected_charge", "charged", "leak", "issue",
]


def generate_phone_number() -> str:
    return f"+1-{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"


def planted_issue(start: datetime, call_type: str) -> str | None:
    """Return the issue caused by a planted incident, or None if none applies."""
    if PATCH_WINDOW[0] <= start < PATCH_WINDOW[1]:
        return "unrated_usage"
    if start.weekday() == WEEKLY_LOAD_FAIL["weekday"] and WEEKLY_LOAD_FAIL["hours"][0] <= start.hour < WEEKLY_LOAD_FAIL["hours"][1]:
        return "unrated_usage"
    if call_type == "international" and start >= TARIFF_CHANGE_DATE and random.random() < 0.5:
        return "wrong_rate"
    return None


def generate_cdr(call_id: int, scenario: bool = False) -> dict:
    """Build a single CDR as a dict."""
    call_type = random.choice(list(RATE_PER_MIN))
    seconds = random.randint(5, 900)
    billed_minutes = math.ceil(seconds / 60)          # per-minute rounding, as billed

    start = FIRST_DAY + timedelta(seconds=random.randint(0, DAYS * 86400 - 1))
    end = start + timedelta(seconds=seconds)

    expected = billed_minutes * RATE_PER_MIN[call_type]

    issue = planted_issue(start, call_type) if scenario else None
    if issue is None:                                 # background random leakage
        r = random.random()
        if r < P_UNRATED:
            issue = "unrated_usage"
        elif r < P_UNRATED + P_WRONG_RATE:
            issue = "wrong_rate"

    if issue == "unrated_usage":
        charged = 0
    elif issue == "wrong_rate":
        charged = round(expected * 0.6, 2)
    else:
        charged, issue = expected, ""

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


def generate_cdrs(number_calls: int = 1000, seed: int | None = 42, scenario: bool = False) -> pd.DataFrame:
    """Return a DataFrame of `number_calls` CDRs, sorted by call_start.
    With scenario=True the planted incidents (see events.csv) are applied on top of the background leakage."""
    if seed is not None:
        random.seed(seed)
    records = [generate_cdr(i, scenario) for i in range(1, number_calls + 1)]
    df = pd.DataFrame.from_records(records, columns=COLUMNS)
    return df.sort_values("call_start").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic CDRs.")
    parser.add_argument("-n", "--number-calls", type=int, default=1000)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-o", "--output", help="optional CSV path to write")
    parser.add_argument("--scenario", action="store_true", help="plant the incidents listed in events.csv")
    args = parser.parse_args()

    df = generate_cdrs(args.number_calls, args.seed, args.scenario)

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
