import csv
import math
import random
import pandas as pd
from datetime import datetime, timedelta

random.seed(42)

FIRST_DAY = datetime(2026, 10, 1)
DAYS = 30
number_calls = 5

rate_per_min = {'on_net': 10, 'x_net': 15, 'international': 100}

def generate_phone_number():
    return f"+1-{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"

rows_records = []

for _ in range(1, number_calls + 1):
    anumber, bnumber = generate_phone_number(), generate_phone_number()
    call_type = random.choice(list(rate_per_min))
    seconds = random.randint(5, 900)
    minutes = math.ceil(seconds / 60)
    start = FIRST_DAY + timedelta(seconds=random.randint(0, DAYS * 86400 - 1))
    end = start + timedelta(seconds=seconds)
    call_start = start.strftime("%Y-%m-%d %H:%M:%S")
    call_end   = end.strftime("%Y-%m-%d %H:%M:%S")
    duration = random.randint(1, 60)  # Duration in minutes
    expected = duration * rate_per_min[call_type]

    charged, issue = expected, ""
    r = random.random()
    if r < 0.03:                                  # 3%: call not charged at all
        charged, issue = 0, "unrated_usage"
    elif r < 0.06:                                # 3%: charged at a lower rate
        charged, issue = round(expected * 0.6, 2), "wrong_rate"

    leak = expected - charged                               # keep only the leaks
    rows_records.append([anumber, bnumber,call_start, call_end, call_type, duration, expected, charged, leak, issue])
df = pd.DataFrame(rows_records)