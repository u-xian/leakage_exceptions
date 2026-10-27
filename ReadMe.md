# leakage_exceptions - revenue leakage detection on CDRs

Find calls that were billed less than the tariff says, work out why, and catch
the same fault automatically next time.

## Quick start - one command

```bash
pip install -r requirements.txt

python runAll.py --scenario                     # synthetic 20k CDRs with 3 planted incidents, all 7 steps
python runAll.py -i my_cdrs.csv -e changes.csv  # your own CDR extract + change/incident log
python runAll.py --scenario --no-db             # everything except the PostgreSQL load
python runAll.py --scenario --steps 2,6         # only some steps (step 1 always runs)
```

`runAll.py` runs every step below in order on the _same_ data and writes to `output/`:

| file                        | from   | content                                                      |
| --------------------------- | ------ | ------------------------------------------------------------ |
| `output/report.txt`         | all    | everything printed, step by step                             |
| `output/cdrs.csv`           | step 1 | the CDRs used                                                |
| `output/leakage_charts.png` | step 4 | leak per day / hour, weekday x hour and day x hour heat-maps |
| `output/causes.csv`         | step 6 | causes, leaked and recoverable amount, daily check per cause |
| `output/daily_checks.csv`   | step 7 | every daily check result (OK / ALERT)                        |

Options: `-n` number of calls, `-s` seed, `-o` output folder, `--rebill-days` (recoverability window, default 60).

## The steps

Each script can still be run alone (same `-i cdrs.csv` argument); `runAll.py` just chains them.

| step | script              | what it does                                                                                                                                  |
| ---- | ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `generateData.py`   | synthetic CDRs in a DataFrame; `--scenario` plants the incidents listed in `events.csv` on top of 3% unrated + 3% wrong-rate background noise |
| 2    | `analyzeLeakage.py` | count leaking calls, sum the leak, % of expected revenue                                                                                      |
| 3    | `analyzeGroups.py`  | leak by zone / issue / day / hour / weekday with lift vs overall                                                                              |
| 4    | `chartLeakage.py`   | charts; detects sudden start (step), spikes, weekly pattern; matches dates with `events.csv`                                                  |
| 5    | `compareCalls.py`   | leaking vs normal calls (duration, zone, time of day); decision-tree rules                                                                    |
| 6    | `leakageSummary.py` | builds the conclusion from steps 2-5: causes, recoverable amount, fix and daily check per cause                                               |
| 7    | `dailyChecks.py`    | one generic check per cause kind, alerts, results upserted into PostgreSQL                                                                    |

`events.csv` is the change / incident log used for root-cause matching
(`date,time,event_type,description`; event types `tariff_change`, `software_patch`, `failed_file_load`).

## Daily checks in production

After the first full run, only step 7 needs to run every day:

```bash
export LEAKAGE_DSN=postgresql://user:pw@dbhost:5432/leakage
# cron - 01:05 every day, after the nightly CDR load
5 1 * * *  cd /opt/leakage && python dailyChecks.py -i /data/cdrs_latest.csv --date yesterday
```

| check                        | catches                         | alert when                                  |
| ---------------------------- | ------------------------------- | ------------------------------------------- |
| `unrated_usage_hourly`       | rating outage, failed file load | >= 30% of an hour's calls charged 0         |
| `effective_rate_by_zone`     | wrong tariff loaded             | >= 15% of a zone's rated calls below tariff |
| `daily_leak_pct_vs_baseline` | anything new / drift            | day >= 2x the 7-day median                  |
| `weekly_slot_recurrence`     | scheduled batch fault           | same weekday-hour unrated in >= 2 weeks     |

PostgreSQL objects (created on first connect): table `leakage_daily_checks`
(PK `check_date, check_name, scope`; re-runs upsert), table `leakage_causes`
(step-6 summary per `as_of` date), view `leakage_alerts` (status = ALERT).
Thresholds are in `THRESHOLDS` at the top of `dailyChecks.py`; alert delivery
(email / Slack / PagerDuty) goes in `send_alert()`.
