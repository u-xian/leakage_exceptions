# leakage_exceptions - revenue leakage detection on CDRs

| step | script                                            | what it does                                                               |
| ---- | ------------------------------------------------- | -------------------------------------------------------------------------- |
| 1    | `generateData.py -n 20000 --scenario -o cdrs.csv` | synthetic CDRs; `--scenario` plants the 3 incidents listed in `events.csv` |
| 2    | `analyzeLeakage.py -i cdrs.csv`                   | leaking calls, leaked amount, % of expected revenue                        |
| 3    | `analyzeGroups.py -i cdrs.csv`                    | leak by zone / issue / day / hour / weekday with lift                      |
| 4    | `chartLeakage.py -i cdrs.csv -e events.csv`       | charts + step / spike / weekly detection matched to known events           |
| 5    | `compareCalls.py -i cdrs.csv`                     | leaking vs normal calls, decision-tree rules                               |
| 6    | `leakageSummary.py -i cdrs.csv`                   | total leak, causes, recoverable amount, fix per cause                      |
| 7    | `dailyChecks.py -i cdrs.csv --date yesterday`     | daily checks + alerts, persisted in PostgreSQL                             |

## Daily checks in production

```
export LEAKAGE_DSN=postgresql://user:pw@dbhost:5432/leakage
5 1 * * *  cd /opt/leakage && python dailyChecks.py -i /data/cdrs_latest.csv --date yesterday
```

Tables: `leakage_daily_checks` (every check, every day; PK check_date+check_name+scope, re-runs upsert),
`leakage_causes` (summary per cause), view `leakage_alerts` (status = ALERT).
Thresholds live in `THRESHOLDS` at the top of `dailyChecks.py`; alert delivery is `send_alert()`.
