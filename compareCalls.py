"""
Leaking vs normal calls: what is different about the calls that leak?

1. Flag every call as leaking or not.
2. Compare duration, zone and time of day between the two groups.
3. Fit a small decision tree on call attributes only (never on the
   charge / leak columns) and print its rules: each leaf that is mostly
   leaking calls is a readable description of a likely root cause.

Usage:
    python compareCalls.py -i cdrs.csv
    python compareCalls.py -i cdrs.csv --max-depth 4
    python compareCalls.py -n 20000 --scenario
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from analyzeGroups import WEEKDAYS, add_time_dimensions
from generateData import generate_cdrs

# Attributes a rating engine / analyst could know *before* rating: no charge, leak, issue.
FEATURES = ["duration_sec", "hour", "weekday_num", "day_of_month", "zone_on_net", "zone_x_net", "zone_international"]


# ---------------------------------------------------------------- 1. flag + 2. compare


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_dimensions(df)
    df["is_leaking"] = df["leak"] > 0
    df["weekday_num"] = df["call_start"].dt.weekday
    df["day_of_month"] = df["call_start"].dt.day
    for z in ["on_net", "x_net", "international"]:
        df[f"zone_{z}"] = (df["zone"] == z).astype(int)
    return df


def compare_numeric(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby("is_leaking")[col]
    out = pd.DataFrame({"normal": g.describe().loc[False], "leaking": g.describe().loc[True]}).loc[["count", "mean", "50%", "min", "max"]]
    return out.round(1)


def compare_categorical(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Share of each group falling in each category, plus leak rate per category."""
    ct = pd.crosstab(df[col], df["is_leaking"])
    ct.columns = ["normal", "leaking"]
    out = pd.DataFrame({
        "normal_%": (100 * ct["normal"] / ct["normal"].sum()).round(1),
        "leaking_%": (100 * ct["leaking"] / ct["leaking"].sum()).round(1),
        "leak_rate_%": (100 * ct["leaking"] / ct.sum(axis=1)).round(1),
    })
    overall = 100 * df["is_leaking"].mean()
    out["lift"] = (out["leak_rate_%"] / overall).round(2)
    return out


def time_bands(hour: pd.Series) -> pd.Series:
    bins = [-1, 5, 8, 17, 21, 23]
    labels = ["night 00-05", "early 06-08", "business 09-17", "evening 18-21", "late 22-23"]
    return pd.cut(hour, bins=bins, labels=labels)


# ---------------------------------------------------------------- 3. decision tree


def fit_tree(df: pd.DataFrame, target: str, max_depth: int = 6, min_leaf: int = 30, ccp_alpha: float = 0.001) -> DecisionTreeClassifier:
    """Shallow, pruned tree: ccp_alpha removes splits that do not buy real purity (noise splits)."""
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf, class_weight="balanced",
                                 ccp_alpha=ccp_alpha, random_state=0)
    clf.fit(df[FEATURES], df[target])
    return clf


def leaf_rules(clf: DecisionTreeClassifier, df: pd.DataFrame, target: str,
               min_lift: float = 2.0, min_rate: float = 0.2) -> pd.DataFrame:
    """Walk the tree; every leaf the tree labels as a leak class becomes a rule, with its rate and
    lift (rate in the leaf / base rate of that class). A rule must hit at least `min_rate` of the
    calls it covers *and* be `min_lift` x the base rate, otherwise it is noise and dropped."""
    t = clf.tree_
    leaf_of = clf.apply(df[FEATURES])
    base = df[target].value_counts(normalize=True)
    rows = []

    def walk(node, bounds):
        if t.children_left[node] == -1:                          # leaf
            cls = clf.classes_[t.value[node].argmax()]           # tree's (class-weighted) decision
            if cls in ("", False):
                return
            sub = df[leaf_of == node]
            rate = (sub[target] == cls).mean()
            lift = rate / base[cls]
            if lift >= min_lift and rate >= min_rate:
                rows.append({"leaf": node, "rule": render(bounds), "predicts": cls, "calls": len(sub),
                             "rate_%": round(100 * rate, 1), "lift": round(lift, 1), "leaked_amount": sub["leak"].sum()})
            return
        f, thr = FEATURES[t.feature[node]], t.threshold[node]
        lo, hi = bounds.get(f, (-np.inf, np.inf))
        walk(t.children_left[node], {**bounds, f: (lo, min(hi, thr))})     # feature <= thr
        walk(t.children_right[node], {**bounds, f: (max(lo, thr), hi)})    # feature >  thr

    walk(0, {})
    cols = ["leaf", "rule", "predicts", "calls", "rate_%", "lift", "leaked_amount"]
    return pd.DataFrame(rows, columns=cols).sort_values("leaked_amount", ascending=False).reset_index(drop=True)


def render(bounds: dict) -> str:
    """Turn per-feature (low, high] bounds into analyst-readable conditions."""
    conds = []
    zones_out = []
    for f, (lo, hi) in bounds.items():
        if f.startswith("zone_"):
            z = f[5:]
            if lo > -np.inf:                                   # right branch: zone_x > 0.5
                conds.append(f"zone == {z}")
            else:
                zones_out.append(z)
        elif f == "weekday_num":
            a = WEEKDAYS[int(lo) + 1] if lo > -np.inf else "Mon"
            b = WEEKDAYS[int(hi)] if hi < np.inf else "Sun"
            conds.append(f"weekday == {a}" if a == b else f"weekday in {a}..{b}")
        elif f == "hour":
            a = int(lo) + 1 if lo > -np.inf else 0
            b = int(hi) if hi < np.inf else 23
            conds.append(f"hour == {a:02d}" if a == b else f"hour in {a:02d}..{b:02d}")
        elif f == "day_of_month":
            a = f"Oct-{int(lo) + 1:02d}" if lo > -np.inf else None
            b = f"Oct-{int(hi):02d}" if hi < np.inf else None
            conds.append(f"day >= {a}" if b is None else f"day <= {b}" if a is None else f"day in {a}..{b}")
        elif f == "duration_sec":
            a = f"{lo:.0f}s < " if lo > -np.inf else ""
            b = f" <= {hi:.0f}s" if hi < np.inf else ""
            conds.append(f"{a}duration{b}")
    if zones_out and not any(c.startswith("zone ==") for c in conds):
        conds.append("zone != " + "/".join(zones_out))
    return " AND ".join(conds) or "(all calls)"


# ---------------------------------------------------------------- report


def print_report(df: pd.DataFrame, max_depth: int) -> None:
    df = prepare(df)
    n_leak = int(df["is_leaking"].sum())
    print("=" * 70); print("LEAKING vs NORMAL CALLS"); print("=" * 70)
    print(f"{len(df):,} calls flagged: {n_leak:,} leaking ({100 * n_leak / len(df):.2f}%), {len(df) - n_leak:,} normal")

    print("\n--- Duration (seconds) ---")
    print(compare_numeric(df, "duration_sec").to_string())

    print("\n--- Zone ---")
    print(compare_categorical(df, "zone").to_string())

    print("\n--- Time of day ---")
    df["time_band"] = time_bands(df["hour"])
    print(compare_categorical(df, "time_band").to_string())

    print("\n--- Weekday ---")
    print(compare_categorical(df, "weekday").to_string())

    # single-feature importances give a quick 'what matters' ranking before reading the tree
    print(f"\n--- Decision tree: leaking vs normal (depth<={max_depth}, features: duration, zone, hour, weekday, day) ---")
    clf = fit_tree(df, "is_leaking", max_depth)
    imp = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("feature importance: " + (", ".join(f"{k}={v:.2f}" for k, v in imp.items() if v > 0) or "none (tree is a single node)"))
    print(export_text(clf, feature_names=FEATURES, class_names=["normal", "leaking"], max_depth=3))

    print("--- Rules for leaking calls (tree leaves labelled 'leaking', lift = leak rate in leaf / overall leak rate) ---")
    rules = leaf_rules(clf, df, "is_leaking")
    print(rules.drop(columns="leaf").to_string() if not rules.empty else "No rule found: leaks look random w.r.t. duration / zone / time.")

    # a tree per issue type reads more naturally as 'root cause' because each cause has its own signature
    print(f"\n--- Decision tree on issue type (normal / unrated_usage / wrong_rate) ---")
    clf2 = fit_tree(df, "issue", max_depth)
    rules2 = leaf_rules(clf2, df, "issue")
    print(rules2.drop(columns="leaf").to_string() if not rules2.empty else "No rule found for any issue type.")

    print("\n--- Calls NOT explained by any rule ---")
    explained = np.isin(clf2.apply(df[FEATURES]), rules2["leaf"].to_numpy())
    rest = df[df["is_leaking"] & ~explained]
    print(f"{len(rest):,} leaking calls ({rest['leak'].sum():,.0f} leaked) fall outside every rule -> background / random leakage")
    if len(rest):
        print("  by issue: " + ", ".join(f"{k}={v}" for k, v in rest["issue"].value_counts().items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare leaking vs normal calls and extract decision-tree rules.")
    parser.add_argument("-i", "--input", help="CSV produced by generateData.py")
    parser.add_argument("-n", "--number-calls", type=int, default=20000)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("--scenario", action="store_true")
    parser.add_argument("--max-depth", type=int, default=6)
    args = parser.parse_args()

    if args.input:
        df = pd.read_csv(args.input, parse_dates=["call_start", "call_end"], keep_default_na=False)
    else:
        df = generate_cdrs(args.number_calls, args.seed, args.scenario)

    with pd.option_context("display.width", 200, "display.max_columns", None, "display.max_colwidth", 120):
        print_report(df, args.max_depth)


if __name__ == "__main__":
    main()
