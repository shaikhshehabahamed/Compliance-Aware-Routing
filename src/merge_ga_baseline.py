"""One-time migration (already applied; kept for provenance): merges the
standalone GA baseline (run_ga_baseline.py, run before ga_allocation was wired
into evaluate_one_seed) into the canonical Part A results, producing exactly
what run_timewindow_evaluation.py now writes natively with compute_ga=True.
Validates route-level consistency between the two independently-computed
result sets before trusting the merge -- if the underlying models/predictions
had diverged between the two separate script runs, this would have caught it
rather than silently overwriting the canonical files with something wrong.
Re-running this script is a no-op against a fresh run_timewindow_evaluation.py
output (there is no separate ga_baseline_route_level.csv left to merge from)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"

main_df = pd.read_csv(METRICS / "timewindow_route_level.csv")
ga_df = pd.read_csv(METRICS / "ga_baseline_route_level.csv")
key = ["route_id", "week_id", "seed"]

print(f"main rows: {len(main_df)}, ga rows: {len(ga_df)}")
check = main_df.merge(ga_df[key + ["optimized_expected", "greedy_expected"]], on=key,
                       suffixes=("_main", "_ga"), how="inner")
print(f"matched rows: {len(check)}")
opt_diff = (check["optimized_expected_main"] - check["optimized_expected_ga"]).abs()
greedy_diff = (check["greedy_expected_main"] - check["greedy_expected_ga"]).abs()
print(f"optimized_expected: max|diff|={opt_diff.max():.6g}, mean|diff|={opt_diff.mean():.6g}")
print(f"greedy_expected:    max|diff|={greedy_diff.max():.6g}, mean|diff|={greedy_diff.mean():.6g}")

assert len(check) == len(main_df) == len(ga_df), "route sets differ between the two files -- do not merge blindly"
assert opt_diff.max() < 0.01, "optimized_expected disagreement too large -- underlying predictions may differ between runs"
assert greedy_diff.max() < 0.01, "greedy_expected disagreement too large -- underlying predictions may differ between runs"
print("Consistency check PASSED -- the two independent runs agree to floating-point/solver-tie precision.")

merged = main_df.merge(ga_df[key + ["ga_expected", "ga_status"]], on=key, how="left")
assert merged["ga_expected"].isna().sum() == 0, "some routes missing GA data after merge"
merged.to_csv(METRICS / "timewindow_route_level.csv", index=False)

all_totals = []
for seed in sorted(merged["seed"].unique()):
    sub = merged[merged["seed"] == seed]
    all_totals.append({
        "n_routes": len(sub), "n_stops": int(sub["n_stops"].sum()),
        "status_quo_rate": sub["status_quo_expected"].sum() / sub["n_stops"].sum(),
        "status_quo_real_rate": sub["status_quo_real_ontime"].sum() / sub["n_stops"].sum(),
        "equal_split_rate": sub["equal_split_expected"].sum() / sub["n_stops"].sum(),
        "recenter_only_rate": sub["recenter_only_expected"].sum() / sub["n_stops"].sum(),
        "optimized_rate": sub["optimized_expected"].sum() / sub["n_stops"].sum(),
        "greedy_rate": sub["greedy_expected"].sum() / sub["n_stops"].sum(),
        "oracle_rate": sub["oracle_expected"].sum() / sub["n_stops"].sum(),
        "ga_rate": sub["ga_expected"].sum() / sub["n_stops"].sum(),
        "seed": int(seed), "n_strata": 5, "budget_scale": 1.0,
    })
totals_df = pd.DataFrame(all_totals)

base_cols = ["status_quo_rate", "equal_split_rate", "recenter_only_rate", "greedy_rate", "ga_rate"]
pvals = {}
for base_col in base_cols:
    a = totals_df["optimized_rate"].values
    b = totals_df[base_col].values
    if len(a) >= 6:
        stat, p = stats.wilcoxon(a, b)
    else:
        stat, p = stats.ttest_rel(a, b)
    pvals[base_col] = {"stat": float(stat), "p": float(p)}

raw_p = {k: v["p"] for k, v in pvals.items()}
sorted_cols = sorted(raw_p, key=lambda c: raw_p[c])
m = len(sorted_cols)
holm, running_max = {}, 0.0
for i, col in enumerate(sorted_cols):
    running_max = max(running_max, min(1.0, raw_p[col] * (m - i)))
    holm[col] = running_max
for col in pvals:
    pvals[col]["p_holm"] = holm[col]

out = {
    "per_seed_totals": totals_df.to_dict(orient="records"),
    "summary_mean": totals_df.mean(numeric_only=True).to_dict(),
    "summary_std": totals_df.std(numeric_only=True).to_dict(),
    "paired_tests_optimized_vs_baseline": pvals,
}
with open(METRICS / "timewindow_evaluation.json", "w") as f:
    json.dump(out, f, indent=2, default=str)

print()
print("summary_mean:")
print(json.dumps(out["summary_mean"], indent=2))
print("paired tests (Holm-corrected):")
print(json.dumps(pvals, indent=2))
print()
print("Wrote merged timewindow_evaluation.json and timewindow_route_level.csv")
