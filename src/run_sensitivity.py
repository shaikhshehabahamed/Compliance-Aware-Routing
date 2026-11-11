"""
Sensitivity analysis for Part A: how robust is the "optimized beats
baselines" conclusion to (1) how generous the total time-window budget is,
and (2) how many risk strata the residual calibrator uses? Both are run at
a reduced seed count (still >1, so still genuinely multi-seed) since this
stage multiplies an already-expensive per-route optimization loop across
several budget/stratum settings.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from run_deviation_modeling import load_splits, prep_task
from run_timewindow_evaluation import evaluate_one_seed

ROOT = Path(__file__).resolve().parents[1]
RESULTS_METRICS = ROOT / "results" / "metrics"


def main(seeds=(0, 1, 2), budget_scales=(0.7, 0.85, 1.0, 1.15, 1.3), strata_options=(1, 3, 5, 8)):
    with open(RESULTS_METRICS / "deviation_modeling_results.json") as f:
        best = json.load(f)
    stop, train, val, test = load_splits()
    data_t1 = prep_task("T1", train, val, test)
    data_t2 = prep_task("T2", train, val, test)

    rows = []
    # sweep 1: budget generosity (n_strata fixed at 5)
    for scale in budget_scales:
        for seed in seeds:
            totals, _ = evaluate_one_seed(seed, data_t1, data_t2, best, test,
                                           n_strata=5, budget_scale=scale, time_limit_s=1.0)
            totals["sweep"] = "budget_scale"
            rows.append(totals)
            print("budget_scale", scale, "seed", seed, "->", totals["optimized_rate"], "vs status quo", totals["status_quo_rate"])

    # sweep 2: number of risk strata (budget_scale fixed at 1.0)
    for n_strata in strata_options:
        for seed in seeds:
            totals, _ = evaluate_one_seed(seed, data_t1, data_t2, best, test,
                                           n_strata=n_strata, budget_scale=1.0, time_limit_s=1.0)
            totals["sweep"] = "n_strata"
            rows.append(totals)
            print("n_strata", n_strata, "seed", seed, "->", totals["optimized_rate"])

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_METRICS / "sensitivity_analysis.csv", index=False)
    print("saved sensitivity_analysis.csv")


if __name__ == "__main__":
    main()
