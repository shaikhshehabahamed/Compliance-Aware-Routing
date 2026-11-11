"""
Supplementary robustness check, run standalone so it never touches (or races
with) run_timewindow_evaluation.py / run_sensitivity.py, which may still be
executing when this runs: does a genetic algorithm reach solutions
competitive with exact CP-SAT optimization on the IDENTICAL per-route
discretized budget-allocation problem? See ga_allocation's docstring in
timewindow_optimizer.py for why GA (not PSO) and why fixed hyperparameters.

Requires results/metrics/deviation_modeling_results.json (already complete).
Writes its own output files (ga_baseline_*) -- does not modify or depend on
anything run_timewindow_evaluation.py / run_sensitivity.py produce beyond
that one already-finished, read-only input.

Usage:
    python run_ga_baseline.py --smoke                 # small, fast check
    python run_ga_baseline.py --seeds 5                # full comparison
    python run_ga_baseline.py --seeds 5 --limit_routes 200
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import timewindow_optimizer as two
from run_deviation_modeling import load_splits, prep_task
from run_timewindow_evaluation import fit_predict

ROOT = Path(__file__).resolve().parents[1]
RESULTS_METRICS = ROOT / "results" / "metrics"


def evaluate_one_seed(seed: int, data_t1: dict, data_t2: dict, best: dict,
                       test_stop: pd.DataFrame, n_strata: int, time_limit_s: float,
                       ga_kwargs: dict, limit_routes: int | None) -> tuple[dict, pd.DataFrame]:
    t1_method, t1_params = best["T1"]["best_method"], best["T1"]["tuning_summary"][best["T1"]["best_method"]]["best_params"]
    t2_method, t2_params = best["T2"]["best_method"], best["T2"]["tuning_summary"][best["T2"]["best_method"]]["best_params"]

    risk_val = fit_predict(t1_method, "T1", t1_params, data_t1["X_train"], data_t1["y_train"],
                            data_t1["X_val"], seed)
    pred_val = fit_predict(t2_method, "T2", t2_params, data_t2["X_train"], data_t2["y_train"],
                            data_t2["X_val"], seed)
    resid_val = data_t2["y_val"] - pred_val
    calibrator = two.ResidualCalibrator(n_strata=n_strata).fit(risk_val, resid_val)

    risk_test = fit_predict(t1_method, "T1", t1_params, data_t1["X_trainval"], data_t1["y_trainval"],
                             data_t1["X_test"], seed)
    pred_test = fit_predict(t2_method, "T2", t2_params, data_t2["X_trainval"], data_t2["y_trainval"],
                             data_t2["X_test"], seed)

    ts = test_stop.reset_index(drop=True).copy()
    ts["pred_arrival"] = pred_test
    ts["risk_score"] = risk_test
    delivery = ts[ts["Delivery"] == 1].copy()

    rows = []
    ga_seconds_total = 0.0
    opt_seconds_total = 0.0
    for route_idx, ((rid, wid), g) in enumerate(delivery.groupby(["Route ID", "Week ID"])):
        if limit_routes is not None and route_idx >= limit_routes:
            break
        n = len(g)
        if n < 2:
            continue
        budget = g["window_width"].sum()
        pred_arrival = g["pred_arrival"].values
        risk_score = g["risk_score"].values
        strata = calibrator.stratum_of(risk_score)

        t0 = time.time()
        sol_opt = two.solve_route_allocation(pred_arrival, budget, calibrator, risk_score,
                                              time_limit_s=time_limit_s)
        opt_seconds_total += time.time() - t0
        p_opt = calibrator.p_on_time(sol_opt["widths"] / 2.0, strata)

        sol_greedy = two.greedy_allocation(pred_arrival, budget, calibrator, risk_score)
        p_greedy = calibrator.p_on_time(sol_greedy["widths"] / 2.0, strata)

        ga_seed = seed * 100_003 + route_idx  # distinct, reproducible RNG stream per route
        t0 = time.time()
        sol_ga = two.ga_allocation(pred_arrival, budget, calibrator, risk_score,
                                    seed=ga_seed, **ga_kwargs)
        ga_seconds_total += time.time() - t0
        p_ga = calibrator.p_on_time(sol_ga["widths"] / 2.0, strata)

        rows.append({
            "route_id": rid, "week_id": wid, "n_stops": n, "budget": budget,
            "optimized_expected": p_opt.sum(), "greedy_expected": p_greedy.sum(),
            "ga_expected": p_ga.sum(),
            "opt_status": sol_opt["status"], "ga_status": sol_ga["status"],
        })

    route_df = pd.DataFrame(rows)
    totals = {
        "n_routes": len(route_df), "n_stops": int(route_df["n_stops"].sum()),
        "optimized_rate": route_df["optimized_expected"].sum() / route_df["n_stops"].sum(),
        "greedy_rate": route_df["greedy_expected"].sum() / route_df["n_stops"].sum(),
        "ga_rate": route_df["ga_expected"].sum() / route_df["n_stops"].sum(),
        "opt_seconds_per_route": opt_seconds_total / max(len(route_df), 1),
        "ga_seconds_per_route": ga_seconds_total / max(len(route_df), 1),
        "seed": seed, "n_strata": n_strata,
    }
    return totals, route_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n_strata", type=int, default=5)
    ap.add_argument("--limit_routes", type=int, default=None)
    ap.add_argument("--population_size", type=int, default=30)
    ap.add_argument("--generations", type=int, default=40)
    args = ap.parse_args()

    with open(RESULTS_METRICS / "deviation_modeling_results.json") as f:
        best = json.load(f)

    stop, train, val, test = load_splits()
    data_t1 = prep_task("T1", train, val, test)
    data_t2 = prep_task("T2", train, val, test)

    seeds = [0] if args.smoke else list(range(args.seeds))
    time_limit = 0.5 if args.smoke else 2.0
    limit_routes = 30 if args.smoke else args.limit_routes
    ga_kwargs = dict(population_size=args.population_size, generations=args.generations)

    all_totals, all_route_level = [], []
    for seed in seeds:
        t0 = time.time()
        totals, route_df = evaluate_one_seed(seed, data_t1, data_t2, best, test,
                                              n_strata=args.n_strata, time_limit_s=time_limit,
                                              ga_kwargs=ga_kwargs, limit_routes=limit_routes)
        print(f"seed={seed} done in {time.time()-t0:.1f}s: {totals}")
        all_totals.append(totals)
        route_df["seed"] = seed
        all_route_level.append(route_df)

    totals_df = pd.DataFrame(all_totals)
    route_level_df = pd.concat(all_route_level, ignore_index=True)

    pvals = {}
    for base_col in ["optimized_rate", "greedy_rate"]:
        a = totals_df["ga_rate"].values
        b = totals_df[base_col].values
        if len(a) >= 6:
            stat, p = stats.wilcoxon(a, b)
        elif len(a) >= 2:
            stat, p = stats.ttest_rel(a, b)
        else:
            stat, p = float("nan"), float("nan")
        pvals[f"ga_vs_{base_col}"] = {"stat": float(stat), "p": float(p)}

    out = {
        "ga_hyperparameters": ga_kwargs,
        "n_seeds": len(seeds), "limit_routes": limit_routes,
        "per_seed_totals": totals_df.to_dict(orient="records"),
        "summary_mean": totals_df.mean(numeric_only=True).to_dict(),
        "summary_std": totals_df.std(numeric_only=True).to_dict(),
        "paired_tests": pvals,
    }
    suffix = "_smoke" if args.smoke else ""
    with open(RESULTS_METRICS / f"ga_baseline_results{suffix}.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    route_level_df.to_csv(RESULTS_METRICS / f"ga_baseline_route_level{suffix}.csv", index=False)
    print(json.dumps(out["summary_mean"], indent=2))
    print(json.dumps(pvals, indent=2))


if __name__ == "__main__":
    main()
