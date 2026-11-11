"""
Part A evaluation: compliance-aware time-window budget allocation vs.
baselines, on the held-out test routes, across multiple seeds, with paired
statistical tests and a budget-scaling sensitivity sweep.

Requires results/metrics/deviation_modeling_results.json to exist (run
run_deviation_modeling.py first) so we know each task's best method +
hyperparameters.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import deviation_models as dm
import features as feat
import timewindow_optimizer as two
from run_deviation_modeling import load_splits, prep_task

ROOT = Path(__file__).resolve().parents[1]
RESULTS_METRICS = ROOT / "results" / "metrics"


def fit_predict(method: str, task: str, params_raw: dict, X_fit, y_fit, X_pred, seed: int):
    params = dm.finalize_params(method, params_raw)
    model = dm.get_model(method, task, params, seed=seed)
    model.fit(X_fit, y_fit)
    if task == "T1":
        return model.predict_proba(X_pred)[:, 1]
    return model.predict(X_pred)


def evaluate_one_seed(seed: int, data_t1: dict, data_t2: dict, best: dict,
                       test_stop: pd.DataFrame, n_strata: int, budget_scale: float,
                       time_limit_s: float, compute_ga: bool = False) -> dict:
    t1_method, t1_params = best["T1"]["best_method"], best["T1"]["tuning_summary"][best["T1"]["best_method"]]["best_params"]
    t2_method, t2_params = best["T2"]["best_method"], best["T2"]["tuning_summary"][best["T2"]["best_method"]]["best_params"]

    # calibration predictions (train-only model -> predict on val) for the residual calibrator
    risk_val = fit_predict(t1_method, "T1", t1_params, data_t1["X_train"], data_t1["y_train"],
                            data_t1["X_val"], seed)
    pred_val = fit_predict(t2_method, "T2", t2_params, data_t2["X_train"], data_t2["y_train"],
                            data_t2["X_val"], seed)
    resid_val = data_t2["y_val"] - pred_val
    calibrator = two.ResidualCalibrator(n_strata=n_strata).fit(risk_val, resid_val)

    # deployment predictions (train+val model -> predict on test)
    risk_test = fit_predict(t1_method, "T1", t1_params, data_t1["X_trainval"], data_t1["y_trainval"],
                             data_t1["X_test"], seed)
    pred_test = fit_predict(t2_method, "T2", t2_params, data_t2["X_trainval"], data_t2["y_trainval"],
                             data_t2["X_test"], seed)

    ts = test_stop.reset_index(drop=True).copy()
    ts["pred_arrival"] = pred_test
    ts["risk_score"] = risk_test
    delivery = ts[ts["Delivery"] == 1].copy()

    rows = []
    for route_idx, ((rid, wid), g) in enumerate(delivery.groupby(["Route ID", "Week ID"])):
        n = len(g)
        if n < 2:
            continue
        budget = g["window_width"].sum() * budget_scale
        pred_arrival = g["pred_arrival"].values
        risk_score = g["risk_score"].values
        true_arrival = g["Arrived Time"].values
        hist_width = g["window_width"].values
        hist_mid = g["window_mid"].values

        strata = calibrator.stratum_of(risk_score)

        # 1. status quo (real historical center+width). The historical window is
        # centred at hist_mid, NOT at the model's prediction, so this must use
        # the asymmetric form: offsets are relative to pred_arrival, not to the
        # window's own centre.
        hist_earliest = hist_mid - hist_width / 2.0
        hist_latest = hist_mid + hist_width / 2.0
        p_sq = calibrator.p_within_offsets(hist_earliest - pred_arrival, hist_latest - pred_arrival, strata)
        real_on_time = g["on_time"].values.astype(float)

        # 2. recentred equal split of the same budget
        w_eq = two.equal_split_allocation(n, budget)
        p_eq = calibrator.p_on_time(w_eq / 2.0, strata)

        # 3. recentred, same per-stop width as status quo (isolates centering value)
        p_recenter_only = calibrator.p_on_time((hist_width * budget_scale) / 2.0, strata)

        # 4. risk-informed optimized allocation (exact, CP-SAT)
        sol = two.solve_route_allocation(pred_arrival, budget, calibrator, risk_score,
                                          time_limit_s=time_limit_s)
        p_opt = calibrator.p_on_time(sol["widths"] / 2.0, strata)

        # 4b. ablation: risk-informed allocation via a cheap greedy heuristic
        # instead of exact CP-SAT, on the IDENTICAL probability grid -- isolates
        # the value of exact optimization from the value of being risk-informed
        # at all (that latter value is already captured by optimized vs. equal_split).
        sol_greedy = two.greedy_allocation(pred_arrival, budget, calibrator, risk_score)
        p_greedy = calibrator.p_on_time(sol_greedy["widths"] / 2.0, strata)

        # 4c. supplementary robustness check: genetic algorithm on the IDENTICAL
        # probability grid (see ga_allocation's docstring for why GA, not PSO).
        # Optional (compute_ga) and off by default: run_sensitivity.py calls this
        # function 27 times per sweep and must not pay GA's ~4.6x-slower-than-exact
        # per-route cost on every one of those calls. Route-indexed seed keeps the
        # comparison reproducible if this is ever re-run from scratch.
        if compute_ga:
            ga_seed = seed * 100_003 + route_idx
            sol_ga = two.ga_allocation(pred_arrival, budget, calibrator, risk_score, seed=ga_seed)
            p_ga = calibrator.p_on_time(sol_ga["widths"] / 2.0, strata)

        # 5. oracle: hindsight upper bound -- same width grid, same budget,
        # but the TRUE arrival times, solved as the identical knapsack
        # solve_route_allocation uses (see oracle_allocation's docstring for
        # why a cheapest-first greedy is wrong here: it can violate the
        # MIN_WIDTH floor every other method respects and score BELOW them).
        sol_oracle = two.oracle_allocation(pred_arrival, true_arrival, budget, time_limit_s=time_limit_s)

        row = {
            "route_id": rid, "week_id": wid, "n_stops": n, "budget": budget,
            "status_quo_expected": p_sq.sum(), "status_quo_real_ontime": real_on_time.sum(),
            "equal_split_expected": p_eq.sum(),
            "recenter_only_expected": p_recenter_only.sum(),
            "optimized_expected": p_opt.sum(),
            "greedy_expected": p_greedy.sum(),
            "oracle_expected": sol_oracle["true_on_time_count"],
            "opt_status": sol["status"],
            "greedy_status": sol_greedy["status"],
            "oracle_status": sol_oracle["status"],
        }
        if compute_ga:
            row["ga_expected"] = p_ga.sum()
            row["ga_status"] = sol_ga["status"]
        rows.append(row)

    route_df = pd.DataFrame(rows)
    totals = {
        "n_routes": len(route_df),
        "n_stops": int(route_df["n_stops"].sum()),
        "status_quo_rate": route_df["status_quo_expected"].sum() / route_df["n_stops"].sum(),
        "status_quo_real_rate": route_df["status_quo_real_ontime"].sum() / route_df["n_stops"].sum(),
        "equal_split_rate": route_df["equal_split_expected"].sum() / route_df["n_stops"].sum(),
        "recenter_only_rate": route_df["recenter_only_expected"].sum() / route_df["n_stops"].sum(),
        "optimized_rate": route_df["optimized_expected"].sum() / route_df["n_stops"].sum(),
        "greedy_rate": route_df["greedy_expected"].sum() / route_df["n_stops"].sum(),
        "oracle_rate": route_df["oracle_expected"].sum() / route_df["n_stops"].sum(),
        "seed": seed, "n_strata": n_strata, "budget_scale": budget_scale,
    }
    if compute_ga:
        totals["ga_rate"] = route_df["ga_expected"].sum() / route_df["n_stops"].sum()
    return totals, route_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n_strata", type=int, default=5)
    args = ap.parse_args()

    with open(RESULTS_METRICS / ("deviation_modeling_results_smoke.json" if args.smoke
                                  else "deviation_modeling_results.json")) as f:
        best = json.load(f)

    stop, train, val, test = load_splits()
    data_t1 = prep_task("T1", train, val, test)
    data_t2 = prep_task("T2", train, val, test)

    seeds = list(range(2)) if args.smoke else list(range(args.seeds))
    time_limit = 0.5 if args.smoke else 2.0

    # GA is skipped in --smoke (its per-route cost doesn't scale down with
    # time_limit_s the way CP-SAT's does, so smoke mode would lose its speed
    # advantage entirely) but always included in a real run.
    compute_ga = not args.smoke

    all_totals = []
    all_route_level = []
    for seed in seeds:
        t0 = time.time()
        totals, route_df = evaluate_one_seed(seed, data_t1, data_t2, best, test,
                                              n_strata=args.n_strata, budget_scale=1.0,
                                              time_limit_s=time_limit, compute_ga=compute_ga)
        print(f"seed={seed} done in {time.time()-t0:.1f}s: {totals}")
        all_totals.append(totals)
        route_df["seed"] = seed
        all_route_level.append(route_df)

    totals_df = pd.DataFrame(all_totals)
    route_level_df = pd.concat(all_route_level, ignore_index=True)

    # paired statistical tests across seeds: optimized vs each baseline
    # (greedy_rate/ga_rate included -- this is the exact-vs-heuristic ablation),
    # Holm-Bonferroni corrected across the family (same convention as the T1/T2
    # method comparison in run_deviation_modeling.py).
    base_cols = ["status_quo_rate", "equal_split_rate", "recenter_only_rate", "greedy_rate"]
    if compute_ga:
        base_cols.append("ga_rate")
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
    suffix = "_smoke" if args.smoke else ""
    with open(RESULTS_METRICS / f"timewindow_evaluation{suffix}.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    route_level_df.to_csv(RESULTS_METRICS / f"timewindow_route_level{suffix}.csv", index=False)
    print(json.dumps(out["summary_mean"], indent=2))
    print(json.dumps(pvals, indent=2))


if __name__ == "__main__":
    main()
