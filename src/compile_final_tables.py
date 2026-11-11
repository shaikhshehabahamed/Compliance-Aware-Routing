"""
Pulls every results/metrics/*.json and *.csv file into a small set of
clean, paper-ready CSV tables plus one consolidated JSON of headline
numbers. Run last, after every other script has completed on full
(non-smoke) settings.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
TABLES = ROOT / "results" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

METHOD_LABELS = {"linear": "Linear/Logistic", "random_forest": "Random Forest",
                  "xgboost": "XGBoost", "lightgbm": "LightGBM", "mlp": "MLP"}


def table_dataset_summary():
    with open(ROOT / "data" / "processed" / "pipeline_summary.json") as f:
        s = json.load(f)
    rows = [
        ("Routes analysed (Route ID x Week ID)", s["n_routes"]),
        ("Drivers", s["n_drivers"]),
        ("Unique delivery addresses", s["n_addresses"]),
        ("Weeks of operation", s["n_weeks"]),
        ("Countries", s["n_countries"]),
        ("Route size, mean (stops)", round(s["route_size_mean"], 2)),
        ("Route size, median (stops)", s["route_size_median"]),
        ("Single-loop routes (<=2 depot visits)", f"{s['pct_single_loop_routes']:.1f}%"),
        ("Stop-level reorder rate", f"{s['reorder_rate_mean']*100:.1f}%"),
        ("Routes driven exactly as planned", f"{s['pct_routes_perfect_compliance']:.1f}%"),
        ("Deliveries on time", f"{s['on_time_rate']*100:.1f}%"),
        ("Deliveries early", f"{s['early_rate']*100:.1f}%"),
        ("Deliveries late", f"{s['late_rate']*100:.1f}%"),
        ("Anomalous routes excluded", f"{s['cleaning_report']['n_routes_dropped']} "
                                        f"({s['cleaning_report']['pct_routes_dropped']}%)"),
        ("Train / Val / Test routes", f"{s['split_route_counts']['train']} / "
                                        f"{s['split_route_counts']['val']} / {s['split_route_counts']['test']}"),
    ]
    df = pd.DataFrame(rows, columns=["Statistic", "Value"])
    df.to_csv(TABLES / "table1_dataset_summary.csv", index=False)
    return df


def table_model_comparison(task: str, metrics: list[str], fname: str):
    with open(METRICS / "deviation_modeling_results.json") as f:
        res = json.load(f)[task]
    df = pd.DataFrame(res["multiseed_results"])
    agg = df.groupby("method")[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    agg["holm_adjusted_p_vs_best"] = pd.Series(res["pvalues_vs_best_holm"])
    agg["tuning_seconds"] = pd.Series({m: v["tuning_seconds"] for m, v in res["tuning_summary"].items()})
    agg = agg.reset_index().rename(columns={"method": "Method"})
    agg["Method"] = agg["Method"].map(METHOD_LABELS)
    agg.insert(1, "Best", ["<-- selected" if m == METHOD_LABELS[res["best_method"]] else "" for m in agg["Method"]])
    agg.to_csv(TABLES / fname, index=False)
    return agg, res["best_method"]


def table_optimization_comparison():
    with open(METRICS / "timewindow_evaluation.json") as f:
        tw = json.load(f)
    strategies = {
        "status_quo_rate": "Status quo (historical window)",
        "equal_split_rate": "Equal split (recentred)",
        "recenter_only_rate": "Recentred, historical widths",
        "greedy_rate": "Greedy heuristic (risk-informed, ablation)",
        "ga_rate": "Genetic algorithm (metaheuristic, ablation)",
        "optimized_rate": "Compliance-aware (proposed)",
        "oracle_rate": "Oracle (hindsight upper bound)",
    }
    rows = []
    for col, label in strategies.items():
        if col not in tw["summary_mean"]:
            continue  # e.g. ga_rate absent if timewindow_evaluation.json predates the GA baseline
        row = {
            "Strategy": label,
            "Predicted on-time rate, mean": round(tw["summary_mean"][col] * 100, 2),
            "Predicted on-time rate, std": round(tw["summary_std"][col] * 100, 2),
        }
        if col != "optimized_rate" and col in tw["paired_tests_optimized_vs_baseline"]:
            test = tw["paired_tests_optimized_vs_baseline"][col]
            row["p-value vs proposed (paired, raw)"] = round(test["p"], 6)
            if "p_holm" in test:
                row["p-value vs proposed (Holm-corrected)"] = round(test["p_holm"], 6)
        rows.append(row)
    df = pd.DataFrame(rows)
    # also report the real (non-simulated) status-quo on-time rate as an
    # independent sanity check that the evaluation model is well-calibrated
    df.attrs["status_quo_real_rate"] = round(tw["summary_mean"]["status_quo_real_rate"] * 100, 2)
    df.to_csv(TABLES / "table4_optimization_comparison.csv", index=False)
    with open(TABLES / "table4_calibration_check.json", "w") as f:
        json.dump({"predicted_status_quo_on_time_pct": round(tw["summary_mean"]["status_quo_rate"] * 100, 2),
                    "real_observed_status_quo_on_time_pct": round(tw["summary_mean"]["status_quo_real_rate"] * 100, 2)}, f, indent=2)

    ga_path = METRICS / "ga_baseline_results.json"
    if ga_path.exists():
        with open(ga_path) as f:
            ga = json.load(f)
        gm = ga["summary_mean"]
        with open(TABLES / "table4c_ga_cost_comparison.json", "w") as f:
            json.dump({
                "ga_hyperparameters": ga["ga_hyperparameters"], "n_seeds": ga["n_seeds"],
                "exact_seconds_per_route_mean": round(gm["opt_seconds_per_route"], 4),
                "ga_seconds_per_route_mean": round(gm["ga_seconds_per_route"], 4),
                "ga_slowdown_factor": round(gm["ga_seconds_per_route"] / gm["opt_seconds_per_route"], 2),
            }, f, indent=2)
    return df


def table_sensitivity():
    sens = pd.read_csv(METRICS / "sensitivity_analysis.csv")
    budget = sens[sens.sweep == "budget_scale"].groupby("budget_scale").agg(
        optimized_mean=("optimized_rate", "mean"), optimized_std=("optimized_rate", "std"),
        status_quo_mean=("status_quo_rate", "mean")).reset_index()
    strata = sens[sens.sweep == "n_strata"].groupby("n_strata").agg(
        optimized_mean=("optimized_rate", "mean"), optimized_std=("optimized_rate", "std")).reset_index()
    budget.to_csv(TABLES / "table5a_sensitivity_budget.csv", index=False)
    strata.to_csv(TABLES / "table5b_sensitivity_strata.csv", index=False)
    return budget, strata


def table_ablation_features():
    with open(METRICS / "ablation_features_results.json") as f:
        ab = json.load(f)
    labels = {"full": "Full feature set", "no_addr_history": "No address-history features",
              "no_driver_history": "No driver-history features", "no_history_all": "No history features (cold start)"}
    rows = []
    for task in ["T1", "T2"]:
        metric = ab[task]["primary_metric"]
        for variant, s in ab[task]["summary"].items():
            rows.append({
                "Task": task, "Metric": metric.upper(), "Variant": labels[variant],
                "n_features": s["n_features"],
                "Mean": round(s[f"{metric}_mean"], 4), "Std": round(s[f"{metric}_std"], 4),
                "Delta vs. full": round(s["delta_vs_full"], 4) if variant != "full" else "",
                "Holm-adjusted p vs. full": round(ab[task]["pvalues_vs_full_holm"][variant], 4)
                                             if variant != "full" else "",
            })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "table7_ablation_features.csv", index=False)
    return df


def table_resequencing():
    with open(METRICS / "resequencing_summary.json") as f:
        s = json.load(f)
    df = pd.DataFrame(list(s.items()), columns=["Statistic", "Value"])
    df.to_csv(TABLES / "table6_resequencing_summary.csv", index=False)
    return df


def main():
    headline = {}
    ds = table_dataset_summary()
    headline["dataset"] = ds.set_index("Statistic")["Value"].to_dict()

    t1, best_t1 = table_model_comparison("T1", ["auroc", "auprc", "brier", "f1", "accuracy"],
                                          "table2_t1_model_comparison.csv")
    t2, best_t2 = table_model_comparison("T2", ["mae", "rmse", "r2"], "table3_t2_model_comparison.csv")
    headline["best_t1_method"] = best_t1
    headline["best_t2_method"] = best_t2

    opt = table_optimization_comparison()
    headline["optimization_comparison"] = opt.to_dict(orient="records")
    ga_cost_path = TABLES / "table4c_ga_cost_comparison.json"
    if ga_cost_path.exists():
        with open(ga_cost_path) as f:
            headline["ga_cost_comparison"] = json.load(f)

    budget, strata = table_sensitivity()
    headline["sensitivity_budget_range_pp"] = round(
        (budget["optimized_mean"].max() - budget["optimized_mean"].min()) * 100, 2)

    try:
        reseq = table_resequencing()
        headline["resequencing"] = reseq.set_index("Statistic")["Value"].to_dict()
    except FileNotFoundError:
        print("resequencing_summary.json not found -- skipping table 6")

    try:
        ablation = table_ablation_features()
        headline["ablation_features"] = ablation.to_dict(orient="records")
    except FileNotFoundError:
        print("ablation_features_results.json not found -- skipping table 7")

    with open(TABLES / "headline_numbers.json", "w") as f:
        json.dump(headline, f, indent=2, default=str)
    print("Wrote tables to", TABLES)
    print(json.dumps(headline, indent=2, default=str))


if __name__ == "__main__":
    main()
