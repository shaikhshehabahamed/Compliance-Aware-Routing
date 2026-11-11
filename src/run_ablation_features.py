"""
Ablation study: how much does the leak-free "compliance-history" feature
block (driver x address familiarity, mined as expanding, past-only
aggregates in data_pipeline.py) actually contribute to T1/T2 accuracy?

Protocol. For each task, take the SAME best method and SAME tuned
hyperparameters selected by run_deviation_modeling.py on the full feature
set -- hyperparameters are held FIXED here, not re-tuned per variant. This
is the standard ablation protocol: it isolates the delta attributable to
the removed features alone, rather than confounding it with a fresh
hyperparameter search that might compensate differently for each variant.
Each variant is retrained across the SAME seeds used in the main run:

  - full               : every feature (already computed by
                          run_deviation_modeling.py; reused, not rerun)
  - no_addr_history     : drop addr_hist_reorder_rate, addr_hist_ontime_rate,
                          addr_visit_seq (does the model still work without
                          knowing this driver's history AT this address?)
  - no_driver_history   : drop driver_hist_reorder_rate, driver_visit_seq,
                          driver_addr_visit_seq (without this driver's
                          general track record?)
  - no_history_all      : drop all six ("cold start": only structural /
                          contextual features -- position, distance, route
                          size, week, depot/delivery flag, country, weekday,
                          plus window features for T1)

Each ablated variant is compared to the full-feature model with the same
paired Wilcoxon/t-test + Holm correction used elsewhere in this project.

Requires results/metrics/deviation_modeling_results.json to exist (run
run_deviation_modeling.py first).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import deviation_models as dm
import features as feat
from run_deviation_modeling import load_splits, prep_task

ROOT = Path(__file__).resolve().parents[1]
RESULTS_METRICS = ROOT / "results" / "metrics"

VARIANTS = {
    "no_addr_history": feat.ADDR_HISTORY_FEATURES,
    "no_driver_history": feat.DRIVER_HISTORY_FEATURES,
    "no_history_all": feat.ALL_HISTORY_FEATURES,
}
VARIANT_LABELS = {
    "full": "Full feature set",
    "no_addr_history": "No address-history features",
    "no_driver_history": "No driver-history features",
    "no_history_all": "No history features (cold start)",
}


def holm_correction(pvals: dict) -> dict:
    methods_sorted = sorted(pvals, key=lambda m: pvals[m])
    m = len(methods_sorted)
    holm = {}
    for i, meth in enumerate(methods_sorted):
        holm[meth] = min(1.0, pvals[meth] * (m - i))
    running_max = 0.0
    for meth in methods_sorted:
        running_max = max(running_max, holm[meth])
        holm[meth] = running_max
    return holm


def run_task_ablation(task: str, train, val, test, best_method: str, best_params_raw: dict,
                       seeds: list[int], full_seed_metrics: pd.DataFrame) -> dict:
    is_clf = task == "T1"
    primary_metric = "auroc" if is_clf else "mae"
    better = "max" if is_clf else "min"

    variant_rows = {"full": full_seed_metrics}
    for name, drop_cols in VARIANTS.items():
        data = prep_task(task, train, val, test, drop_features=drop_cols)
        seed_results = dm.multi_seed_test_eval(
            best_method, task, best_params_raw, data["X_trainval"], data["y_trainval"],
            data["X_test"], data["y_test"], seeds=seeds)
        df = pd.DataFrame(seed_results)
        df["n_features"] = len(data["feature_names"])
        variant_rows[name] = df
        print(f"[{task}] ablation {name}: n_features={len(data['feature_names'])}, "
              f"{primary_metric} mean={df[primary_metric].mean():.4f} "
              f"(full={full_seed_metrics[primary_metric].mean():.4f})")

    pvals = {}
    for name in VARIANTS:
        wide = pd.DataFrame({
            "full": variant_rows["full"].set_index("seed")[primary_metric],
            name: variant_rows[name].set_index("seed")[primary_metric],
        }).dropna()
        a, b = wide["full"].values, wide[name].values
        if len(a) >= 6 and np.any(a != b):
            stat, p = stats.wilcoxon(a, b)
        elif len(a) >= 2:
            stat, p = stats.ttest_rel(a, b)
        else:
            p = float("nan")
        pvals[name] = float(p)
    holm = holm_correction(pvals)

    summary = {}
    for name, df in variant_rows.items():
        summary[name] = {
            f"{primary_metric}_mean": float(df[primary_metric].mean()),
            f"{primary_metric}_std": float(df[primary_metric].std()),
            "n_features": int(df["n_features"].iloc[0]) if "n_features" in df.columns else None,
        }
    for name in VARIANTS:
        delta = summary[name][f"{primary_metric}_mean"] - summary["full"][f"{primary_metric}_mean"]
        summary[name]["delta_vs_full"] = delta
        summary[name]["degrades_performance"] = bool((delta < 0) if better == "max" else (delta > 0))

    return {
        "task": task, "primary_metric": primary_metric, "best_method": best_method,
        "seeds": seeds, "summary": summary, "pvalues_vs_full_holm": holm,
        "per_variant_records": {k: v.to_dict(orient="records") for k, v in variant_rows.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    suffix = "_smoke" if args.smoke else ""

    with open(RESULTS_METRICS / f"deviation_modeling_results{suffix}.json") as f:
        dev_results = json.load(f)

    stop, train, val, test = load_splits()

    all_results = {}
    for task in ["T1", "T2"]:
        res = dev_results[task]
        best_method = res["best_method"]
        best_params_raw = res["tuning_summary"][best_method]["best_params"]
        full_df = pd.DataFrame(res["multiseed_results"])
        full_df = full_df[full_df.method == best_method].copy()
        full_df["n_features"] = len(res["feature_names"])
        seeds = sorted(full_df["seed"].unique().tolist())

        out = run_task_ablation(task, train, val, test, best_method, best_params_raw, seeds, full_df)
        all_results[task] = out
        print(f"[{task}] ablation baseline method = {best_method}, seeds={seeds}")

    with open(RESULTS_METRICS / f"ablation_features_results{suffix}.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    rows = []
    for task in ["T1", "T2"]:
        out = all_results[task]
        metric = out["primary_metric"]
        for variant, s in out["summary"].items():
            rows.append({
                "Task": task, "Metric": metric, "Variant": VARIANT_LABELS[variant],
                "n_features": s["n_features"],
                "metric_mean": round(s[f"{metric}_mean"], 4),
                "metric_std": round(s[f"{metric}_std"], 4),
                "delta_vs_full": round(s["delta_vs_full"], 4) if variant != "full" else "",
                "holm_p_vs_full": round(out["pvalues_vs_full_holm"][variant], 4) if variant != "full" else "",
            })
    pd.DataFrame(rows).to_csv(RESULTS_METRICS / f"ablation_features_summary{suffix}.csv", index=False)
    print("Saved ablation_features_results.json and ablation_features_summary.csv")


if __name__ == "__main__":
    main()
