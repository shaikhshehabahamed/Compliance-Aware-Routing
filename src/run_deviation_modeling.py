"""
Orchestrates Part 1: train/tune/compare 5 ML methods on 2 prediction tasks
(T1 = will this stop be reordered; T2 = predicted arrival time), select the
best method per task, evaluate across multiple random seeds on the held-out
test weeks, and run pairwise statistical significance tests between methods.

Usage:
    python run_deviation_modeling.py --smoke      # fast correctness check
    python run_deviation_modeling.py               # full run
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.inspection import permutation_importance

import deviation_models as dm
import features as feat

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS_METRICS = ROOT / "results" / "metrics"
RESULTS_METRICS.mkdir(parents=True, exist_ok=True)


def load_splits():
    stop = pd.read_parquet(PROCESSED / "stop_table.parquet")
    train = stop[stop["split"] == "train"]
    val = stop[stop["split"] == "val"]
    test = stop[stop["split"] == "test"]
    return stop, train, val, test


def prep_task(task: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
              drop_features: list[str] | None = None):
    X_train, y_train = feat.build_xy(train, task, drop_features=drop_features)
    X_val, y_val = feat.build_xy(val, task, drop_features=drop_features)
    X_test, y_test = feat.build_xy(test, task, drop_features=drop_features)
    # align dummy columns across splits (val/test may lack a category level seen in train)
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0.0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)
    medians = feat.impute_fit(X_train)
    X_train = feat.impute_apply(X_train, medians)
    X_val = feat.impute_apply(X_val, medians)
    X_test = feat.impute_apply(X_test, medians)
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val])
    return dict(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test, X_trainval=X_trainval,
                y_trainval=y_trainval, medians=medians, feature_names=list(X_train.columns))


def run_task(task: str, data: dict, methods: list[str], n_trials: int, seeds: list[int]) -> dict:
    is_clf = task == "T1"
    primary_metric = "auroc" if is_clf else "mae"
    better = "max" if is_clf else "min"

    tuning_summary = {}
    multiseed_rows = []
    for method in methods:
        t0 = time.time()
        best_params, best_val_score, _study = dm.tune_method(
            method, task, data["X_train"], data["y_train"], data["X_val"], data["y_val"],
            n_trials=n_trials, seed=0)
        elapsed = time.time() - t0
        tuning_summary[method] = {"best_params": best_params, "best_val_score": best_val_score,
                                   "tuning_seconds": round(elapsed, 1)}
        print(f"[{task}] {method}: tuned in {elapsed:.1f}s, "
              f"val {'AUROC' if is_clf else 'neg-MAE'}={best_val_score:.4f}, params={best_params}")

        seed_results = dm.multi_seed_test_eval(
            method, task, best_params, data["X_trainval"], data["y_trainval"],
            data["X_test"], data["y_test"], seeds=seeds)
        multiseed_rows.extend(seed_results)

    results_df = pd.DataFrame(multiseed_rows)

    # pick best method by mean primary metric on test (multi-seed)
    agg = results_df.groupby("method")[primary_metric].mean()
    best_method = agg.idxmax() if better == "max" else agg.idxmin()

    # pairwise statistical tests (paired across seeds) vs. best method, Holm correction.
    # Pivot to seed x method explicitly (rather than relying on row order) so
    # the pairing is correct even if results were produced out of order.
    wide = results_df.pivot_table(index="seed", columns="method", values=primary_metric)
    pvals = {}
    for method in methods:
        if method == best_method:
            continue
        paired = wide[[best_method, method]].dropna()
        a = paired[best_method].values
        b = paired[method].values
        if len(a) < 2:
            pvals[method] = float("nan")
            continue
        # Wilcoxon requires non-zero differences and enough samples to be well-defined
        diffs_nonzero = np.any(a != b)
        if len(a) >= 6 and diffs_nonzero:
            stat, p = stats.wilcoxon(a, b)
        else:
            stat, p = stats.ttest_rel(a, b)
        pvals[method] = float(p)
    if pvals:
        methods_sorted = sorted(pvals, key=lambda m: pvals[m])
        m = len(methods_sorted)
        holm = {}
        for i, meth in enumerate(methods_sorted):
            holm[meth] = min(1.0, pvals[meth] * (m - i))
        # enforce monotonicity of Holm-adjusted p-values
        running_max = 0.0
        for meth in methods_sorted:
            running_max = max(running_max, holm[meth])
            holm[meth] = running_max
    else:
        holm = {}

    test_summary_df = results_df.groupby("method").agg(["mean", "std"])
    test_summary_df.columns = [f"{metric}_{stat}" for metric, stat in test_summary_df.columns]
    test_summary = test_summary_df.to_dict(orient="index")  # {method: {"auroc_mean": ..., ...}}

    return {
        "task": task,
        "primary_metric": primary_metric,
        "tuning_summary": tuning_summary,
        "multiseed_results": results_df.to_dict(orient="records"),
        "test_summary": test_summary,
        "best_method": best_method,
        "pvalues_vs_best_holm": holm,
        "feature_names": data["feature_names"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="fast run to check correctness")
    args = ap.parse_args()

    methods = dm.METHODS
    if args.smoke:
        n_trials, seeds = 3, [0, 1]
    else:
        n_trials, seeds = 30, list(range(10))

    stop, train, val, test = load_splits()
    print(f"train={len(train)} val={len(val)} test={len(test)} rows")

    all_results = {}
    for task in ["T1", "T2"]:
        print(f"\n===== TASK {task} =====")
        data = prep_task(task, train, val, test)
        print(f"n_features={len(data['feature_names'])}: {data['feature_names']}")
        res = run_task(task, data, methods, n_trials=n_trials, seeds=seeds)
        all_results[task] = res
        print(f"[{task}] BEST METHOD = {res['best_method']}")

        best_method = res["best_method"]
        best_params_raw = res["tuning_summary"][best_method]["best_params"]
        params = dm.finalize_params(best_method, best_params_raw)
        final_model = dm.get_model(best_method, task, params, seed=0)
        final_model.fit(data["X_trainval"], data["y_trainval"])
        scoring = "roc_auc" if task == "T1" else "neg_mean_absolute_error"
        pi = permutation_importance(final_model, data["X_test"], data["y_test"],
                                     scoring=scoring, n_repeats=8, random_state=0, n_jobs=-1)
        imp_df = pd.DataFrame({"feature": data["feature_names"],
                                "importance": pi.importances_mean,
                                "importance_std": pi.importances_std}).sort_values("importance", ascending=False)
        suffix_tmp = "_smoke" if args.smoke else ""
        imp_df.to_csv(RESULTS_METRICS / f"feature_importance_{task}{suffix_tmp}.csv", index=False)

    suffix = "_smoke" if args.smoke else ""
    with open(RESULTS_METRICS / f"deviation_modeling_results{suffix}.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nSaved results.")


if __name__ == "__main__":
    main()
