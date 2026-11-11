"""
Generates every figure for the paper from the saved results/metrics files.
Run after run_deviation_modeling.py, run_timewindow_evaluation.py, and
run_sensitivity.py have all completed on the full (non-smoke) settings.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting_style import METHOD_COLORS, OPT_COLORS, PALETTE, apply_style, save

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
FIGS = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

METHOD_LABELS = {"linear": "Linear/Logistic", "random_forest": "Random Forest",
                  "xgboost": "XGBoost", "lightgbm": "LightGBM", "mlp": "MLP"}
OPT_LABELS = {"status_quo_rate": "Status quo\n(historical)", "equal_split_rate": "Equal split\n(recentred)",
              "recenter_only_rate": "Recentred,\nsame widths", "greedy_rate": "Greedy\n(heuristic)",
              "ga_rate": "Genetic\nalgorithm", "optimized_rate": "Compliance-aware\n(ours)",
              "oracle_rate": "Oracle\n(hindsight)"}


def fig_model_comparison(results: dict, task: str, metric: str, ylabel: str, fname: str, higher_better: bool):
    df = pd.DataFrame(results[task]["multiseed_results"])
    order = df.groupby("method")[metric].mean().sort_values(ascending=not higher_better).index.tolist()
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    data = [df.loc[df.method == m, metric].values for m in order]
    bp = ax.boxplot(data, tick_labels=[METHOD_LABELS[m] for m in order], patch_artist=True, widths=0.55,
                     medianprops=dict(color="black", linewidth=1.2), showmeans=True,
                     meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=4))
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(METHOD_COLORS[m])
        patch.set_alpha(0.75)
    best = results[task]["best_method"]
    if best in order:
        ax.get_xticklabels()[order.index(best)].set_fontweight("bold")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Task {task[-1]}: {'sequence-reorder classification' if task=='T1' else 'arrival-time regression'} "
                 f"({len(df['seed'].unique())} seeds)", fontsize=9)
    plt.xticks(rotation=0, fontsize=8)
    fig.tight_layout()
    save(fig, FIGS / fname)
    plt.close(fig)


def fig_optimization_comparison(tw_summary: dict):
    means = tw_summary["summary_mean"]
    stds = tw_summary["summary_std"]
    cols = ["status_quo_rate", "equal_split_rate", "recenter_only_rate", "greedy_rate",
            "ga_rate", "optimized_rate", "oracle_rate"]
    vals = [means[c] * 100 for c in cols]
    errs = [stds[c] * 100 for c in cols]
    colors = [OPT_COLORS[c.replace("_rate", "")] for c in cols]

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(len(cols))
    ax.bar(x, vals, yerr=errs, capsize=3, color=colors, alpha=0.85,
           error_kw=dict(elinewidth=1, ecolor="black"))
    for xi, v in zip(x, vals):
        ax.text(xi, v + 1.0, f"{v:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([OPT_LABELS[c] for c in cols], fontsize=8)
    ax.set_ylabel("Predicted on-time rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Time-window budget allocation: predicted on-time rate by strategy\n"
                  "(same total promised-window budget in every strategy)", fontsize=9)
    fig.tight_layout()
    save(fig, FIGS / "fig_optimization_comparison")
    plt.close(fig)


def fig_sensitivity(sens: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))

    budget = sens[sens.sweep == "budget_scale"]
    g = budget.groupby("budget_scale").agg(opt=("optimized_rate", "mean"), opt_sd=("optimized_rate", "std"),
                                            sq=("status_quo_rate", "mean"))
    axes[0].errorbar(g.index, g["opt"] * 100, yerr=g["opt_sd"] * 100, marker="o", color=PALETTE["blue"],
                      label="Compliance-aware (ours)", capsize=3)
    axes[0].plot(g.index, g["sq"] * 100, marker="s", color=PALETTE["grey"], linestyle="--", label="Status quo")
    axes[0].set_xlabel("Budget scale (x historical total window width)")
    axes[0].set_ylabel("Predicted on-time rate (%)")
    axes[0].set_title("(a) Sensitivity to budget generosity", fontsize=9)
    axes[0].legend(fontsize=7, frameon=False)

    strata = sens[sens.sweep == "n_strata"]
    g2 = strata.groupby("n_strata").agg(opt=("optimized_rate", "mean"), opt_sd=("optimized_rate", "std"))
    axes[1].errorbar(g2.index, g2["opt"] * 100, yerr=g2["opt_sd"] * 100, marker="o", color=PALETTE["blue"], capsize=3)
    axes[1].set_xlabel("Number of risk strata in residual calibrator")
    axes[1].set_ylabel("Predicted on-time rate (%)")
    axes[1].set_title("(b) Sensitivity to risk-stratification granularity", fontsize=9)

    fig.tight_layout()
    save(fig, FIGS / "fig_sensitivity")
    plt.close(fig)


def fig_deviation_overview(stop: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0))

    route_dev = stop.groupby(["Route ID", "Week ID"])["reordered"].mean() * 100
    axes[0].hist(route_dev, bins=30, color=PALETTE["blue"], alpha=0.85)
    axes[0].set_xlabel("Stops reordered per route (%)")
    axes[0].set_ylabel("Number of routes")
    axes[0].set_title("(a) Route-level deviation", fontsize=9)

    delivery = stop[stop["Delivery"] == 1]
    counts = [delivery["on_time"].mean() * 100, delivery["early"].mean() * 100, delivery["late"].mean() * 100]
    axes[1].bar(["On time", "Early", "Late"], counts, color=[PALETTE["green"], PALETTE["sky"], PALETTE["vermillion"]], alpha=0.85)
    for i, v in enumerate(counts):
        axes[1].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=8)
    axes[1].set_ylabel("Share of deliveries (%)")
    axes[1].set_title("(b) Time-window compliance", fontsize=9)
    axes[1].set_ylim(0, 100)

    week_dev = stop.groupby("Week ID")["reordered"].mean() * 100
    axes[2].plot(week_dev.index, week_dev.values, marker="o", color=PALETTE["blue"], markersize=3)
    axes[2].axvspan(19, 24, color=PALETTE["grey"], alpha=0.15, label="Validation")
    axes[2].axvspan(25, week_dev.index.max(), color=PALETTE["orange"], alpha=0.15, label="Test")
    axes[2].set_xlabel("Week")
    axes[2].set_ylabel("Stops reordered (%)")
    axes[2].set_title("(c) Deviation rate over time", fontsize=9)
    axes[2].legend(fontsize=7, frameon=False)

    fig.tight_layout()
    save(fig, FIGS / "fig_deviation_overview")
    plt.close(fig)


def fig_ablation_features(ablation: dict):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    variant_order = ["full", "no_addr_history", "no_driver_history", "no_history_all"]
    variant_labels = {"full": "Full\nfeature set", "no_addr_history": "No address\nhistory",
                       "no_driver_history": "No driver\nhistory", "no_history_all": "No history\n(cold start)"}
    colors = [PALETTE["blue"], PALETTE["sky"], PALETTE["orange"], PALETTE["vermillion"]]

    panels = [(axes[0], "T1", "AUROC (higher is better)"), (axes[1], "T2", "MAE, minutes (lower is better)")]
    for ax, task, ylabel in panels:
        summ = ablation[task]["summary"]
        metric = ablation[task]["primary_metric"]
        vals = [summ[v][f"{metric}_mean"] for v in variant_order]
        errs = [summ[v][f"{metric}_std"] for v in variant_order]
        x = np.arange(len(variant_order))
        ax.bar(x, vals, yerr=errs, capsize=3, color=colors, alpha=0.85,
               error_kw=dict(elinewidth=1, ecolor="black"))
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.3f}" if task == "T1" else f"{v:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([variant_labels[v] for v in variant_order], fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"Task {task[-1]} ({ablation[task]['best_method']})", fontsize=9)

    fig.suptitle("Ablation: contribution of leak-free driver x address familiarity features", fontsize=9.5, y=1.02)
    fig.tight_layout()
    save(fig, FIGS / "fig_ablation_features")
    plt.close(fig)


def fig_feature_importance(importances: pd.DataFrame, task: str, fname: str):
    top = importances.sort_values("importance", ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(top["feature"], top["importance"], color=PALETTE["blue"], alpha=0.85)
    ax.set_xlabel("Permutation importance")
    ax.set_title(f"Feature importance — best model, Task {task[-1]}", fontsize=9)
    fig.tight_layout()
    save(fig, FIGS / fname)
    plt.close(fig)


def main():
    apply_style()
    with open(METRICS / "deviation_modeling_results.json") as f:
        dev_results = json.load(f)
    fig_model_comparison(dev_results, "T1", "auroc", "AUROC (higher is better)", "fig_t1_model_comparison", True)
    fig_model_comparison(dev_results, "T2", "mae", "MAE, minutes (lower is better)", "fig_t2_model_comparison", False)

    with open(METRICS / "timewindow_evaluation.json") as f:
        tw = json.load(f)
    fig_optimization_comparison(tw)

    sens = pd.read_csv(METRICS / "sensitivity_analysis.csv")
    fig_sensitivity(sens)

    stop = pd.read_parquet(ROOT / "data" / "processed" / "stop_table.parquet")
    fig_deviation_overview(stop)

    for task, fname in [("T1", "fig_t1_feature_importance"), ("T2", "fig_t2_feature_importance")]:
        p = METRICS / f"feature_importance_{task}.csv"
        if p.exists():
            fig_feature_importance(pd.read_csv(p), task, fname)

    p = METRICS / "ablation_features_results.json"
    if p.exists():
        with open(p) as f:
            fig_ablation_features(json.load(f))

    print("All figures written to", FIGS)


if __name__ == "__main__":
    main()
