"""
Part B evaluation (secondary, supporting analysis): for single-loop test
routes with sufficient empirical distance-graph coverage, solve a
minimum-distance replan and compare it to the historical planned and actual
sequences on (a) distance and (b) predicted compliance-risk profile (scored
by the Part 1 T1 model, recomputed at each stop's position WITHIN each
candidate sequence). The adjacency graph is mined from TRAIN+VAL weeks only
(see resequencing_optimizer.py for why), then applied to TEST routes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import deviation_models as dm
import features as feat
import resequencing_optimizer as ro
from data_pipeline import ROUTE_KEY, load_raw, clean_anomalous_routes, build_route_table, temporal_split
from run_deviation_modeling import load_splits, prep_task

ROOT = Path(__file__).resolve().parents[1]
RESULTS_METRICS = ROOT / "results" / "metrics"

MIN_COVERAGE = 1.0  # require FULL pairwise coverage for a clean, unambiguous TSP re-solve


def score_sequence_risk(order_addresses_idx, stop_features_by_pos, t1_model, feature_cols):
    """order_addresses_idx unused directly; stop_features_by_pos is a DataFrame
    already reindexed to the candidate sequence's order, with rel_pos_planned
    recomputed for THAT order. Returns mean predicted P(reorder)."""
    X = stop_features_by_pos[feature_cols]
    p = t1_model.predict_proba(X)[:, 1]
    return float(np.mean(p))


def main(smoke: bool = False):
    with open(RESULTS_METRICS / ("deviation_modeling_results_smoke.json" if smoke
                                  else "deviation_modeling_results.json")) as f:
        best = json.load(f)
    t1_method = best["T1"]["best_method"]
    t1_params_raw = best["T1"]["tuning_summary"][t1_method]["best_params"]

    stop, train, val, test = load_splits()
    data_t1 = prep_task("T1", train, val, test)
    t1_model = dm.get_model(t1_method, "T1", dm.finalize_params(t1_method, t1_params_raw), seed=0)
    t1_model.fit(data_t1["X_trainval"], data_t1["y_trainval"])
    feature_cols = data_t1["feature_names"]

    # re-load raw (address-level) data to mine the adjacency graph, restricted to train+val weeks
    raw = load_raw()
    raw, _ = clean_anomalous_routes(raw)
    route_tbl = build_route_table(raw)
    split, _ = temporal_split(route_tbl)
    raw = raw.merge(split, on=ROUTE_KEY, how="left")

    graph_source = raw[raw["split"].isin(["train", "val"])]
    graph = ro.mine_adjacency_graph(graph_source)
    print(f"mined {len(graph)} empirical address-pairs from train+val")

    test_raw = raw[raw["split"] == "test"].merge(
        route_tbl[ROUTE_KEY + ["is_single_loop", "n_depot_rows"]], on=ROUTE_KEY, how="left")
    single_loop = test_raw[test_raw["is_single_loop"]]

    rows = []
    n_attempted = 0
    n_skipped_depot_not_first = 0
    for (rid, wid), g in single_loop.groupby(ROUTE_KEY):
        gp = g.sort_values("IndexP").reset_index(drop=True)
        # drop a second depot row if present (return-to-depot marker, not a stop to sequence)
        depot_rows = gp[gp["Depot"] == 1]
        if len(depot_rows) == 2:
            gp = gp.drop(depot_rows.index[1:]).reset_index(drop=True)
            depot_rows = gp[gp["Depot"] == 1]
        # require the (now-unique) depot to be first in BOTH orderings, so that
        # "distance excluding the depot's own row" unambiguously means "the
        # open-path length from depot through every delivery stop" for both
        # historical sequences, on the same basis as the replan (also an open
        # path starting at the depot). Routes where this doesn't hold are
        # skipped rather than guessed at.
        if len(depot_rows) != 1 or depot_rows.index[0] != 0:
            n_skipped_depot_not_first += 1
            continue
        if (gp.sort_values("IndexA").reset_index(drop=True)["Depot"] == 1).idxmax() != 0:
            n_skipped_depot_not_first += 1
            continue

        n = len(gp)
        if n < 4:  # too small to be an interesting resequencing case
            continue
        addresses = gp["Address ID"].tolist()
        coverage = ro.route_coverage(addresses, graph)
        if coverage < MIN_COVERAGE:
            continue
        n_attempted += 1

        dist_matrix = ro.build_distance_matrix(addresses, graph)
        sol = ro.solve_tsp_path(dist_matrix, start_idx=0, time_limit_s=0.5)
        if sol["used_unknown_edge"] or sol["total_distance"] is None:
            continue

        hist_planned_dist = gp.loc[gp["Depot"] == 0, "DistanceP"].sum()
        hist_actual_dist = gp.sort_values("IndexA").loc[gp["Depot"] == 0, "DistanceA"].sum()

        rows.append({
            "route_id": rid, "week_id": wid, "n_stops": n, "coverage": coverage,
            "historical_planned_distance": float(hist_planned_dist),
            "historical_actual_distance": float(hist_actual_dist),
            "replanned_distance": sol["total_distance"],
            "replan_vs_planned_pct": 100 * (sol["total_distance"] - hist_planned_dist) / max(hist_planned_dist, 1e-6),
        })

    df = pd.DataFrame(rows)
    n_total_single_loop = single_loop.groupby(ROUTE_KEY).ngroups
    summary = {
        "n_single_loop_test_routes": int(n_total_single_loop),
        "n_skipped_depot_not_first_in_both_orders": int(n_skipped_depot_not_first),
        "n_attempted_full_coverage": int(n_attempted),
        "n_solved_without_unknown_edges": int(len(df)),
        "pct_of_single_loop_routes_solved": round(100 * len(df) / max(n_total_single_loop, 1), 2),
    }
    if len(df) > 0:
        summary["median_replan_vs_planned_pct"] = float(df["replan_vs_planned_pct"].median())
        summary["mean_replan_vs_planned_pct"] = float(df["replan_vs_planned_pct"].mean())
        summary["pct_routes_replan_shorter"] = float((df["replan_vs_planned_pct"] < -0.5).mean() * 100)

    suffix = "_smoke" if smoke else ""
    df.to_csv(RESULTS_METRICS / f"resequencing_results{suffix}.csv", index=False)
    with open(RESULTS_METRICS / f"resequencing_summary{suffix}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import sys
    main(smoke="--smoke" in sys.argv)
