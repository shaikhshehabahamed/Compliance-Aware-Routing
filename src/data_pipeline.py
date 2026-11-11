"""
Data pipeline for the compliance-aware routing project.

Source data: "Last-mile delivery route deviations dataset: planned vs. actual
routes" (Konovalenko, Hvattum & Iversen; Molde University College; Mendeley
Data DOI 10.17632/kkwgfvmtxn.1; CC BY 4.0).

Key structural facts established by direct inspection of the raw file
(see _verify_structure.py for the verification run):
  - The atomic route instance is uniquely identified by (Route ID, Week ID).
    This key determines a single Driver ID and a single Country in 100% of
    groups (checked). "Day of Week" is a per-stop attribute (most likely a
    customer's standing delivery-day label), NOT the calendar day the route
    was physically driven, and must not be used to further split routes.
  - IndexP / IndexA (planned / actual sequence position) are usually, but not
    always, a clean 0..n-1 permutation within a route instance -- roughly
    4-5% of instances have gaps, most likely because some stops present in
    the operational system were not included in this data export. We
    therefore always treat IndexP/IndexA as ORDINAL ranks (order matters,
    literal contiguity does not), which is robust to this.
  - Multiple depot visits within one (Route ID, Week ID) occur for ~7% of
    instances (multi-trip routes, e.g. the driver reloads at the depot and
    goes out again). We keep these in the stop-level tables but restrict the
    resequencing optimizer (Part B) to single-loop routes (<=2 depot rows).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "routes_performance.csv"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ROUTE_KEY = ["Route ID", "Week ID"]

REQUIRED_COLUMNS = [
    "Route ID", "Driver ID", "Stop ID", "Address ID", "Week ID", "Country",
    "Day of Week", "IndexP", "IndexA", "Arrived Time", "Earliest Time",
    "Latest Time", "DistanceP", "DistanceA", "Depot", "Delivery",
]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Raw file is missing expected columns: {missing}")
    n_before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    n_after = len(df)
    if n_before != n_after:
        print(f"[load_raw] dropped {n_before - n_after} rows with missing values "
              f"in required columns ({n_before} -> {n_after})")
    for c in ["Route ID", "Driver ID", "Stop ID", "Address ID", "Week ID",
              "Country", "IndexP", "IndexA", "Depot", "Delivery"]:
        df[c] = df[c].astype(int)
    return df.reset_index(drop=True)


def _rank_within_group(s: pd.Series) -> pd.Series:
    """Dense rank (0-indexed) -- robust to gaps/ties in the raw index."""
    return s.rank(method="first").astype(int) - 1


ANOMALY_HORIZON_MIN = 2880  # 48 hours; Time fields are offsets from route start


def clean_anomalous_routes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop route instances that are not plausible single/next-day local
    delivery rounds. "Local Delivery" (the dataset's own Mendeley category
    tag) rounds should complete within a couple of days of their own start;
    a small fraction (~2%) of route instances have a promised time-window
    edge or a realized arrival more than 48h after the route's own t=0,
    which is far outside any plausible same-/next-day operation and is far
    more consistent with a data/export artifact (e.g. a standing multi-week
    order picked up by this extract) than a real execution event. We drop
    the WHOLE route instance (not just the offending row) so that no partial,
    internally-inconsistent route survives into modelling or optimization.
    The threshold and rationale are reported in full so the choice is
    auditable, not just applied silently."""
    window_width = (df["Latest Time"] - df["Earliest Time"]).clip(lower=0)
    bad_mask = (window_width > ANOMALY_HORIZON_MIN) | (df["Arrived Time"] > ANOMALY_HORIZON_MIN)
    bad_keys = df.loc[bad_mask, ROUTE_KEY].drop_duplicates()
    n_routes_before = df.groupby(ROUTE_KEY).ngroups
    df_clean = df.merge(bad_keys.assign(_bad=1), on=ROUTE_KEY, how="left")
    n_rows_dropped = int((df_clean["_bad"] == 1).sum())
    df_clean = df_clean.loc[df_clean["_bad"].isna()].drop(columns=["_bad"]).reset_index(drop=True)
    n_routes_after = df_clean.groupby(ROUTE_KEY).ngroups
    report = {
        "rule": f"drop route if any stop has (Latest-Earliest)>{ANOMALY_HORIZON_MIN}min "
                f"or Arrived Time>{ANOMALY_HORIZON_MIN}min",
        "n_routes_before": int(n_routes_before),
        "n_routes_dropped": int(n_routes_before - n_routes_after),
        "pct_routes_dropped": round(100 * (n_routes_before - n_routes_after) / n_routes_before, 3),
        "n_rows_dropped": n_rows_dropped,
    }
    print(f"[clean_anomalous_routes] dropped {report['n_routes_dropped']} / {n_routes_before} "
          f"route instances ({report['pct_routes_dropped']}%), {n_rows_dropped} rows")
    return df_clean, report


def build_route_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (Route ID, Week ID) route instance."""
    g = df.groupby(ROUTE_KEY, sort=False)
    route = g.agg(
        driver_id=("Driver ID", "first"),
        country=("Country", "first"),
        n_stops=("Stop ID", "size"),
        n_depot_rows=("Depot", "sum"),
        n_delivery_rows=("Delivery", "sum"),
        planned_total_distance=("DistanceP", "sum"),
        actual_total_distance=("DistanceA", "sum"),
    ).reset_index()
    route["is_single_loop"] = route["n_depot_rows"] <= 2
    return route


def build_stop_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per stop-visit with engineered, leak-free features and targets."""
    df = df.copy()
    df["rankP"] = df.groupby(ROUTE_KEY)["IndexP"].transform(_rank_within_group)
    df["rankA"] = df.groupby(ROUTE_KEY)["IndexA"].transform(_rank_within_group)
    df["n_stops"] = df.groupby(ROUTE_KEY)["Stop ID"].transform("size")
    df["rel_pos_planned"] = df["rankP"] / (df["n_stops"] - 1).clip(lower=1)

    # --- targets -----------------------------------------------------
    df["reordered"] = (df["rankP"] != df["rankA"]).astype(int)
    df["position_shift"] = (df["rankA"] - df["rankP"]).astype(int)
    df["abs_position_shift"] = df["position_shift"].abs()
    # normalised shift magnitude (0-1), robust to route length
    df["rel_abs_shift"] = df["abs_position_shift"] / (df["n_stops"] - 1).clip(lower=1)

    # arrival-time deviation target: Arrived Time is already an offset (minutes)
    # from the route's earliest timestamp; window midpoint gives a natural
    # "on schedule" reference so the regression target is comparable across
    # routes/stops with different absolute window placements.
    df["window_width"] = (df["Latest Time"] - df["Earliest Time"]).clip(lower=0)
    df["window_mid"] = (df["Earliest Time"] + df["Latest Time"]) / 2.0
    df["arrival_offset_from_mid"] = df["Arrived Time"] - df["window_mid"]
    df["on_time"] = ((df["Arrived Time"] >= df["Earliest Time"]) &
                      (df["Arrived Time"] <= df["Latest Time"])).astype(int)
    df["late"] = (df["Arrived Time"] > df["Latest Time"]).astype(int)
    df["early"] = (df["Arrived Time"] < df["Earliest Time"]).astype(int)

    # --- leak-free historical (panel) features ------------------------
    # For every stop instance, use ONLY prior weeks (WeekID < this WeekID) to
    # compute historical familiarity / deviation-rate features. Implemented
    # via expanding aggregation ordered by Week ID, then shifted by one
    # occurrence so the current row's own outcome never leaks into its own
    # feature.
    df = df.sort_values(["Address ID", "Week ID"]).reset_index(drop=True)
    grp_addr = df.groupby("Address ID")
    df["addr_visit_seq"] = grp_addr.cumcount()  # 0 = first time this address is seen
    df["addr_hist_reorder_rate"] = (
        grp_addr["reordered"].apply(lambda s: s.shift().expanding().mean())
        .reset_index(level=0, drop=True)
    )
    df["addr_hist_ontime_rate"] = (
        grp_addr["on_time"].apply(lambda s: s.shift().expanding().mean())
        .reset_index(level=0, drop=True)
    )

    df = df.sort_values(["Driver ID", "Week ID"]).reset_index(drop=True)
    grp_drv = df.groupby("Driver ID")
    df["driver_visit_seq"] = grp_drv.cumcount()
    df["driver_hist_reorder_rate"] = (
        grp_drv["reordered"].apply(lambda s: s.shift().expanding().mean())
        .reset_index(level=0, drop=True)
    )

    df = df.sort_values(["Driver ID", "Address ID", "Week ID"]).reset_index(drop=True)
    grp_da = df.groupby(["Driver ID", "Address ID"])
    df["driver_addr_visit_seq"] = grp_da.cumcount()  # familiarity: this driver x this stop

    # fill first-occurrence NaNs (no history yet) with the population mean
    # computed on training-eligible rows only happens later in the split step;
    # here we just leave NaN and impute inside the modelling pipeline so no
    # information from later splits leaks into the imputation value.
    df = df.sort_values(ROUTE_KEY + ["rankP"]).reset_index(drop=True)
    return df


def temporal_split(route: pd.DataFrame, train_frac: float = 0.60, val_frac: float = 0.20):
    """Chronological split by Week ID: train on earliest weeks, tune on the
    next slice, evaluate once on the latest, untouched weeks. This avoids
    both route-level leakage (a route never appears in two splits) and
    temporal leakage (no feature or model ever sees the future)."""
    weeks = np.sort(route["Week ID"].unique())
    n = len(weeks)
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac)))
    train_weeks = weeks[:n_train]
    val_weeks = weeks[n_train:n_train + n_val]
    test_weeks = weeks[n_train + n_val:]

    conditions = [
        route["Week ID"].isin(train_weeks),
        route["Week ID"].isin(val_weeks),
        route["Week ID"].isin(test_weeks),
    ]
    split_col = np.select(conditions, ["train", "val", "test"], default="unassigned")
    out = route[ROUTE_KEY].copy()
    out["split"] = split_col
    return out, {
        "train_weeks": train_weeks.tolist(),
        "val_weeks": val_weeks.tolist(),
        "test_weeks": test_weeks.tolist(),
    }


def run(save: bool = True):
    df = load_raw()
    df, cleaning_report = clean_anomalous_routes(df)
    route = build_route_table(df)
    stop = build_stop_table(df)
    split, week_info = temporal_split(route)
    route = route.merge(split, on=ROUTE_KEY, how="left")
    stop = stop.merge(split, on=ROUTE_KEY, how="left")

    summary = {
        "cleaning_report": cleaning_report,
        "n_rows_raw": int(len(df)),
        "n_routes": int(len(route)),
        "n_drivers": int(df["Driver ID"].nunique()),
        "n_addresses": int(df["Address ID"].nunique()),
        "n_weeks": int(df["Week ID"].nunique()),
        "n_countries": int(df["Country"].nunique()),
        "route_size_mean": float(route["n_stops"].mean()),
        "route_size_median": float(route["n_stops"].median()),
        "pct_single_loop_routes": float(route["is_single_loop"].mean() * 100),
        "reorder_rate_mean": float(stop["reordered"].mean()),
        "pct_routes_perfect_compliance": float(
            (stop.groupby(ROUTE_KEY)["reordered"].sum() == 0).mean() * 100),
        "on_time_rate": float(stop.loc[stop["Delivery"] == 1, "on_time"].mean()),
        "late_rate": float(stop.loc[stop["Delivery"] == 1, "late"].mean()),
        "early_rate": float(stop.loc[stop["Delivery"] == 1, "early"].mean()),
        "split_route_counts": route["split"].value_counts().to_dict(),
        "week_info": week_info,
    }

    if save:
        route.to_parquet(PROCESSED_DIR / "route_table.parquet", index=False)
        stop.to_parquet(PROCESSED_DIR / "stop_table.parquet", index=False)
        with open(PROCESSED_DIR / "pipeline_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
    return route, stop, summary


if __name__ == "__main__":
    route, stop, summary = run()
    print(json.dumps(summary, indent=2, default=str))
