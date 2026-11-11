"""
Feature definitions for the two predictive targets.

Design principle (important for correctness of the downstream optimizer):
  - T1 ("reordered"): a standalone predictive task -- may use any feature
    that is set at planning time, INCLUDING the assigned time window, since
    a wide window plausibly gives a driver more real slack to deviate from
    the planned order. This task is not used inside a counterfactual
    "what if the window were different" evaluation, so there is no leakage
    risk in using window features here.
  - T2 ("arrival time"): feeds directly into the compliance-aware
    time-window optimizer, which asks "what would happen under a DIFFERENT
    window than the one historically assigned?". If the model were trained
    with the historical window as an input, it could not be used to score
    counterfactual windows (the very thing we need it for). T2 therefore
    deliberately EXCLUDES Earliest Time / Latest Time / window_width /
    window_mid and anything derived from the actually-realized route
    (IndexA, DistanceA, rankA, position_shift, on_time/late/early), and is
    trained only on features fixed at planning time.

All historical/familiarity features are built as expanding, past-only
aggregates in data_pipeline.py (see addr_hist_*, driver_hist_*), so no
target information from the current or any future row ever enters a
feature value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL = ["Country", "Day of Week"]

BASE_NUMERIC = [
    "rel_pos_planned", "DistanceP", "n_stops", "Week ID", "Depot", "Delivery",
    "addr_visit_seq", "addr_hist_reorder_rate", "addr_hist_ontime_rate",
    "driver_visit_seq", "driver_hist_reorder_rate", "driver_addr_visit_seq",
]

T1_EXTRA_NUMERIC = ["window_width", "window_mid"]  # planning-time info, OK for T1 only

T1_TARGET = "reordered"
T2_TARGET = "Arrived Time"

# Named feature groups used for the ablation study (run_ablation_features.py).
# These isolate the paper's core feature-engineering contribution: leak-free,
# past-only "familiarity" aggregates, split by whose history they summarise.
ADDR_HISTORY_FEATURES = ["addr_hist_reorder_rate", "addr_hist_ontime_rate", "addr_visit_seq"]
DRIVER_HISTORY_FEATURES = ["driver_hist_reorder_rate", "driver_visit_seq", "driver_addr_visit_seq"]
ALL_HISTORY_FEATURES = ADDR_HISTORY_FEATURES + DRIVER_HISTORY_FEATURES


def _encode(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    X = df[numeric_cols].copy()
    for c in CATEGORICAL:
        dummies = pd.get_dummies(df[c], prefix=c, dtype=float)
        X = pd.concat([X, dummies], axis=1)
    return X


def build_xy(stop: pd.DataFrame, task: str, drop_features: list[str] | None = None):
    """task in {'T1', 'T2'}. Returns (X, y). `drop_features` optionally removes
    named numeric columns before encoding -- used only by the feature-group
    ablation study; every other caller omits it and gets the full feature set."""
    if task == "T1":
        numeric_cols = BASE_NUMERIC + T1_EXTRA_NUMERIC
        y = stop[T1_TARGET].astype(int).values
    elif task == "T2":
        numeric_cols = BASE_NUMERIC
        y = stop[T2_TARGET].astype(float).values
    else:
        raise ValueError(task)
    if drop_features:
        numeric_cols = [c for c in numeric_cols if c not in drop_features]
    X = _encode(stop, numeric_cols)
    return X, y


def impute_fit(X_train: pd.DataFrame) -> dict:
    """Median imputation fitted on TRAIN ONLY, applied identically to
    val/test to avoid any leakage of val/test statistics."""
    return X_train.median(numeric_only=True).to_dict()


def impute_apply(X: pd.DataFrame, medians: dict) -> pd.DataFrame:
    return X.fillna(medians)
