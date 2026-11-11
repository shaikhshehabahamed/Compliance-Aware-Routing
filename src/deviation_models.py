"""
Model zoo, hyperparameter tuning, and multi-seed evaluation for the two
prediction tasks (T1: sequence-reorder classification, T2: arrival-time
regression). One shared interface is used for every method so the model
comparison is apples-to-apples (same features, same splits, same metrics).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import optuna
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score,
                              brier_score_loss, f1_score, mean_absolute_error,
                              mean_squared_error, r2_score, roc_auc_score)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

METHODS = ["linear", "random_forest", "xgboost", "lightgbm", "mlp"]


def get_model(method: str, task: str, params: dict, seed: int):
    is_clf = task == "T1"
    if method == "linear":
        if is_clf:
            base = LogisticRegression(max_iter=2000, random_state=seed, **params)
        else:
            base = Ridge(random_state=seed, **params)
        return Pipeline([("scaler", StandardScaler()), ("model", base)])
    if method == "random_forest":
        cls = RandomForestClassifier if is_clf else RandomForestRegressor
        return cls(random_state=seed, n_jobs=-1, **params)
    if method == "xgboost":
        cls = XGBClassifier if is_clf else XGBRegressor
        extra = dict(random_state=seed, n_jobs=-1, tree_method="hist")
        if is_clf:
            extra["eval_metric"] = "logloss"
        return cls(**extra, **params)
    if method == "lightgbm":
        cls = LGBMClassifier if is_clf else LGBMRegressor
        return cls(random_state=seed, n_jobs=-1, verbosity=-1, **params)
    if method == "mlp":
        if is_clf:
            base = MLPClassifier(random_state=seed, max_iter=500, early_stopping=True, **params)
            return Pipeline([("scaler", StandardScaler()), ("model", base)])
        # regression MLPs are sensitive to target scale (unscaled arrival-time
        # targets, O(100)-O(1000) minutes, slow gradient-based convergence
        # badly); scale y internally and invert on prediction, matching
        # standard practice for neural nets, so the comparison to the other
        # (scale-invariant) methods is fair rather than handicapped.
        inner = MLPRegressor(random_state=seed, max_iter=500, early_stopping=True, **params)
        pipe = Pipeline([("scaler", StandardScaler()), ("model", inner)])
        return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())
    raise ValueError(method)


def suggest_params(method: str, task: str, trial: optuna.Trial) -> dict:
    is_clf = task == "T1"
    if method == "linear":
        if is_clf:
            return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True),
                     "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])}
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}
    if method == "random_forest":
        # depth/n_estimators capped for tractable fit times on ~150k rows;
        # gradient boosting (xgboost/lightgbm) covers the higher-capacity
        # end of the search more efficiently, so RF is kept as the
        # bagging-family baseline rather than pushed to its slowest regime.
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 350, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 14),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }
    if method == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 700, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        }
    if method == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 700, step=50),
            "num_leaves": trial.suggest_int("num_leaves", 15, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }
    if method == "mlp":
        return {
            "n_layers": trial.suggest_int("n_layers", 1, 3),
            "width": trial.suggest_categorical("width", [32, 64, 128]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        }
    raise ValueError(method)


def finalize_params(method: str, raw_params: dict) -> dict:
    """Convert a raw Optuna param dict (exactly what trial.suggest_*
    produced, i.e. what study.best_params returns) into the kwargs actually
    accepted by the corresponding sklearn-style estimator."""
    if method == "mlp":
        p = dict(raw_params)
        n_layers = p.pop("n_layers")
        width = p.pop("width")
        p["hidden_layer_sizes"] = tuple([width] * n_layers)
        return p
    return dict(raw_params)


def _predict_proba_or_value(model, X, is_clf: bool):
    if is_clf:
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def classification_metrics(y_true, p_pred, threshold: float = 0.5) -> dict:
    y_hat = (p_pred >= threshold).astype(int)
    return {
        "auroc": roc_auc_score(y_true, p_pred),
        "auprc": average_precision_score(y_true, p_pred),
        "brier": brier_score_loss(y_true, p_pred),
        "f1": f1_score(y_true, y_hat),
        "accuracy": accuracy_score(y_true, y_hat),
    }


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }


def tune_method(method: str, task: str, X_train, y_train, X_val, y_val,
                 n_trials: int = 30, seed: int = 0) -> tuple[dict, float, "optuna.Study"]:
    is_clf = task == "T1"

    def objective(trial: optuna.Trial) -> float:
        raw_params = suggest_params(method, task, trial)
        params = finalize_params(method, raw_params)
        model = get_model(method, task, params, seed=seed)
        model.fit(X_train, y_train)
        pred = _predict_proba_or_value(model, X_val, is_clf)
        if is_clf:
            return roc_auc_score(y_val, pred)
        return -mean_absolute_error(y_val, pred)  # maximize negative MAE

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value, study


def multi_seed_test_eval(method: str, task: str, best_params_raw: dict,
                          X_trainval, y_trainval, X_test, y_test,
                          seeds: list[int]) -> list[dict]:
    is_clf = task == "T1"
    params = finalize_params(method, best_params_raw)
    results = []
    for seed in seeds:
        model = get_model(method, task, params, seed=seed)
        model.fit(X_trainval, y_trainval)
        pred = _predict_proba_or_value(model, X_test, is_clf)
        m = classification_metrics(y_test, pred) if is_clf else regression_metrics(y_test, pred)
        m["seed"] = seed
        m["method"] = method
        results.append(m)
    return results
