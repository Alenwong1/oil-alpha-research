"""Nested chronological model selection; holdout models are never refit."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def estimator(kind: str, parameter: float, seed: int):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=parameter))
    if kind == "tree":
        return HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=int(parameter),
            min_samples_leaf=40,
            l2_regularization=10.0,
            early_stopping=False,
            random_state=seed,
        )
    raise ValueError(kind)


def fit_selected(x: pd.DataFrame, meta: pd.DataFrame, kind: str, seed: int):
    if len(x) < 600:
        raise ValueError("At least 600 labeled training rows required")
    validation_start = x.index[-252]
    inner_train = (x.index < validation_start) & (meta.label_end < validation_start)
    validation = x.index >= validation_start
    parameters = [1.0, 100.0, 10000.0] if kind == "ridge" else [7, 15]
    scores = []
    with threadpool_limits(limits=1):
        for parameter in parameters:
            model = estimator(kind, parameter, seed)
            model.fit(x.loc[inner_train], meta.loc[inner_train, "target_return"])
            p = model.predict(x.loc[validation])
            mse = float(np.mean((p - meta.loc[validation, "target_return"].to_numpy()) ** 2))
            scores.append((parameter, mse))
        best = min(scores, key=lambda pair: pair[1])[0]
        model = estimator(kind, best, seed)
        model.fit(x, meta.target_return)
    audit = {
        "parameter": best,
        "candidate_validation_mse": json.dumps(scores),
        "training_rows": len(x),
        "training_start": str(x.index.min().date()),
        "training_signal_end": str(x.index.max().date()),
        "max_training_label_end": str(meta.label_end.max().date()),
        "validation_start": str(validation_start.date()),
        "max_inner_training_label_end": str(meta.loc[inner_train, "label_end"].max().date()),
        "validation_zero_forecast_mse": float(np.mean(meta.loc[validation, "target_return"] ** 2)),
    }
    return model, audit


def predict_period(x, meta, config, period):
    start = pd.Timestamp(
        config["development_start"] if period == "development" else config["holdout_start"]
    )
    end = pd.Timestamp(
        config["holdout_start"] if period == "development" else config["data_end_exclusive"]
    )
    # Keep every realized holding period strictly inside its evaluation partition.
    evaluation = (x.index >= start) & (meta.label_end < end)
    dates = x.index[evaluation]
    if len(dates) == 0:
        raise ValueError(f"No evaluation rows for {period}")
    periods = dates.to_period("Q")
    groups = (
        [(str(q), dates[periods == q]) for q in periods.unique()]
        if period == "development"
        else [("frozen_holdout", dates)]
    )
    predictions = pd.DataFrame(index=dates)
    audits = []
    variants = [
        ("ridge_uso", "ridge", [c for c in x if c.startswith(config["target"] + "_")]),
        ("ridge_full", "ridge", list(x)),
        ("tree_uso", "tree", [c for c in x if c.startswith(config["target"] + "_")]),
        ("tree_full", "tree", list(x)),
    ]
    for fold, test_dates in groups:
        # Strict inequality deliberately excludes outcomes ending on the first signal day.
        cutoff = test_dates.min() if period == "development" else start
        eligible = (x.index < cutoff) & (meta.label_end < cutoff)
        print(f"{period} / {fold}: {eligible.sum()} train, {len(test_dates)} test", flush=True)
        for name, kind, columns in variants:
            model, audit = fit_selected(
                x.loc[eligible, columns], meta.loc[eligible], kind, config["seed"]
            )
            with threadpool_limits(limits=1):
                predictions.loc[test_dates, name] = model.predict(x.loc[test_dates, columns])
            audits.append(
                {
                    "period": period,
                    "fold": fold,
                    "model": name,
                    "feature_count": len(columns),
                    "test_start": str(test_dates.min().date()),
                    "test_end": str(test_dates.max().date()),
                    **audit,
                }
            )
    predictions.index.name = "signal_date"
    return predictions, pd.DataFrame(audits)
