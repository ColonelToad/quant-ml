"""
competitions/jane_street/scripts/train.py
-----------------------------------------
LightGBM baseline for Jane Street Real-Time Market Data Forecasting.

Key decisions:
- Walk-forward CV on date_id (temporal boundary, no shuffling)
- sample_weight = weight column (mandatory for this competition)
- Multi-task: train on responder_6 only for baseline,
  add auxiliary targets in next iteration
- Metric: weighted R² (what the competition scores)
- Drop first time_id per (date, symbol) — lag nulls
"""

import polars as pl
import numpy as np
import lightgbm as lgb
import pandas as pd
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[1] / "feature_cache"
PRIMARY_TARGET = "responder_6"
AUX_TARGETS = ["responder_3", "responder_7", "responder_8"]
ALL_RESPONDERS = [f"responder_{i}" for i in range(9)]

EXCLUDE_COLS = set(
    ["date_id", "time_id", "symbol_id", "weight"] + ALL_RESPONDERS
)


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray,
                weights: np.ndarray) -> float:
    """Weighted R² — the competition metric."""
    ss_res = np.sum(weights * (y_true - y_pred) ** 2)
    ss_tot = np.sum(weights * (y_true - np.average(y_true, weights=weights)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-8)


def load_cached_partitions(date_min: int = None,
                           date_max: int = None) -> pl.DataFrame:
    """Load and concatenate cached feature partitions."""
    parts = []
    for path in sorted(CACHE_DIR.glob("partition_*.parquet")):
        df = pl.read_parquet(path)
        if date_min is not None:
            df = df.filter(pl.col("date_id") >= date_min)
        if date_max is not None:
            df = df.filter(pl.col("date_id") <= date_max)
        if len(df) > 0:
            parts.append(df)
    return pl.concat(parts)


def get_feature_cols(df: pl.DataFrame) -> list:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def walk_forward_splits(dates: list, n_splits: int = 5):
    """Expanding window walk-forward splits on date_id."""
    fold_size = len(dates) // (n_splits + 1)
    splits = []
    for i in range(1, n_splits + 1):
        train_dates = set(dates[:i * fold_size])
        val_dates   = set(dates[i * fold_size:(i + 1) * fold_size])
        splits.append((train_dates, val_dates))
    return splits


def train_lgbm():
    print("Loading cached features...")
    df = load_cached_partitions()

    # Drop first time_id per (date, symbol) — lag nulls
    df = df.filter(pl.col("time_id") > 0)

    # Fill any remaining nulls in lag/rolling cols with 0
    df = df.fill_null(0.0)

    print(f"Dataset: {df.shape}")
    print(f"Date range: {df['date_id'].min()} to {df['date_id'].max()}")
    print(f"Unique dates: {df['date_id'].n_unique()}")

    feature_cols = get_feature_cols(df)
    print(f"Features: {len(feature_cols)}")

    # Convert to pandas for LightGBM
    # Process in chunks to avoid OOM
    print("Converting to pandas...")
    pdf = df.select(feature_cols + ["date_id", PRIMARY_TARGET, "weight"]).to_pandas()

    X = pdf[feature_cols]
    y = pdf[PRIMARY_TARGET]
    w = pdf["weight"]
    dates = pdf["date_id"]

    all_dates = sorted(pdf["date_id"].unique())
    splits = walk_forward_splits(all_dates, n_splits=5)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 128,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "n_jobs": -1,
    }

    cv_scores = []
    feature_importance = pd.Series(0.0, index=feature_cols)

    for fold, (train_dates, val_dates) in enumerate(splits):
        train_mask = dates.isin(train_dates)
        val_mask   = dates.isin(val_dates)

        X_tr, y_tr, w_tr = X[train_mask], y[train_mask], w[train_mask]
        X_val, y_val, w_val = X[val_mask], y[val_mask], w[val_mask]

        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        dval   = lgb.Dataset(X_val, label=y_val, weight=w_val)

        print(f"\nFold {fold+1}: train={len(X_tr):,} val={len(X_val):,}")

        model = lgb.train(
            params, dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(100),
            ]
        )

        preds = model.predict(X_val)
        preds = np.clip(preds, -5, 5)  # responders are clipped at ±5

        score = weighted_r2(y_val.values, preds, w_val.values)
        cv_scores.append(score)

        feature_importance += pd.Series(
            model.feature_importance("gain"),
            index=feature_cols
        )

        print(f"Fold {fold+1} weighted R²: {score:.6f}  "
              f"best_iter: {model.best_iteration}")

    print(f"\n{'='*50}")
    print(f"CV Weighted R²: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
    print(f"{'='*50}")

    fi = feature_importance.sort_values(ascending=False)
    print(f"\nTop 20 features by gain:")
    print(fi.head(20).round(1))

    return cv_scores, feature_importance


if __name__ == "__main__":
    train_lgbm()