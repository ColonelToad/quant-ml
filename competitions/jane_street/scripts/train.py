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

import gc
import polars as pl
import numpy as np
import lightgbm as lgb
import pandas as pd
from pathlib import Path
import polars.selectors as cs

CACHE_DIR = Path(__file__).resolve().parents[1] / "feature_cache"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)
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


# 1. Update your loading function to return a LazyFrame
def load_cached_partitions(date_min: int = None, date_max: int = None) -> pl.LazyFrame:
    """Scan cached feature partitions lazily."""
    # scan_parquet can take a glob pattern directly, replacing the entire loop!
    lf = pl.scan_parquet(CACHE_DIR / "partition_*.parquet")
    
    if date_min is not None:
        lf = lf.filter(pl.col("date_id") >= date_min)
    if date_max is not None:
        lf = lf.filter(pl.col("date_id") <= date_max)
        
    return lf


def get_feature_cols(df: pl.DataFrame) -> list:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def walk_forward_splits(dates: list, n_splits: int = 5, max_train_folds: int = 2):
    """
    Rolling window walk-forward splits on date_id.
    max_train_folds limits the lookback window to prevent OOM.
    """
    fold_size = len(dates) // (n_splits + 1)
    splits = []
    
    for i in range(1, n_splits + 1):
        # Instead of starting at 0, we start at a calculated index
        # to only keep the most recent 'max_train_folds' worth of data
        start_idx = max(0, (i - max_train_folds) * fold_size)
        
        train_dates = set(dates[start_idx : i * fold_size])
        val_dates   = set(dates[i * fold_size : (i + 1) * fold_size])
        
        splits.append((train_dates, val_dates))
        
    return splits

def train_lgbm():
    print("Scanning cached features lazily...")
    lf = load_cached_partitions()

    # Apply lazy transformations
    lf = lf.filter(pl.col("time_id") > 0)
    lf = lf.fill_null(0.0)
    lf = lf.with_columns(cs.float().cast(pl.Float32))

    all_columns = lf.collect_schema().names()
    feature_cols = [c for c in all_columns if c not in EXCLUDE_COLS]
    print(f"Features: {len(feature_cols)}")

    cols_to_keep = feature_cols + ["date_id", PRIMARY_TARGET, "weight"]
    lf = lf.select(cols_to_keep)

    # 1. Extract unique dates via a tiny, lightweight query
    print("Extracting unique dates from disk...")
    all_dates = sorted(lf.select("date_id").unique().collect().to_series().to_list())
    splits = walk_forward_splits(all_dates, n_splits=5, max_train_folds=2)
    
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
    best_val_r2 = -np.inf
    best_model_path = MODEL_DIR / "lgbm_baseline_best.txt"

    for fold, (train_dates, val_dates) in enumerate(splits):
        print(f"\n{'='*30}")
        print(f"Preparing Fold {fold+1}...")
        
        # 2. Collect ONLY training data directly to NumPy (Skips Pandas!)
        print("  Loading training data...")
        train_polars = lf.filter(pl.col("date_id").is_in(list(train_dates))).collect()
        
        X_tr = train_polars.select(feature_cols).to_numpy()
        y_tr = train_polars.select(PRIMARY_TARGET).to_numpy().ravel()
        w_tr = train_polars.select("weight").to_numpy().ravel()
        
        # Destroy the Polars dataframe immediately
        del train_polars
        gc.collect()

        # 3. Collect ONLY validation data directly to NumPy
        print("  Loading validation data...")
        val_polars = lf.filter(pl.col("date_id").is_in(list(val_dates))).collect()
        
        X_val = val_polars.select(feature_cols).to_numpy()
        y_val = val_polars.select(PRIMARY_TARGET).to_numpy().ravel()
        w_val = val_polars.select("weight").to_numpy().ravel()
        
        del val_polars
        gc.collect()

        print(f"  Fold {fold+1} Shapes: train={len(X_tr):,} val={len(X_val):,}")

        # 4. Create LightGBM Datasets
        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, free_raw_data=True)
        dval   = lgb.Dataset(X_val, label=y_val, weight=w_val, free_raw_data=True)

        # Destroy ONLY the training arrays before training begins. 
        # Keep X_val alive for predictions!
        del X_tr, y_tr, w_tr
        gc.collect()

        print("  Training model...")
        model = lgb.train(
            params, dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(100),
            ]
        )

        print("  Evaluating...")
        # Predict directly on the X_val array we kept in memory!
        preds = model.predict(X_val)
        preds = np.clip(preds, -5, 5)

        score = weighted_r2(y_val, preds, w_val)
        cv_scores.append(score)

        feature_importance += pd.Series(
            model.feature_importance("gain"),
            index=feature_cols
        )

        print(f"  Fold {fold+1} weighted R²: {score:.6f} (best_iter: {model.best_iteration})")
        
        # ADD THIS BLOCK: Check if this is the best fold and save it
        if score > best_val_r2:
            best_val_r2 = score
            # LightGBM has a native, highly efficient save format (.txt)
            model.save_model(str(best_model_path))
            print(f"  *** Saved new best model (val_r2={score:.6f}) to {best_model_path}")
        
        # Final cleanup for the fold
        del dtrain, dval, model, X_val, preds, y_val, w_val
        gc.collect()

    print(f"\n{'='*50}")
    print(f"CV Weighted R²: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
    print(f"{'='*50}")

    fi = feature_importance.sort_values(ascending=False)
    print(f"\nTop 20 features by gain:")
    print(fi.head(20).round(1))

    return cv_scores, feature_importance


if __name__ == "__main__":
    train_lgbm()