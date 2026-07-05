"""
competitions/optiver_close/scripts/train.py
-------------------------------------------
Model training for Optiver Trading at the Close (2023).
Metric: MAE of (predicted - actual) price movement vs synthetic index.
"""

import polars as pl
import pandas as pd
import lightgbm as lgb
import numpy as np
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[1]
CACHE_PATH = CACHE_DIR / "features_cache.parquet"

def train_lgbm_baseline():
    print("Loading feature cache...")
    df = pl.read_parquet(CACHE_PATH).to_pandas()

    # Define split boundary
    split_day = 400
    
    # Define features and target
    drop_cols = ["target", "time_id", "date_id"]
    features = [c for c in df.columns if c not in drop_cols]
    
    print(f"Training on {len(features)} features...")

    # Split data chronologically
    train = df[df["date_id"] < split_day]
    valid = df[df["date_id"] >= split_day]

    X_train, y_train = train[features], train["target"]
    X_valid, y_valid = valid[features], valid["target"]

    # LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=["stock_id"])
    valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=["stock_id"], reference=train_data)

    # Baseline hyperparameters
    params = {
        "objective": "mae",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": 8,
        "feature_fraction": 0.7,
        "random_state": 42,
        "verbose": -1
    }

    print("Training LightGBM model...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, valid_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )

    model.save_model(CACHE_DIR / "lgbm_model.txt")
    print(f"LightGBM model saved to {CACHE_DIR / 'lgbm_model.txt'}")

    # Feature Importance
    importance_df = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importance(importance_type="gain")
    }).sort_values(by="importance", ascending=False)
    
    print("\nTop 10 Features:")
    print(importance_df.head(10).to_string(index=False))

if __name__ == "__main__":
    train_lgbm_baseline()
