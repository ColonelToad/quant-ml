"""
competitions/optiver_volatility/scripts/train.py
-------------------------------------------------
LightGBM training for Optiver Realized Volatility Prediction.
Metric: RMSPE (Root Mean Squared Percentage Error)
CV: Purged walk-forward splits on time_id
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.eval import rmspe

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "optiver-volatility-pred"
CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "optiver-volatility-pred"

FEATURE_COLS = [
    "rv_wap1", "rv_wap2", "rv_combined", "rv_first_half", "rv_second_half",
    "spread1_mean", "spread1_std", "spread2_mean",
    "imbalance1_mean", "imbalance1_std", "imbalance2_mean",
    "bid_depth_ratio_mean", "ask_depth_ratio_mean",
    "total_depth_mean", "total_depth_std",
    "wap1_drift", "book_update_count", "rv_accel",
    "trade_size_sum", "trade_size_mean", "trade_size_std",
    "trade_count", "trade_price_std", "log_trade_volume",
]

# Custom RMSPE objective for LightGBM
def rmspe_objective(y_pred, dataset):
    y_true = dataset.get_label()
    grad = -2 * (y_true - y_pred) / (y_true ** 2)
    hess = 2 / (y_true ** 2)
    return grad, hess

def rmspe_metric(y_pred, dataset):
    y_true = dataset.get_label()
    score = rmspe(y_true, y_pred)
    return "rmspe", score, False  # False = lower is better


def load_data():
    cache_path = CACHE_DIR / "features_cache_v2.parquet"
    feats = pd.read_parquet(cache_path)

    fill_zero = [
        "trade_size_sum", "trade_size_mean", "trade_size_std",
        "trade_count", "trade_price_std", "trade_price_mean", "log_trade_volume"
    ]
    feats[fill_zero] = feats[fill_zero].fillna(0)

    train = pd.read_csv(DATA_DIR / "train.csv")
    df = feats.merge(train, on=["stock_id", "time_id"], how="inner")
    return df


def walk_forward_cv(df, n_splits=5):
    """
    Walk-forward CV on time_id. Earlier time_ids train, later ones validate.
    time_id is not a true timestamp but is ordered — higher = later.
    """
    time_ids = sorted(df["time_id"].unique())
    fold_size = len(time_ids) // (n_splits + 1)

    splits = []
    for i in range(1, n_splits + 1):
        train_ids = set(time_ids[: i * fold_size])
        val_ids   = set(time_ids[i * fold_size : (i + 1) * fold_size])
        train_idx = df[df["time_id"].isin(train_ids)].index
        val_idx   = df[df["time_id"].isin(val_ids)].index
        splits.append((train_idx, val_idx))
    return splits


def train(use_log_target=False):
    print("Loading data...")
    df = load_data()
    print(f"  Dataset: {df.shape}")

    X = df[FEATURE_COLS]
    y = df["target"]
    if use_log_target:
        y_train_raw = np.log(y)
        print("  Using log(target)")
    else:
        y_train_raw = y
        print("  Using raw target")

    splits = walk_forward_cv(df, n_splits=5)
    print(f"  CV splits: {len(splits)}")

    lgb_params = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 128,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
    }

    cv_scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        X_tr, y_tr = X.loc[train_idx], y_train_raw.loc[train_idx]
        X_val, y_val = X.loc[val_idx], y.loc[val_idx]  # always eval on raw target

        if use_log_target:
            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, np.log(y_val))],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
            )
            val_preds = np.exp(model.predict(X_val))
        else:
            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval   = lgb.Dataset(X_val, label=y_val)
            model = lgb.train(
                {**lgb_params, "objective": "regression"},
                dtrain,
                valid_sets=[dval],
                feval=rmspe_metric,
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
            )
            val_preds = model.predict(X_val)

        score = rmspe(y_val.values, val_preds)
        cv_scores.append(score)
        models.append(model)
        print(f"  Fold {fold+1}: RMSPE = {score:.4f}  (best iter: {model.best_iteration_  if hasattr(model, 'best_iteration_') else model.best_iteration})")

    print(f"\nCV RMSPE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # Feature importance from last fold
    if use_log_target:
        fi = pd.Series(models[-1].feature_importances_, index=FEATURE_COLS)
    else:
        fi = pd.Series(models[-1].feature_importance("gain"), index=FEATURE_COLS)
    fi = fi.sort_values(ascending=False)
    print(f"\nTop feature importances (gain):\n{fi.head(15).round(1)}")

    return models, cv_scores


if __name__ == "__main__":
    print("=== Raw target ===")
    models_raw, scores_raw = train(use_log_target=False)
    print("\n=== Log target ===")
    models_log, scores_log = train(use_log_target=True)