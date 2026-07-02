"""
competitions/optiver_volatility/scripts/train.py
-------------------------------------------------
LightGBM training for Optiver Realized Volatility Prediction.
Metric: RMSPE (Root Mean Squared Percentage Error)
CV: GroupKFold on time_id (prevents temporal leakage)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import sys
from pathlib import Path
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
from sklearn.model_selection import GroupKFold

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
    "target_lag1", "target_lag2"  # Added our true chronological lags
]

# Custom RMSPE objective for LightGBM
def rmspe_objective(y_pred, dataset):
    y_true = dataset.get_label()
    # grad: 1st derivative of RMSPE squared error
    grad = -2 * (y_true - y_pred) / (y_true ** 2)
    # hess: 2nd derivative
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
    
    print("  Reconstructing chronological timeline...")
    pivot_df = df.pivot(index='time_id', columns='stock_id', values='target')
    pivot_df = pivot_df.apply(lambda col: col.fillna(col.mean()), axis=0)

    distances = pdist(pivot_df.values, metric='correlation')
    Z = linkage(distances, method='ward')
    Z_opt = optimal_leaf_ordering(Z, distances)
    ordered_indices = leaves_list(Z_opt)

    ordered_time_ids = pivot_df.index[ordered_indices]
    time_map = {tid: rank for rank, tid in enumerate(ordered_time_ids)}

    df['chrono_rank'] = df['time_id'].map(time_map)
    df = df.sort_values(['stock_id', 'chrono_rank']).reset_index(drop=True)

    print("  Generating true chronological lags...")
    df['target_lag1'] = df.groupby('stock_id')['target'].shift(1)
    df['target_lag2'] = df.groupby('stock_id')['target'].shift(2)

    df = df.dropna(subset=['target_lag1', 'target_lag2']).reset_index(drop=True)

    return df


def train(use_log_target=False):
    print("Loading data...")
    df = load_data()
    print(f"  Dataset: {df.shape}")

    X = df[FEATURE_COLS]
    y = df["target"]
    groups = df["time_id"]
    
    if use_log_target:
        y_train_raw = np.log(y)
        print("  Using log(target)")
    else:
        y_train_raw = y
        print("  Using raw target with Custom RMSPE Objective")

    # Swapped to GroupKFold to prevent leakage across time_ids
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X, y_train_raw, groups=groups))
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
    oof_preds_lgb = np.zeros(len(df))

    for fold, (train_idx, val_idx) in enumerate(splits):
        X_tr, y_tr = X.loc[train_idx], y_train_raw.loc[train_idx]
        X_val, y_val = X.loc[val_idx], y.loc[val_idx]  # always eval on raw target

        if use_log_target:
            # We use standard regression for log target
            model = lgb.LGBMRegressor(**lgb_params, objective="regression")
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, np.log(y_val))],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
            val_preds = np.exp(model.predict(X_val))
        else:
            # We use the Dataset API to pass the custom objective natively
            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            
            params_copy = lgb_params.copy()
            iters = params_copy.pop("n_estimators")
            
            # --- FIX: Assign the function directly to the objective key ---
            params_copy["objective"] = rmspe_objective
            
            model = lgb.train(
                params_copy, 
                dtrain,
                num_boost_round=iters,
                valid_sets=[dval],
                feval=rmspe_metric,   # feval is still a valid argument for lgb.train()
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
            val_preds = model.predict(X_val)
            oof_preds_lgb[val_idx] = val_preds

        score = rmspe(y_val.values, val_preds)
        cv_scores.append(score)
        models.append(model)
        # Save the LightGBM model for this fold
        model.save_model(f"lgb_model_fold_{fold+1}.txt")
        
        best_iter = model.best_iteration_ if hasattr(model, 'best_iteration_') else model.best_iteration
        print(f"  Fold {fold+1}: RMSPE = {score:.4f}  (best iter: {best_iter})")

    print(f"\nCV RMSPE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # Feature importance from last fold
    if use_log_target:
        fi = pd.Series(models[-1].feature_importances_, index=FEATURE_COLS)
    else:
        fi = pd.Series(models[-1].feature_importance("gain"), index=FEATURE_COLS)
    fi = fi.sort_values(ascending=False)
    print(f"\nTop 15 feature importances (gain):\n{fi.head(15).round(1)}")
    np.save("oof_lgb.npy", oof_preds_lgb)

    return models, cv_scores


if __name__ == "__main__":
    print("=== Raw target ===")
    models_raw, scores_raw = train(use_log_target=False)
    #print("\n=== Log target ===")
    #models_log, scores_log = train(use_log_target=True)