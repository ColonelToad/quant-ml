"""
competitions/optiver_close/scripts/blend_experiment.py
------------------------------------------------------
Trains LightGBM on Fold 2, loads the saved MLP for Fold 2, 
and evaluates the ensemble blend.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightgbm as lgb
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error

# Import your MLP classes and scaler from your existing script
from train_mlp import OptiverDataset, OptiverMLP, apply_expanding_scaler

CACHE_DIR = Path(__file__).resolve().parents[1]
CACHE_PATH = CACHE_DIR / "features_cache.parquet"
MLP_MODEL_PATH = CACHE_DIR / "optiver_mlp_fold2.pth"

def run_blend_experiment():
    print("Loading feature cache...")
    df = pd.read_parquet(CACHE_PATH)
    
    print("Sanitizing infinite and NaN values...")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    
    # Feature Setup
    df["time_idx"] = (df["seconds_in_bucket"] // 10).astype(int)
    drop_cols = ["target", "time_id", "date_id", "stock_id", "seconds_in_bucket", "time_idx"]
    flag_cols = ["is_first_bucket", "is_auction_cross_missing", "imbalance_buy_sell_flag"]
    
    cont_cols = [c for c in df.columns if c not in drop_cols and c not in flag_cols]
    final_cont_cols = cont_cols + flag_cols
    cat_cols = ["stock_id", "time_idx"]
    
    # We apply the scaler to the whole dataframe so the MLP can use it.
    # Note: LightGBM is scale-invariant, so using scaled features won't hurt its performance!
    df = apply_expanding_scaler(df, cont_cols)
    
    # --- Isolate Fold 2 ---
    train_end, valid_end = 435, 481
    train_df = df[df["date_id"] < train_end].reset_index(drop=True)
    valid_df = df[(df["date_id"] >= train_end) & (df["date_id"] < valid_end)].reset_index(drop=True)
    
    y_valid = valid_df["target"].values

    # ==========================================
    # 1. Train LightGBM on Fold 2
    # ==========================================
    print("\n--- Training LightGBM (Fold 2) ---")
    lgb_features = final_cont_cols + ["stock_id"] # LGBM uses stock_id natively
    
    train_data = lgb.Dataset(train_df[lgb_features], label=train_df["target"], categorical_feature=["stock_id"])
    valid_data = lgb.Dataset(valid_df[lgb_features], label=valid_df["target"], categorical_feature=["stock_id"], reference=train_data)
    
    params = {
        "objective": "mae", "metric": "mae", "boosting_type": "gbdt",
        "learning_rate": 0.05, "num_leaves": 64, "max_depth": 8,
        "feature_fraction": 0.7, "random_state": 42, "verbose": -1
    }
    
    lgb_model = lgb.train(
        params, train_data, num_boost_round=2000, valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
    )
    
    lgb_preds = lgb_model.predict(valid_df[lgb_features])
    lgb_mae = mean_absolute_error(y_valid, lgb_preds)
    print(f"LightGBM MAE: {lgb_mae:.4f}")

    # ==========================================
    # 2. Load MLP on Fold 2
    # ==========================================
    print("\n--- Running MLP Inference (Fold 2) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mlp_model = OptiverMLP(num_cont_features=len(final_cont_cols), num_stocks=200, num_time_steps=55).to(device)
    mlp_model.load_state_dict(torch.load(MLP_MODEL_PATH, map_location=device, weights_only=True))
    mlp_model.eval()
    
    valid_dataset = OptiverDataset(valid_df, final_cont_cols, cat_cols)
    valid_loader = DataLoader(valid_dataset, batch_size=4096, shuffle=False)
    
    mlp_preds = []
    with torch.no_grad():
        for X_cont, X_cat in valid_loader:
            X_cont, X_cat = X_cont.to(device), X_cat.to(device)
            preds = mlp_model(X_cont, X_cat)
            mlp_preds.extend(preds.cpu().numpy())
            
    mlp_preds = np.array(mlp_preds)
    mlp_mae = mean_absolute_error(y_valid, mlp_preds)
    print(f"MLP MAE:      {mlp_mae:.4f}")

    # ==========================================
    # 3. The Ensemble Blend
    # ==========================================
    print("\n--- Evaluating Blends ---")
    # 50/50 Blend
    blend_5050 = (lgb_preds * 0.5) + (mlp_preds * 0.5)
    mae_5050 = mean_absolute_error(y_valid, blend_5050)
    print(f"50/50 Blend MAE: {mae_5050:.4f}")
    
    # 70% LGBM / 30% MLP Blend (Trees are usually slightly more robust)
    blend_7030 = (lgb_preds * 0.7) + (mlp_preds * 0.3)
    mae_7030 = mean_absolute_error(y_valid, blend_7030)
    print(f"70/30 Blend MAE: {mae_7030:.4f}")

if __name__ == "__main__":
    run_blend_experiment()