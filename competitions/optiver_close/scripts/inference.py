"""
competitions/optiver_close/scripts/inference.py
-----------------------------------------------
The final Kaggle streaming API inference loop.
"""

import torch
import lightgbm as lgb
import pandas as pd
import polars as pl
import numpy as np
import json
from pathlib import Path

# Important: This references the Kaggle mock utility you downloaded
import public_timeseries_testing_util as optiver2023 

# Import your PyTorch architecture so we can load the weights
from train_mlp import OptiverMLP

CACHE_DIR = Path(__file__).resolve().parents[1]

# --- 1. The Pre-Allocated State Buffer ---

class OptiverStateBuffer:
    def __init__(self, num_stocks=200):
        self.num_stocks = num_stocks
        self.reset_daily_state()

    def reset_daily_state(self):
        self.spread_hist = np.full((self.num_stocks, 5), np.nan)
        self.imb_size_hist = np.full((self.num_stocks, 5), np.nan)
        self.last_spread = np.full(self.num_stocks, np.nan)
        self.last_wap = np.full(self.num_stocks, np.nan)
        self.last_matched_size = np.full(self.num_stocks, np.nan)
        self.last_imb_size = np.full(self.num_stocks, np.nan)
        self.last_imb_ratio = np.full(self.num_stocks, np.nan)
        
        self.cumsum_sq_ret = np.zeros(self.num_stocks)
        self.ptr = 0

    def update_and_extract_features(self, df_current: pl.DataFrame, seconds_in_bucket: int) -> pl.DataFrame:
        if seconds_in_bucket == 0:
            self.reset_daily_state()
            
        # Initialize safe, empty arrays to handle missing stocks
        curr_spread = np.full(self.num_stocks, np.nan)
        curr_imb_size = np.full(self.num_stocks, np.nan)
        curr_wap = np.full(self.num_stocks, np.nan)
        curr_matched = np.full(self.num_stocks, np.nan)
        
        # Calculate instantaneous features for current bucket
        df_current = df_current.with_columns(
            spread=(pl.col("ask_price") - pl.col("bid_price")),
            imbalance_ratio=(
                (pl.col("imbalance_size") * (2 * pl.col("imbalance_buy_sell_flag") - 1)) / 
                (pl.col("imbalance_size") + 1e-8)
            )
        )
        
        # Get integer IDs of the stocks present in this specific bucket
        present_stocks = df_current["stock_id"].to_numpy()
        
        # Route the data to the correct memory slots
        curr_spread[present_stocks] = df_current["spread"].to_numpy()
        curr_imb_size[present_stocks] = df_current["imbalance_size"].to_numpy()
        curr_wap[present_stocks] = df_current["wap"].to_numpy()
        curr_matched[present_stocks] = df_current["matched_size"].to_numpy()
        curr_imb_ratio = np.full(self.num_stocks, np.nan)
        curr_imb_ratio[present_stocks] = df_current["imbalance_ratio"].to_numpy()
        
        # Extract historical features BEFORE updating
        with np.errstate(invalid='ignore'): 
            roll5_spread = np.nanmean(self.spread_hist, axis=1)[present_stocks]
            roll5_imb_size = np.nanmean(self.imb_size_hist, axis=1)[present_stocks]
        lag1_spread = self.last_spread[present_stocks]
        lag1_wap = self.last_wap[present_stocks]
        lag1_matched = self.last_matched_size[present_stocks]
        lag1_imb_size = self.last_imb_size[present_stocks]
        lag1_imb_ratio = self.last_imb_ratio[present_stocks]
        rv_so_far = np.sqrt(self.cumsum_sq_ret[present_stocks])
        
        # Calculate WAP returns for Volatility
        wap_ret = np.where(np.isnan(self.last_wap), 0.0, np.log(curr_wap / self.last_wap))
        
        # UPDATE the state for the *next* iteration
        self.spread_hist[:, self.ptr] = curr_spread
        self.imb_size_hist[:, self.ptr] = curr_imb_size
        self.last_spread[:] = curr_spread
        self.last_wap[:] = curr_wap
        self.last_matched_size[:] = curr_matched
        self.last_imb_size[:] = curr_imb_size
        self.last_imb_ratio[:] = curr_imb_ratio
        self.cumsum_sq_ret += (wap_ret ** 2)
        
        self.ptr = (self.ptr + 1) % 5
        
        # Inject history back into the Polars DataFrame
        df_out = df_current.with_columns(
            spread_lag1=pl.Series(lag1_spread),
            spread_roll5_mean=pl.Series(roll5_spread),
            imbalance_roll5_mean=pl.Series(roll5_imb_size),
            wap_lag1=pl.Series(lag1_wap),
            matched_size_lag1=pl.Series(lag1_matched),
            imbalance_size_lag1=pl.Series(lag1_imb_size),
            rv_so_far=pl.Series(rv_so_far),
            
            # Momentum Features (Requires lags)
            imbalance_size_momentum=(pl.col("imbalance_size") - pl.Series(lag1_imb_size)),
            imbalance_ratio_momentum=(pl.col("imbalance_ratio") - pl.Series(lag1_imb_ratio)),
            
            # Drift
            wap_drift=(pl.col("wap") / pl.Series(lag1_wap) - 1)
        )
        
        return df_out

# --- 2. Online Feature Generator ---

def apply_online_features(df: pl.DataFrame) -> pl.DataFrame:
    """Calculates instantaneous cross-sectional features for a single bucket."""
    
    # We do NOT group by date_id/seconds_in_bucket here because 
    # the API guarantees this DataFrame is exactly one single bucket in time!
    df = df.with_columns(
        is_first_bucket=pl.col("seconds_in_bucket").eq(0).cast(pl.Int8),
        is_auction_cross_missing=pl.col("far_price").is_null().cast(pl.Int8),
        far_near_spread=(pl.col("far_price") - pl.col("near_price")).fill_null(0.0),
        
        # Cross-Sectional Features (Calculated instantly across all present stocks)
        global_wap_drift=pl.col("wap_drift").mean().fill_null(0.0),
        imbalance_size_rank=pl.col("imbalance_size").rank(),
        matched_size_rank=pl.col("matched_size").rank(),
        spread_rank=pl.col("spread").rank()
    )
    
    df = df.with_columns(
        relative_wap_drift=(pl.col("wap_drift") - pl.col("global_wap_drift")).fill_null(0.0)
    )
    
    # Time Index for the MLP embedding
    df = df.with_columns(time_idx=(pl.col("seconds_in_bucket") // 10).cast(pl.Int32))
    
    # Fill remaining Nulls just like the training pipeline
    df = df.fill_nan(0.0).fill_null(0.0) 
    return df

# --- 3. The Main API Loop ---

def main():
    print("Loading Models and Environment...")
    
    # 1. Load LightGBM
    lgb_model = lgb.Booster(model_file=str(CACHE_DIR / "lgbm_model.txt"))
    
    # 2. Load PyTorch MLP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- FIXED: Explicitly define the exact 30 features the model expects ---
    # 27 Continuous Columns
    cont_cols = [
        "imbalance_size", "spread", "imbalance_ratio", "rv_so_far", "wap_drift",
        "spread_lag1", "imbalance_size_lag1", "matched_size_lag1", 
        "spread_roll5_mean", "imbalance_roll5_mean", "reference_price", 
        "matched_size", "near_price", "far_price", "bid_price", "bid_size", 
        "ask_price", "ask_size", "wap", "global_wap_drift", "relative_wap_drift", 
        "imbalance_size_momentum", "imbalance_ratio_momentum", "far_near_spread",
        "imbalance_size_rank", "matched_size_rank", "spread_rank"
    ]
    # 3 Binary Flags
    flag_cols = ["is_first_bucket", "is_auction_cross_missing", "imbalance_buy_sell_flag"]
    # 2 Categorical Embedding Columns
    cat_cols = ["stock_id", "time_idx"]
    
    final_cont_cols = cont_cols + flag_cols
    NUM_CONT_FEATURES = len(final_cont_cols)  # This will precisely equal 30
    # -------------------------------------------------------------------------
    
    mlp_model = OptiverMLP(num_cont_features=NUM_CONT_FEATURES, num_stocks=200, num_time_steps=55).to(device)
    mlp_model.load_state_dict(torch.load(CACHE_DIR / "optiver_mlp_fold2.pth", map_location=device))
    mlp_model.eval()
    
    # 3. Load Frozen Scaler
    with open(CACHE_DIR / "frozen_scaler.json", "r") as f:
        scaler_data = json.load(f)
    frozen_means = scaler_data["means"]
    frozen_stds = scaler_data["stds"]

    # 4. Initialize State Buffer & API
    state_buffer = OptiverStateBuffer(num_stocks=200)
    env = optiver2023.make_env()
    iter_test = env.iter_test()

    print("Starting Streaming Inference...")
    for (test_df, revealed_targets, sample_prediction_df) in iter_test:
        
        # 1. Convert to Polars
        pl_test = pl.from_pandas(test_df)
        sec_in_bucket = pl_test["seconds_in_bucket"][0]
        
        # 2. Apply History and Online Features
        pl_features = state_buffer.update_and_extract_features(pl_test, sec_in_bucket)
        pl_features = apply_online_features(pl_features)
        
        # 3. Prepare features for models
        pd_features = pl_features.to_pandas()
        
        # 4. LightGBM Prediction
        lgb_features = final_cont_cols + ["stock_id", "seconds_in_bucket"]
        lgb_preds = lgb_model.predict(pd_features[lgb_features])
        
        # 5. Scale & MLP Prediction
        # Scale only the continuous columns
        for col in cont_cols:
            mean_val = frozen_means.get(col, 0.0)
            std_val = frozen_stds.get(col, 1.0)
            pd_features[col] = (pd_features[col] - mean_val) / (std_val + 1e-8)
            pd_features[col] = pd_features[col].clip(-10, 10)
            
        X_cont = torch.tensor(pd_features[final_cont_cols].values, dtype=torch.float32).to(device)
        X_cat = torch.tensor(pd_features[cat_cols].values, dtype=torch.long).to(device)
        
        with torch.no_grad():
            mlp_preds = mlp_model(X_cont, X_cat).cpu().numpy()
            
        # 6. Ensemble Blend (70% LGBM / 30% MLP as tested)
        final_preds = (lgb_preds * 0.7) + (mlp_preds * 0.3)
        
        # 7. Submit
        sample_prediction_df['target'] = final_preds
        env.predict(sample_prediction_df)
        
    print("Inference Complete!")

if __name__ == "__main__":
    main()