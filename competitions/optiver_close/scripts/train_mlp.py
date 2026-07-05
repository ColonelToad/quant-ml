"""
competitions/optiver_close/scripts/train_mlp.py
-------------------------------------------
PyTorch MLP training with Time Embeddings, OneCycleLR, and Walk-Forward CV.
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

CACHE_DIR = Path(__file__).resolve().parents[1]
CACHE_PATH = CACHE_DIR / "features_cache.parquet"

# --- 1. Class Definitions ---

class OptiverDataset(Dataset):
    """
    PyTorch Dataset for Optiver Data.
    Assumes missing values are handled and continuous features are scaled.
    """
    def __init__(self, df, cont_cols, cat_cols, target_col=None):
        # Continuous features (floats)
        self.X_cont = torch.tensor(df[cont_cols].values, dtype=torch.float32)
        # Categorical features (integers: stock_id, time_idx)
        self.X_cat = torch.tensor(df[cat_cols].values, dtype=torch.long)
        
        self.y = None
        if target_col is not None:
            self.y = torch.tensor(df[target_col].values, dtype=torch.float32)

    def __len__(self):
        return len(self.X_cont)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_cont[idx], self.X_cat[idx], self.y[idx]
        return self.X_cont[idx], self.X_cat[idx]


class OptiverMLP(nn.Module):
    """
    MLP with Stock and Time Embeddings.
    """
    def __init__(self, num_cont_features, num_stocks=200, num_time_steps=55, embedding_dim=16, hidden_dims=[128, 64, 32], dropout=0.2):
        super().__init__()
        
        # 1. Embedding Layers (Stock and Time)
        self.stock_embedding = nn.Embedding(num_embeddings=num_stocks, embedding_dim=embedding_dim)
        self.time_embedding = nn.Embedding(num_embeddings=num_time_steps, embedding_dim=embedding_dim)
        
        # 2. Continuous Feature Normalization
        self.batch_norm_cont = nn.BatchNorm1d(num_cont_features)
        
        # 3. Dense Network (Input = continuous + 2 embeddings)
        input_dim = num_cont_features + (embedding_dim * 2)
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.SiLU())  # SiLU often outperforms ReLU in finance
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
            
        self.mlp = nn.Sequential(*layers)
        
        # 4. Final Output Head
        self.output = nn.Linear(prev_dim, 1)

    def forward(self, x_cont, x_cat):
        # x_cat[:, 0] is stock_id, x_cat[:, 1] is time_idx
        stock_embeds = self.stock_embedding(x_cat[:, 0])
        time_embeds = self.time_embedding(x_cat[:, 1])
        
        # Normalize raw continuous features
        x_cont_norm = self.batch_norm_cont(x_cont)
        
        # Concatenate continuous features with learned embeddings
        x = torch.cat([x_cont_norm, stock_embeds, time_embeds], dim=1)
        
        # Pass through the MLP
        x = self.mlp(x)
        out = self.output(x)
        
        return out.squeeze(-1)


# --- 2. Scaler and Training Loop ---

def apply_expanding_scaler(df: pd.DataFrame, cont_cols: list) -> pd.DataFrame:
    """
    Scales continuous features using strictly historical data.
    Day T is scaled using the expanding mean/std from Day 0 up to Day T-1.
    """
    print("Applying running historical scaler...")
    
    # Calculate daily statistics
    daily_mean = df.groupby("date_id")[cont_cols].mean()
    daily_std = df.groupby("date_id")[cont_cols].std()
    
    # Calculate expanding historical stats and shift by 1 to prevent leakage
    hist_mean = daily_mean.expanding().mean().shift(1)
    hist_std = daily_std.expanding().mean().shift(1)
    
    # Backfill Day 0 (since it has no history) to avoid dropping it
    hist_mean = hist_mean.bfill()
    hist_std = hist_std.bfill().replace(0, 1.0) 
    
    # Apply scaling
    df_scaled = df.copy()
    for col in cont_cols:
        day_means = df_scaled["date_id"].map(hist_mean[col]).fillna(0.0)
        day_stds = df_scaled["date_id"].map(hist_std[col]).fillna(1.0)
        
        df_scaled[col] = (df_scaled[col] - day_means) / (day_stds + 1e-8)
        
        # Clip extreme outliers
        df_scaled[col] = df_scaled[col].clip(-10, 10)
        
    final_means = hist_mean.iloc[-1].to_dict()
    final_stds = hist_std.iloc[-1].to_dict()
    
    scaler_path = CACHE_DIR / "frozen_scaler.json"
    with open(scaler_path, "w") as f:
        json.dump({"means": final_means, "stds": final_stds}, f)
    print(f"Frozen scaler state saved to {scaler_path}")
    
    return df_scaled


def train_mlp_cv():
    print("Loading feature cache...")
    df = pd.read_parquet(CACHE_PATH)
    
    print("Sanitizing infinite and NaN values...")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0.0)
    
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    
    # --- Feature Definition Updates ---
    # Map seconds_in_bucket (0 to 540) to time_idx (0 to 54)
    df["time_idx"] = (df["seconds_in_bucket"] // 10).astype(int)
    
    drop_cols = ["target", "time_id", "date_id", "stock_id", "seconds_in_bucket", "time_idx"]
    flag_cols = ["is_first_bucket", "is_auction_cross_missing", "imbalance_buy_sell_flag"]
    
    # Separate binary flags so they are not scaled
    cont_cols = [c for c in df.columns if c not in drop_cols and c not in flag_cols]
    final_cont_cols = cont_cols + flag_cols
    
    # We now have two categorical columns for embedding
    cat_cols = ["stock_id", "time_idx"]
    
    # Scale entire dataset (safely historical) before splitting
    df = apply_expanding_scaler(df, cont_cols)
    
    # --- Walk-Forward CV Setup ---
    # We evaluate two different market regimes to ensure stability
    cv_folds = [
        {"train_end": 390, "valid_end": 435},
        {"train_end": 435, "valid_end": 481}
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")
    
    cv_scores = []
    
    for fold, fold_info in enumerate(cv_folds):
        print(f"\n{'='*40}")
        print(f"Fold {fold+1}: Train < {fold_info['train_end']} | Valid {fold_info['train_end']} to {fold_info['valid_end']-1}")
        print(f"{'='*40}")
        
        # Subset Data
        train_df = df[df["date_id"] < fold_info["train_end"]].reset_index(drop=True)
        valid_df = df[(df["date_id"] >= fold_info["train_end"]) & (df["date_id"] < fold_info["valid_end"])].reset_index(drop=True)
        
        # Create DataLoaders
        train_dataset = OptiverDataset(train_df, final_cont_cols, cat_cols, target_col="target")
        valid_dataset = OptiverDataset(valid_df, final_cont_cols, cat_cols, target_col="target")
        
        batch_size = 4096
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize Model, Optimizer, Scheduler
        model = OptiverMLP(num_cont_features=len(final_cont_cols), num_stocks=200, num_time_steps=55).to(device)
        criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        
        epochs = 10
        # OneCycleLR Scheduler pushes LR up to max_lr, then drops it
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=3e-3, 
            steps_per_epoch=len(train_loader), 
            epochs=epochs,
            pct_start=0.1
        )
        
        best_valid_loss = float("inf")
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for X_cont, X_cat, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
                X_cont, X_cat, y = X_cont.to(device), X_cat.to(device), y.to(device)
                
                optimizer.zero_grad()
                predictions = model(X_cont, X_cat)
                loss = criterion(predictions, y)
                loss.backward()
                
                # Clip gradients before stepping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()  # Step the scheduler AFTER every batch
                
                train_loss += loss.item()
                
            # Validation Phase
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for X_cont, X_cat, y in valid_loader:
                    X_cont, X_cat, y = X_cont.to(device), X_cat.to(device), y.to(device)
                    predictions = model(X_cont, X_cat)
                    loss = criterion(predictions, y)
                    valid_loss += loss.item()
                    
            train_loss /= len(train_loader)
            valid_loss /= len(valid_loader)
            
            print(f"Epoch {epoch+1} | Train MAE: {train_loss:.4f} | Valid MAE: {valid_loss:.4f}")
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), CACHE_DIR / f"optiver_mlp_fold{fold+1}.pth")
                
        print(f">>> Fold {fold+1} Best Valid MAE: {best_valid_loss:.4f}")
        cv_scores.append(best_valid_loss)
        
    print(f"\n{'='*40}")
    print(f"FINAL CV MAE: {np.mean(cv_scores):.4f} (Folds: {[round(s, 4) for s in cv_scores]})")
    print(f"{'='*40}")

if __name__ == "__main__":
    train_mlp_cv()