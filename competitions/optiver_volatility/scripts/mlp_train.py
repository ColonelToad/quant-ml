import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold
from pathlib import Path
import sys
# Dynamically append your scripts directory so we can import train.py
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPTS_DIR))
from train import load_data

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "optiver-volatility-pred"

class VolatilityMLP(nn.Module):
    def __init__(self, num_cont_features, num_stocks, embed_dim=16):
        super(VolatilityMLP, self).__init__()
        
        # The Embedding Layer for stock_id
        self.stock_embedding = nn.Embedding(num_stocks, embed_dim)
        
        # The Main Network (Input is continuous features + embedding size)
        self.net = nn.Sequential(
            nn.Linear(num_cont_features + embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(), # SiLU (Swish) often outperforms ELU in tabular deep learning
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, cont_x, stock_idx):
        # Pass the stock integer through the embedding layer
        embed = self.stock_embedding(stock_idx)
        
        # Concatenate the continuous features with the new stock embedding
        x = torch.cat([cont_x, embed], dim=1)
        return self.net(x)

# Custom RMSPE Loss for PyTorch
class RMSPELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        y_true_raw = torch.exp(y_true)
        y_pred_raw = torch.exp(y_pred)
        return torch.sqrt(torch.mean(((y_true_raw - y_pred_raw) / y_true_raw) ** 2))

def apply_stock_normalization(X_tr, X_val, stock_tr, stock_val, features):
    tr_grouped = X_tr.assign(stock_id=stock_tr).groupby('stock_id')
    means = tr_grouped[features].mean()
    stds = tr_grouped[features].std().replace(0, 1e-8).fillna(1e-8) 
    
    X_tr_scaled, X_val_scaled = X_tr.copy(), X_val.copy()
    for col in features:
        X_tr_scaled[col] = (X_tr[col] - stock_tr.map(means[col])) / stock_tr.map(stds[col])
        X_val_scaled[col] = (X_val[col] - stock_val.map(means[col])) / stock_val.map(stds[col])
    return X_tr_scaled, X_val_scaled

def train_mlp():
    #df = pd.read_parquet(DATA_DIR / "features_with_lags.parquet") 
    df = load_data()
    
    FEATURE_COLS = [c for c in df.columns if c not in ['stock_id', 'time_id', 'target', 'chrono_rank']]
    X = df[FEATURE_COLS]
    y = np.log(df["target"])
    groups = df["time_id"]
    
    # 1. Map stock_id to contiguous integers (0 to N-1) for the embedding layer
    unique_stocks = df["stock_id"].unique()
    stock2idx = {stock: idx for idx, stock in enumerate(unique_stocks)}
    stock_indices = df["stock_id"].map(stock2idx)
    num_stocks = len(unique_stocks)

    gkf = GroupKFold(n_splits=5)
    cv_scores = []
    oof_preds_mlp = np.zeros(len(df))
    print("Starting MLP training with Stock Embeddings...")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, y_tr = X.loc[train_idx], y.loc[train_idx]
        X_val, y_val = X.loc[val_idx], y.loc[val_idx]

        # Extract both raw stock_ids (for normalization) and index stock_ids (for embeddings)
        stock_idx_tr, stock_idx_val = stock_indices.loc[train_idx], stock_indices.loc[val_idx]
        stock_raw_tr, stock_raw_val = df["stock_id"].loc[train_idx], df["stock_id"].loc[val_idx]

        if X_tr.isnull().values.any() or X_val.isnull().values.any():
            train_means = X_tr.mean()
            X_tr = X_tr.fillna(train_means)
            X_val = X_val.fillna(train_means) 
        
        # Normalize using RAW stock ids for the groupby logic
        X_tr, X_val = apply_stock_normalization(
            X_tr, X_val, stock_raw_tr, stock_raw_val, FEATURE_COLS
        )
        
        # 2. TensorDataset now takes 3 inputs: continuous, stock indices (LongTensor), and targets
        train_ds = TensorDataset(
            torch.FloatTensor(X_tr.values.copy()), 
            torch.LongTensor(stock_idx_tr.values.copy()), 
            torch.FloatTensor(y_tr.values.copy())
        )
        train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
        
        model = VolatilityMLP(num_cont_features=X_tr.shape[1], num_stocks=num_stocks)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        criterion = RMSPELoss()
        
        model.train()
        for epoch in range(40):
            epoch_loss = 0
            # 3. Unpack all 3 elements from the batch
            for batch_cont_X, batch_stock_idx, batch_y in train_loader:
                optimizer.zero_grad()
                preds = model(batch_cont_X, batch_stock_idx).squeeze()
                
                loss = criterion(preds, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
                optimizer.step()
                epoch_loss += loss.item()
                
            scheduler.step(epoch_loss)
        
        model.eval()
        with torch.no_grad():
            val_cont_X = torch.FloatTensor(X_val.values.copy())
            val_stock_idx = torch.LongTensor(stock_idx_val.values.copy())
            
            # Pass both to the model during inference
            preds_raw = model(val_cont_X, val_stock_idx).squeeze().detach().numpy()
            preds_raw = np.clip(preds_raw, -10, 10) 
            preds = np.exp(preds_raw)
            raw_val_y = np.exp(y_val.values)
            oof_preds_mlp[val_idx] = preds
            score = np.sqrt(np.mean(((raw_val_y - preds) / raw_val_y) ** 2))
            cv_scores.append(score)
            print(f"Fold {fold+1}: RMSPE = {score:.4f}")
            # Save the PyTorch weights for this fold
            torch.save(model.state_dict(), f"mlp_model_fold_{fold+1}.pt")

    print(f"\nFinal CV RMSPE: {np.mean(cv_scores):.4f}")
    np.save("oof_mlp.npy", oof_preds_mlp)

if __name__ == "__main__":
    train_mlp()