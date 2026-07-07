"""
competitions/jane_street/scripts/model_multimae.py
--------------------------------------------------
Multitask AutoEncoder + MLP for Jane Street Real-Time Market Data.

Architecture:
- Encoder: Compresses static + time-varying features into a bottleneck.
- Decoder: Reconstructs original features (Unsupervised regularizer).
- Predictor: Predicts responder_6 (primary) + aux targets from bottleneck.
- Sequence Handling: Treats each time step independently (row-wise), but 
  accepts standard (batch, seq_len, features) tensors for easy swap with GRU.
"""

import torch
import torch.nn as nn
import numpy as np
import polars as pl
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

CACHE_DIR = Path(__file__).resolve().parents[1] / "feature_cache"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True)

PRIMARY_TARGET   = "responder_6"
AUX_TARGETS      = ["responder_3", "responder_7", "responder_8"]
ALL_TARGETS      = [PRIMARY_TARGET] + AUX_TARGETS
ALL_RESPONDERS   = [f"responder_{i}" for i in range(9)]
EXCLUDE_COLS     = set(["date_id", "time_id", "symbol_id", "weight"] + ALL_RESPONDERS)
SEQ_LEN          = 849

STATIC_FEATURES = [
    "feature_09", "feature_10", "feature_11", "feature_20",
    "feature_22", "feature_23", "feature_24", "feature_25",
    "feature_28", "feature_29", "feature_30", "feature_61",
]


def get_feature_splits(all_feature_cols: list):
    static = [f for f in all_feature_cols if f in STATIC_FEATURES]
    time_varying = [f for f in all_feature_cols if f not in STATIC_FEATURES]
    return static, time_varying


class JaneStreetSequenceDataset(Dataset):
    """
    Groups data by (date_id, symbol_id). 
    Pads short sequences with zeros (ignored by loss via weights=0).
    """
    def __init__(self, df: pl.DataFrame, feature_cols: list):
        self.static_cols, self.tv_cols = get_feature_splits(feature_cols)

        groups = (
            df.sort(["date_id", "symbol_id", "time_id"])
            .group_by(["date_id", "symbol_id"], maintain_order=True)
        )

        self.sequences = []
        for _, group in groups:
            seq_len = len(group)
            static_x = group.select(self.static_cols).head(1).to_numpy()[0]
            
            if seq_len >= SEQ_LEN:
                seq_x    = group.select(self.tv_cols).head(SEQ_LEN).to_numpy()
                targets  = group.select(ALL_TARGETS).head(SEQ_LEN).to_numpy()
                weights  = group["weight"].head(SEQ_LEN).to_numpy()
            else:
                # Pad sequences shorter than SEQ_LEN
                seq_x   = np.zeros((SEQ_LEN, len(self.tv_cols)), dtype=np.float32)
                targets = np.zeros((SEQ_LEN, len(ALL_TARGETS)), dtype=np.float32)
                weights = np.zeros(SEQ_LEN, dtype=np.float32)
                
                seq_x[:seq_len, :]   = group.select(self.tv_cols).to_numpy()
                targets[:seq_len, :] = group.select(ALL_TARGETS).to_numpy()
                weights[:seq_len]    = group["weight"].to_numpy()

            self.sequences.append((
                static_x.astype(np.float32),
                seq_x.astype(np.float32),
                targets.astype(np.float32),
                weights.astype(np.float32),
            ))

        print(f"  Built {len(self.sequences)} padded sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        static_x, seq_x, targets, weights = self.sequences[idx]
        return (
            torch.tensor(static_x),
            torch.tensor(seq_x),
            torch.tensor(targets),
            torch.tensor(weights),
        )


class JaneStreetAutoEncoder(nn.Module):
    def __init__(self, static_dim: int, tv_dim: int, 
                 bottleneck_dim: int = 64, num_targets: int = 4, 
                 dropout: float = 0.2):
        super().__init__()
        input_dim = static_dim + tv_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.SiLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, input_dim)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_targets)
        )

    def forward(self, static_x: torch.Tensor, seq_x: torch.Tensor):
        batch_size, seq_len, _ = seq_x.shape
        
        # Expand static features across the sequence
        static_expanded = static_x.unsqueeze(1).expand(-1, seq_len, -1)
        x = torch.cat([static_expanded, seq_x], dim=-1)
        
        bottleneck = self.encoder(x)
        reconstruction = self.decoder(bottleneck)
        predictions = self.predictor(bottleneck)
        
        return predictions, reconstruction, x

    def freeze_encoder(self):
        """Freeze encoder/decoder for online fine-tuning of the predictor."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False


def weighted_r2_torch(y_true: torch.Tensor, y_pred: torch.Tensor,
                      weights: torch.Tensor) -> float:
    """Weighted R² on primary target."""
    y_t = y_true[:, :, 0].flatten()
    y_p = y_pred[:, :, 0].flatten()
    w   = weights.flatten()

    ss_res = (w * (y_t - y_p) ** 2).sum()
    y_mean = (w * y_t).sum() / w.sum()
    ss_tot = (w * (y_t - y_mean) ** 2).sum()
    return (1 - ss_res / (ss_tot + 1e-8)).item()


def autoencoder_loss(preds: torch.Tensor, targets: torch.Tensor, 
                     recon: torch.Tensor, inputs: torch.Tensor, 
                     weights: torch.Tensor, 
                     aux_weight: float = 0.3, 
                     target_weight: float = 0.9) -> torch.Tensor:
    
    # 1. Target Loss (ignores padded steps because weight=0)
    w = weights.unsqueeze(-1)
    primary_loss = (w * (preds[:, :, 0:1] - targets[:, :, 0:1]) ** 2).mean()
    aux_loss = (w * (preds[:, :, 1:] - targets[:, :, 1:]) ** 2).mean()
    target_loss = (1 - aux_weight) * primary_loss + aux_weight * aux_loss
    
    # 2. Reconstruction Loss (Applies to all inputs to map the feature space)
    recon_loss = nn.functional.mse_loss(recon, inputs)
    
    return target_weight * target_loss + (1 - target_weight) * recon_loss


def train_epoch(model: JaneStreetAutoEncoder, loader: DataLoader,
                optimizer: torch.optim.Optimizer, device: torch.device) -> tuple:
    model.train()
    total_loss, total_r2, n_batches = 0.0, 0.0, 0
    
    for static_x, seq_x, targets, weights in loader:
        static_x, seq_x = static_x.to(device), seq_x.to(device)
        targets, weights = targets.to(device), weights.to(device)

        optimizer.zero_grad()
        preds, recon, inputs = model(static_x, seq_x)
        loss = autoencoder_loss(preds, targets, recon, inputs, weights)
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_r2   += weighted_r2_torch(targets, preds, weights)
        n_batches  += 1

    return total_loss / n_batches, total_r2 / n_batches


@torch.no_grad()
def eval_epoch(model: JaneStreetAutoEncoder, loader: DataLoader,
               device: torch.device) -> tuple:
    model.eval()
    total_loss, total_r2, n_batches = 0.0, 0.0, 0

    for static_x, seq_x, targets, weights in loader:
        static_x, seq_x = static_x.to(device), seq_x.to(device)
        targets, weights = targets.to(device), weights.to(device)

        preds, recon, inputs = model(static_x, seq_x)
        loss = autoencoder_loss(preds, targets, recon, inputs, weights)
        
        total_loss += loss.item()
        total_r2   += weighted_r2_torch(targets, preds, weights)
        n_batches  += 1

    return total_loss / n_batches, total_r2 / n_batches


def load_partition(path: Path, feature_cols: list) -> pl.DataFrame:
    return (
        pl.read_parquet(path)
        .filter(pl.col("time_id") > 0)
        .fill_null(0.0)
        .select(feature_cols + ["date_id", "time_id", "symbol_id",
                                "weight"] + ALL_TARGETS)
    )


def train_multimae(n_epochs: int = 5, batch_size: int = 32, lr: float = 1e-3, val_partition_id: int = 8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    partition_paths = sorted(CACHE_DIR.glob("partition_*.parquet"))
    train_paths = [p for p in partition_paths if int(p.stem.split("_")[1]) < val_partition_id]
    val_paths   = [p for p in partition_paths if int(p.stem.split("_")[1]) >= val_partition_id]

    sample = pl.read_parquet(partition_paths[0]).head(100)
    feature_cols = [c for c in sample.columns if c not in EXCLUDE_COLS]
    static_cols, tv_cols = get_feature_splits(feature_cols)
    print(f"Static features: {len(static_cols)}, Time-varying: {len(tv_cols)}")

    print("Building validation dataset...")
    val_df = pl.concat([load_partition(p, feature_cols) for p in val_paths])
    val_ds = JaneStreetSequenceDataset(val_df, feature_cols)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    del val_df

    model = JaneStreetAutoEncoder(
        static_dim=len(static_cols),
        tv_dim=len(tv_cols),
        bottleneck_dim=64,
        dropout=0.2,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs * len(train_paths)
    )

    best_val_r2 = -np.inf
    best_model_path = MODEL_DIR / "multimae_best.pt"

    for epoch in range(n_epochs):
        print(f"\n--- Epoch {epoch+1}/{n_epochs} ---")
        epoch_train_loss, epoch_train_r2, n_parts = 0.0, 0.0, 0

        for path in train_paths:
            print(f"  Partition {path.stem}...")
            train_df = load_partition(path, feature_cols)
            train_ds = JaneStreetSequenceDataset(train_df, feature_cols)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
            del train_df

            loss, r2 = train_epoch(model, train_loader, optimizer, device)
            scheduler.step()
            epoch_train_loss += loss
            epoch_train_r2   += r2
            n_parts += 1
            print(f"    loss={loss:.4f}  train_r2={r2:.4f}")

        val_loss, val_r2 = eval_epoch(model, val_loader, device)
        print(f"  Epoch {epoch+1} summary: train_r2={epoch_train_r2/n_parts:.4f}  val_loss={val_loss:.4f}  val_r2={val_r2:.4f}")

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_r2": val_r2,
                "feature_cols": feature_cols,
                "static_cols": static_cols,
                "tv_cols": tv_cols,
            }, best_model_path)
            print(f"  Saved best model (val_r2={val_r2:.4f})")

    print(f"\nBest val R²: {best_val_r2:.4f}")
    return model


if __name__ == "__main__":
    train_multimae(n_epochs=5, batch_size=32, lr=1e-3)