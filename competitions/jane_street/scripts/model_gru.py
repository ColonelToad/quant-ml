"""
competitions/jane_street/scripts/model_gru.py
---------------------------------------------
GRU model for Jane Street Real-Time Market Data Forecasting.

Architecture:
- Static feature encoder: linear projection → GRU initial hidden state
- Time-varying features: input at each time step
- GRU: 2 layers, hidden_dim=128
- Multi-task head: predicts responder_6 (primary) + responder_3/7/8 (aux)
- Online fine-tuning: freeze GRU weights, update only output head

Sequence structure:
- One sequence = one (date_id, symbol_id) pair
- Length: 849 steps (truncate 968-step sequences)
- Input dim: 63 time-varying + 9 lag features + cross-symbol features
- Static dim: 12 features → hidden state init
"""

import torch
import torch.nn as nn
import numpy as np
import polars as pl
import pandas as pd
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
    """Split feature columns into static and time-varying."""
    static = [f for f in all_feature_cols if f in STATIC_FEATURES]
    time_varying = [f for f in all_feature_cols if f not in STATIC_FEATURES]
    return static, time_varying


class JaneStreetSequenceDataset(Dataset):
    """
    Each item is one (date_id, symbol_id) sequence of length SEQ_LEN.
    Returns:
        static_x:  (static_dim,)          — static features for hidden init
        seq_x:     (SEQ_LEN, tv_dim)      — time-varying features
        targets:   (SEQ_LEN, 4)           — responder_6/3/7/8
        weights:   (SEQ_LEN,)             — sample weights
    """
    def __init__(self, df: pl.DataFrame, feature_cols: list):
        self.static_cols, self.tv_cols = get_feature_splits(feature_cols)

        # Group by (date_id, symbol_id) — each group is one sequence
        groups = (
            df.sort(["date_id", "symbol_id", "time_id"])
            .group_by(["date_id", "symbol_id"], maintain_order=True)
        )

        self.sequences = []
        for _, group in groups:
            seq_len = len(group)
            
            # Extract actual data
            static_x = group.select(self.static_cols).head(1).to_numpy()[0]
            
            if seq_len >= SEQ_LEN:
                # Truncate if too long
                seq_x    = group.select(self.tv_cols).head(SEQ_LEN).to_numpy()
                targets  = group.select(ALL_TARGETS).head(SEQ_LEN).to_numpy()
                weights  = group["weight"].head(SEQ_LEN).to_numpy()
            else:
                # Pad if too short (initialize arrays with zeros)
                seq_x   = np.zeros((SEQ_LEN, len(self.tv_cols)), dtype=np.float32)
                targets = np.zeros((SEQ_LEN, len(ALL_TARGETS)), dtype=np.float32)
                weights = np.zeros(SEQ_LEN, dtype=np.float32)
                
                # Insert the actual data into the beginning of the arrays
                seq_x[:seq_len, :]   = group.select(self.tv_cols).to_numpy()
                targets[:seq_len, :] = group.select(ALL_TARGETS).to_numpy()
                weights[:seq_len]    = group["weight"].to_numpy()

            self.sequences.append((
                static_x.astype(np.float32),
                seq_x.astype(np.float32),
                targets.astype(np.float32),
                weights.astype(np.float32),
            ))

        print(f"  Built {len(self.sequences)} sequences")

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


class JaneStreetGRU(nn.Module):
    def __init__(self, static_dim: int, tv_dim: int,
                 hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, num_targets: int = 4):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers

        # Static encoder: projects static features to initial hidden state
        # Output: (num_layers, hidden_dim) per sequence
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_layers * hidden_dim),
        )

        # GRU: processes time-varying features sequentially
        self.gru = nn.GRU(
            input_size=tv_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,       # (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output head: maps GRU output to target predictions
        # This is the layer we fine-tune online
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_targets),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def encode_static(self, static_x: torch.Tensor) -> torch.Tensor:
        """
        Project static features to initial GRU hidden state.
        static_x: (batch, static_dim)
        returns:  (num_layers, batch, hidden_dim)
        """
        h = self.static_encoder(static_x)           # (batch, num_layers * hidden_dim)
        h = h.view(-1, self.num_layers, self.hidden_dim)  # (batch, layers, hidden)
        h = h.permute(1, 0, 2).contiguous()         # (layers, batch, hidden)
        return h

    def forward(self, static_x: torch.Tensor,
                seq_x: torch.Tensor) -> torch.Tensor:
        """
        static_x: (batch, static_dim)
        seq_x:    (batch, seq_len, tv_dim)
        returns:  (batch, seq_len, num_targets)
        """
        h0 = self.encode_static(static_x)
        gru_out, _ = self.gru(seq_x, h0)   # (batch, seq_len, hidden_dim)
        return self.output_head(gru_out)    # (batch, seq_len, num_targets)

    def freeze_gru(self):
        """Freeze GRU and static encoder for online fine-tuning."""
        for param in self.gru.parameters():
            param.requires_grad = False
        for param in self.static_encoder.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True


def weighted_r2_torch(y_true: torch.Tensor, y_pred: torch.Tensor,
                      weights: torch.Tensor) -> float:
    """Weighted R² on primary target (index 0 = responder_6)."""
    y_t = y_true[:, :, 0].flatten()
    y_p = y_pred[:, :, 0].flatten()
    w   = weights.flatten()

    ss_res = (w * (y_t - y_p) ** 2).sum()
    y_mean = (w * y_t).sum() / w.sum()
    ss_tot = (w * (y_t - y_mean) ** 2).sum()
    return (1 - ss_res / (ss_tot + 1e-8)).item()


def multitask_loss(y_pred: torch.Tensor, y_true: torch.Tensor,
                   weights: torch.Tensor,
                   aux_weight: float = 0.3) -> torch.Tensor:
    """
    Weighted MSE loss on primary target + auxiliary targets.
    Primary gets (1 - aux_weight) of total loss weight.
    Auxiliary targets split the remaining aux_weight equally.
    """
    w = weights.unsqueeze(-1)  # (batch, seq, 1)

    # Primary target loss (responder_6 = index 0)
    primary_loss = (w * (y_pred[:, :, 0:1] - y_true[:, :, 0:1]) ** 2).mean()

    # Auxiliary target losses
    aux_loss = (w * (y_pred[:, :, 1:] - y_true[:, :, 1:]) ** 2).mean()

    return (1 - aux_weight) * primary_loss + aux_weight * aux_loss


def train_epoch(model: JaneStreetGRU, loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                device: torch.device) -> tuple:
    model.train()
    total_loss = 0.0
    total_r2   = 0.0
    n_batches  = 0

    for static_x, seq_x, targets, weights in loader:
        static_x = static_x.to(device)
        seq_x    = seq_x.to(device)
        targets  = targets.to(device)
        weights  = weights.to(device)

        optimizer.zero_grad()
        preds = model(static_x, seq_x)
        loss  = multitask_loss(preds, targets, weights)
        loss.backward()

        # Gradient clipping — important for RNNs
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_r2   += weighted_r2_torch(targets, preds, weights)
        n_batches  += 1

    return total_loss / n_batches, total_r2 / n_batches


@torch.no_grad()
def eval_epoch(model: JaneStreetGRU, loader: DataLoader,
               device: torch.device) -> tuple:
    model.eval()
    total_loss = 0.0
    total_r2   = 0.0
    n_batches  = 0

    for static_x, seq_x, targets, weights in loader:
        static_x = static_x.to(device)
        seq_x    = seq_x.to(device)
        targets  = targets.to(device)
        weights  = weights.to(device)

        preds     = model(static_x, seq_x)
        loss      = multitask_loss(preds, targets, weights)
        total_loss += loss.item()
        total_r2   += weighted_r2_torch(targets, preds, weights)
        n_batches  += 1

    return total_loss / n_batches, total_r2 / n_batches


def load_partition_for_gru(path: Path, feature_cols: list) -> pl.DataFrame:
    """Load a cached partition, drop time_id=0 lag nulls, fill remaining."""
    return (
        pl.read_parquet(path)
        .filter(pl.col("time_id") > 0)
        .fill_null(0.0)
        .select(feature_cols + ["date_id", "time_id", "symbol_id",
                                "weight"] + ALL_TARGETS)
    )


def train_gru(
    n_epochs: int = 5,
    batch_size: int = 32,
    hidden_dim: int = 128,
    lr: float = 1e-3,
    val_partition_id: int = 8,  # hold out last two partitions for val
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    partition_paths = sorted(CACHE_DIR.glob("partition_*.parquet"))
    train_paths = [p for p in partition_paths
                   if int(p.stem.split("_")[1]) < val_partition_id]
    val_paths   = [p for p in partition_paths
                   if int(p.stem.split("_")[1]) >= val_partition_id]

    # Get feature cols from sample
    sample = pl.read_parquet(partition_paths[0]).head(100)
    exclude = set(["date_id", "time_id", "symbol_id", "weight"] + ALL_RESPONDERS)
    feature_cols = [c for c in sample.columns if c not in exclude]
    static_cols, tv_cols = get_feature_splits(feature_cols)
    print(f"Static features: {len(static_cols)}, Time-varying: {len(tv_cols)}")

    # Build validation dataset once
    print("Building validation dataset...")
    val_dfs = [load_partition_for_gru(p, feature_cols) for p in val_paths]
    val_df  = pl.concat(val_dfs)
    val_ds  = JaneStreetSequenceDataset(val_df, feature_cols)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=0)
    del val_dfs, val_df

    # Initialize model
    model = JaneStreetGRU(
        static_dim=len(static_cols),
        tv_dim=len(tv_cols),
        hidden_dim=hidden_dim,
        num_layers=2,
        dropout=0.2,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs * len(train_paths)
    )

    best_val_r2 = -np.inf
    best_model_path = MODEL_DIR / "gru_best.pt"

    for epoch in range(n_epochs):
        print(f"\n--- Epoch {epoch+1}/{n_epochs} ---")

        # Train partition by partition to keep memory bounded
        epoch_train_loss = 0.0
        epoch_train_r2   = 0.0
        n_parts = 0

        for path in train_paths:
            print(f"  Partition {path.stem}...")
            train_df = load_partition_for_gru(path, feature_cols)
            train_ds = JaneStreetSequenceDataset(train_df, feature_cols)
            train_loader = DataLoader(train_ds, batch_size=batch_size,
                                      shuffle=True, num_workers=0)
            del train_df

            loss, r2 = train_epoch(model, train_loader, optimizer, device)
            scheduler.step()
            epoch_train_loss += loss
            epoch_train_r2   += r2
            n_parts += 1
            print(f"    loss={loss:.4f}  train_r2={r2:.4f}")

        val_loss, val_r2 = eval_epoch(model, val_loader, device)
        print(f"  Epoch {epoch+1} summary: "
              f"train_r2={epoch_train_r2/n_parts:.4f}  "
              f"val_loss={val_loss:.4f}  val_r2={val_r2:.4f}")

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

def online_finetune_step(
    model: JaneStreetGRU, 
    static_x: torch.Tensor, 
    seq_x: torch.Tensor, 
    targets: torch.Tensor, 
    weights: torch.Tensor, 
    device: torch.device,
    lr: float = 1e-4
) -> float:
    """
    Performs a single online fine-tuning step using recently revealed targets.
    """
    model.train()
    
    # 1. Freeze the deep layers (static encoder + GRU)
    model.freeze_gru()
    
    # 2. Initialize optimizer ONLY for the unfrozen parameters (output head)
    # Filtering required_grad parameters saves memory and compute
    finetune_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, 
        weight_decay=1e-4
    )
    
    static_x = static_x.to(device)
    seq_x    = seq_x.to(device)
    targets  = targets.to(device)
    weights  = weights.to(device)

    finetune_optimizer.zero_grad()
    
    # 3. Forward pass
    preds = model(static_x, seq_x)
    loss  = multitask_loss(preds, targets, weights)
    
    # 4. Backward pass (only computes gradients for the output head)
    loss.backward()
    
    # 5. Update weights
    finetune_optimizer.step()
    
    # (Optional) Unfreeze all if you share this model instance with base training
    # model.unfreeze_all()
    
    return loss.item()


if __name__ == "__main__":
    train_gru(n_epochs=5, batch_size=32, hidden_dim=128, lr=1e-3)