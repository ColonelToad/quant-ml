"""
competitions/jane_street/scripts/model_tft.py
---------------------------------------------
Temporal Fusion Transformer for Jane Street Real-Time Market Data.

Architecture:
    Static pathway:  12 static features → GRN → static context vectors
    Temporal pathway: 97 time-varying features → VSN → LSTM encoder → 
                      multi-head self-attention (window=32) → GRN → output
    Output head: multi-task prediction of responder_6/3/7/8

Key design decisions from EDA:
    - Attention window = 32 steps (autocorr gone by lag-20, window covers it)
    - known_future = time_of_day (the one feature known ahead of time)
    - Static context used to initialize LSTM and enrich attention
    - Full attention over 32 steps: O(32²) = trivial, no sparse tricks needed

Reference: Lim et al. "Temporal Fusion Transformers for Interpretable
           Multi-horizon Time Series Forecasting" (2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import polars as pl
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

CACHE_DIR = Path(__file__).resolve().parents[1] / "feature_cache"
MODEL_DIR  = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True)

PRIMARY_TARGET = "responder_6"
AUX_TARGETS    = ["responder_3", "responder_7", "responder_8"]
ALL_TARGETS    = [PRIMARY_TARGET] + AUX_TARGETS
ALL_RESPONDERS = [f"responder_{i}" for i in range(9)]
EXCLUDE_COLS   = set(["date_id", "time_id", "symbol_id", "weight"] + ALL_RESPONDERS)
ATTN_WINDOW    = 32   # steps per attention window
SEQ_LEN        = 849  # full sequence length

STATIC_FEATURES = [
    "feature_09", "feature_10", "feature_11", "feature_20",
    "feature_22", "feature_23", "feature_24", "feature_25",
    "feature_28", "feature_29", "feature_30", "feature_61",
]
KNOWN_FUTURE = ["time_of_day"]  # only feature known ahead of time


def get_feature_splits(feature_cols: list):
    static = [f for f in feature_cols if f in STATIC_FEATURES]
    future = [f for f in feature_cols if f in KNOWN_FUTURE]
    observed = [f for f in feature_cols
                if f not in STATIC_FEATURES and f not in KNOWN_FUTURE]
    return static, observed, future


# ── Core TFT building blocks ─────────────────────────────────────────────────

class GRN(nn.Module):
    """
    Gated Residual Network — fundamental TFT building block.
    Applies a gated nonlinear transformation with residual connection.
    Gate learns how much of the transformation to apply — useful for
    skipping irrelevant transformations on noisy financial features.
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 output_dim: int = None, dropout: float = 0.1,
                 context_dim: int = None):
        super().__init__()
        output_dim = output_dim or input_dim

        self.fc1  = nn.Linear(input_dim + (context_dim or 0), hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.drop = nn.Dropout(dropout)

        # Residual projection if dims differ
        self.residual = (
            nn.Linear(input_dim, output_dim, bias=False)
            if input_dim != output_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor,
                context: torch.Tensor = None) -> torch.Tensor:
        residual = self.residual(x)
        if context is not None:
            x = torch.cat([x, context], dim=-1)
        h = F.elu(self.fc1(x))
        h = self.drop(h)
        gate = torch.sigmoid(self.gate(h))
        out  = gate * self.fc2(h)
        return self.norm(out + residual)


class VSN(nn.Module):
    """
    Variable Selection Network — learns which features matter most
    at each time step. Produces a weighted combination of per-feature
    GRN outputs. Directly addresses the 'which of 97 features are
    useful?' problem without manual feature selection.
    """
    def __init__(self, input_dim: int, num_features: int,
                 hidden_dim: int, dropout: float = 0.1,
                 context_dim: int = None):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim   = hidden_dim

        # Per-feature GRNs: each feature gets its own transformation
        self.feature_grns = nn.ModuleList([
            GRN(1, hidden_dim, hidden_dim, dropout)
            for _ in range(num_features)
        ])

        # Selection weights: softmax over all features
        self.selection_grn = GRN(
            input_dim, hidden_dim, num_features,
            dropout, context_dim=context_dim
        )

    def forward(self, x: torch.Tensor,
                context: torch.Tensor = None) -> torch.Tensor:
        """
        x: (..., num_features) — last dim is features
        context: (..., context_dim) optional static context
        returns: (..., hidden_dim) weighted feature combination
        """
        # Per-feature transformations
        feature_outputs = []
        for i, grn in enumerate(self.feature_grns):
            feat = x[..., i:i+1]  # (..., 1)
            feature_outputs.append(grn(feat))
        # Stack: (..., num_features, hidden_dim)
        features = torch.stack(feature_outputs, dim=-2)

        # Selection weights: (..., num_features)
        weights = torch.softmax(
            self.selection_grn(x, context), dim=-1
        ).unsqueeze(-1)  # (..., num_features, 1)

        # Weighted sum: (..., hidden_dim)
        return (weights * features).sum(dim=-2)


class TFTAttention(nn.Module):
    """
    Interpretable multi-head attention with causal masking.
    Causal: position T can only attend to positions 0..T-1.
    Uses shared value projection across heads for interpretability.
    Window size controls how far back attention reaches.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)  # shared across heads
        self.out    = nn.Linear(hidden_dim, hidden_dim)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, hidden_dim)
        Applies causal self-attention — each position attends to past only.
        """
        B, T, D = x.shape
        H, Dh   = self.num_heads, self.head_dim

        Q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)  # (B,H,T,Dh)
        K = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        #V = self.v_proj(x).view(B, T, 1, Dh).transpose(1, 2).expand(B, T, H, Dh).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) * self.scale  # (B,H,T,T)

        # Causal mask: prevent attending to future positions
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1e9)

        attn = self.drop(torch.softmax(scores, dim=-1))
        out  = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


# ── Full TFT model ────────────────────────────────────────────────────────────

class JaneStreetTFT(nn.Module):
    def __init__(self, static_dim: int, obs_dim: int, future_dim: int,
                 hidden_dim: int = 128, num_heads: int = 4,
                 num_layers: int = 1, dropout: float = 0.2,
                 num_targets: int = 4, attn_window: int = ATTN_WINDOW):
        super().__init__()
        self.attn_window = attn_window
        self.hidden_dim  = hidden_dim

        # Static pathway: encode static features into context vectors
        self.static_vsn   = VSN(static_dim, static_dim, hidden_dim, dropout)
        self.static_grn_h = GRN(hidden_dim, hidden_dim, dropout=dropout)
        self.static_grn_c = GRN(hidden_dim, hidden_dim, dropout=dropout)
        self.static_grn_e = GRN(hidden_dim, hidden_dim, dropout=dropout)

        # Observed time-varying VSN (uses static context)
        self.obs_vsn = VSN(
            obs_dim, obs_dim, hidden_dim, dropout,
            context_dim=hidden_dim
        )

        # Known future VSN (time_of_day)
        self.future_linear = nn.Linear(future_dim, hidden_dim)

        # LSTM encoder: processes selected features sequentially
        # Initialized with static context vectors
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Post-LSTM gating
        self.post_lstm_gate = nn.Linear(hidden_dim, hidden_dim)
        self.post_lstm_norm = nn.LayerNorm(hidden_dim)

        # Enrichment GRN: uses static enrichment context
        self.enrich_grn = GRN(
            hidden_dim, hidden_dim, dropout=dropout,
            context_dim=hidden_dim
        )

        # Temporal self-attention over window
        self.attention  = TFTAttention(hidden_dim, num_heads, dropout)
        self.attn_gate  = nn.Linear(hidden_dim, hidden_dim)
        self.attn_norm  = nn.LayerNorm(hidden_dim)

        # Final GRN before output
        self.output_grn  = GRN(hidden_dim, hidden_dim, dropout=dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)

        # Multi-task output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_targets),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def encode_static(self, static_x: torch.Tensor):
        """
        static_x: (batch, static_dim)
        Returns context vectors for LSTM init and attention enrichment.
        """
        c_s = self.static_vsn(static_x)     # (batch, hidden)
        h0  = self.static_grn_h(c_s)        # LSTM hidden init
        c0  = self.static_grn_c(c_s)        # LSTM cell init
        c_e = self.static_grn_e(c_s)        # enrichment context
        return h0, c0, c_e

    def forward(self, static_x: torch.Tensor,
                obs_x: torch.Tensor,
                future_x: torch.Tensor) -> torch.Tensor:
        """
        static_x:  (batch, static_dim)
        obs_x:     (batch, seq_len, obs_dim)
        future_x:  (batch, seq_len, future_dim)
        returns:   (batch, seq_len, num_targets)
        """
        B, T, _ = obs_x.shape

        # 1. Static encoding
        h0, c0, c_e = self.encode_static(static_x)
        h0 = h0.unsqueeze(0)  # (1, batch, hidden)
        c0 = c0.unsqueeze(0)

        # 2. Variable selection on observed features
        #    Expand static context across time steps
        c_s_expanded = self.static_vsn(static_x).unsqueeze(1).expand(B, T, -1)
        # VSN processes each time step independently
        obs_selected = self.obs_vsn(
            obs_x.reshape(B * T, -1),
            c_s_expanded.reshape(B * T, -1)
        ).reshape(B, T, self.hidden_dim)

        # 3. Add known future information
        future_emb = self.future_linear(future_x)  # (B, T, hidden)
        lstm_input = obs_selected + future_emb

        # 4. LSTM sequence encoding
        lstm_out, _ = self.lstm(lstm_input, (h0, c0))

        # Post-LSTM gating
        gate = torch.sigmoid(self.post_lstm_gate(lstm_out))
        lstm_out = self.post_lstm_norm(gate * lstm_out + lstm_input)

        # 5. Static enrichment
        c_e_expanded = c_e.unsqueeze(1).expand(B, T, -1)
        enriched = self.enrich_grn(lstm_out, c_e_expanded)

        # 6. Windowed self-attention
        # Process sequence in non-overlapping windows of attn_window steps
        # Pad sequence to multiple of attn_window
        pad_len = (self.attn_window - T % self.attn_window) % self.attn_window
        if pad_len > 0:
            pad = torch.zeros(B, pad_len, self.hidden_dim, device=obs_x.device)
            enriched_pad = torch.cat([enriched, pad], dim=1)
        else:
            enriched_pad = enriched

        T_pad = enriched_pad.shape[1]
        n_windows = T_pad // self.attn_window

        # Reshape to windows, apply attention within each window
        windowed = enriched_pad.view(
            B * n_windows, self.attn_window, self.hidden_dim
        )
        attn_out = self.attention(windowed)
        attn_out = attn_out.view(B, T_pad, self.hidden_dim)[:, :T, :]

        # Post-attention gating
        gate2   = torch.sigmoid(self.attn_gate(attn_out))
        attn_out = self.attn_norm(gate2 * attn_out + enriched)

        # 7. Output
        out = self.output_grn(attn_out)
        out = self.output_norm(out)
        return self.output_head(out)

    def freeze_base(self):
        """Freeze everything except output_head for online fine-tuning."""
        for name, param in self.named_parameters():
            param.requires_grad = "output_head" in name


# ── Dataset ───────────────────────────────────────────────────────────────────

class JaneStreetTFTDataset(Dataset):
    def __init__(self, df: pl.DataFrame, feature_cols: list):
        self.static_cols, self.obs_cols, self.future_cols = \
            get_feature_splits(feature_cols)

        groups = (
            df.sort(["date_id", "symbol_id", "time_id"])
            .group_by(["date_id", "symbol_id"], maintain_order=True)
        )

        self.sequences = []
        for _, g in groups:
            n = len(g)
            static_x  = g.select(self.static_cols).head(1).to_numpy()[0]
            if n >= SEQ_LEN:
                obs_x    = g.select(self.obs_cols).head(SEQ_LEN).to_numpy()
                future_x = g.select(self.future_cols).head(SEQ_LEN).to_numpy()
                targets  = g.select(ALL_TARGETS).head(SEQ_LEN).to_numpy()
                weights  = g["weight"].head(SEQ_LEN).to_numpy()
            else:
                obs_x    = np.zeros((SEQ_LEN, len(self.obs_cols)),    dtype=np.float32)
                future_x = np.zeros((SEQ_LEN, len(self.future_cols)), dtype=np.float32)
                targets  = np.zeros((SEQ_LEN, len(ALL_TARGETS)),      dtype=np.float32)
                weights  = np.zeros(SEQ_LEN, dtype=np.float32)
                obs_x[:n]    = g.select(self.obs_cols).to_numpy()
                future_x[:n] = g.select(self.future_cols).to_numpy()
                targets[:n]  = g.select(ALL_TARGETS).to_numpy()
                weights[:n]  = g["weight"].to_numpy()

            self.sequences.append((
                static_x.astype(np.float32),
                obs_x.astype(np.float32),
                future_x.astype(np.float32),
                targets.astype(np.float32),
                weights.astype(np.float32),
            ))

        print(f"  Built {len(self.sequences)} sequences")

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        s, o, f, t, w = self.sequences[idx]
        return (torch.tensor(s), torch.tensor(o),
                torch.tensor(f), torch.tensor(t), torch.tensor(w))


# ── Loss and metrics ──────────────────────────────────────────────────────────

def weighted_r2(y_true, y_pred, weights):
    y_t = y_true[:, :, 0].flatten()
    y_p = y_pred[:, :, 0].flatten()
    w   = weights.flatten()
    ss_res = (w * (y_t - y_p) ** 2).sum()
    y_mean = (w * y_t).sum() / (w.sum() + 1e-8)
    ss_tot = (w * (y_t - y_mean) ** 2).sum()
    return (1 - ss_res / (ss_tot + 1e-8)).item()


def multitask_loss(pred, true, weights, aux_weight=0.3):
    w = weights.unsqueeze(-1)
    primary = (w * (pred[:, :, 0:1] - true[:, :, 0:1]) ** 2).mean()
    aux     = (w * (pred[:, :, 1:]  - true[:, :, 1:])  ** 2).mean()
    return (1 - aux_weight) * primary + aux_weight * aux


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = total_r2 = n = 0
    for s, o, f, t, w in loader:
        s, o, f, t, w = [x.to(device) for x in (s, o, f, t, w)]
        optimizer.zero_grad()
        pred = model(s, o, f)
        loss = multitask_loss(pred, t, w)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        total_r2   += weighted_r2(t, pred, w)
        n += 1
    return total_loss / n, total_r2 / n


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = total_r2 = n = 0
    for s, o, f, t, w in loader:
        s, o, f, t, w = [x.to(device) for x in (s, o, f, t, w)]
        pred = model(s, o, f)
        loss = multitask_loss(pred, t, w)
        total_loss += loss.item()
        total_r2   += weighted_r2(t, pred, w)
        n += 1
    return total_loss / n, total_r2 / n


def load_partition(path, feature_cols):
    return (
        pl.read_parquet(path)
        .filter(pl.col("time_id") > 0)
        .fill_null(0.0)
        .select(feature_cols + ["date_id", "time_id", "symbol_id",
                                "weight"] + ALL_TARGETS)
    )


def train_tft(n_epochs=5, batch_size=16, hidden_dim=128,
              num_heads=4, lr=1e-3, val_partition_id=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    paths = sorted(CACHE_DIR.glob("partition_*.parquet"))
    train_paths = [p for p in paths if int(p.stem.split("_")[1]) < val_partition_id]
    val_paths   = [p for p in paths if int(p.stem.split("_")[1]) >= val_partition_id]

    sample = pl.read_parquet(paths[0]).head(100)
    feature_cols = [c for c in sample.columns if c not in EXCLUDE_COLS]
    static_cols, obs_cols, future_cols = get_feature_splits(feature_cols)
    print(f"Static: {len(static_cols)}, Observed: {len(obs_cols)}, "
          f"Future: {len(future_cols)}")

    print("Building validation dataset...")
    val_df = pl.concat([load_partition(p, feature_cols) for p in val_paths])
    val_ds = JaneStreetTFTDataset(val_df, feature_cols)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=0)
    del val_df

    model = JaneStreetTFT(
        static_dim=len(static_cols),
        obs_dim=len(obs_cols),
        future_dim=len(future_cols),
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=0.2,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs * len(train_paths)
    )

    best_val_r2 = -np.inf
    best_path   = MODEL_DIR / "tft_best.pt"

    for epoch in range(n_epochs):
        print(f"\n--- Epoch {epoch+1}/{n_epochs} ---")
        ep_loss = ep_r2 = n_parts = 0

        for path in train_paths:
            print(f"  {path.stem}...")
            df = load_partition(path, feature_cols)
            ds = JaneStreetTFTDataset(df, feature_cols)
            loader = DataLoader(ds, batch_size=batch_size,
                                shuffle=True, num_workers=0)
            del df
            loss, r2 = train_epoch(model, loader, optimizer, device)
            scheduler.step()
            ep_loss += loss; ep_r2 += r2; n_parts += 1
            print(f"    loss={loss:.4f}  r2={r2:.4f}")

        val_loss, val_r2 = eval_epoch(model, val_loader, device)
        print(f"  Epoch {epoch+1}: train_r2={ep_r2/n_parts:.4f}  "
              f"val_r2={val_r2:.4f}  val_loss={val_loss:.4f}")

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "val_r2": val_r2, "feature_cols": feature_cols,
                "static_cols": static_cols, "obs_cols": obs_cols,
                "future_cols": future_cols,
            }, best_path)
            print(f"  Saved best (val_r2={val_r2:.4f})")

    print(f"\nBest val R²: {best_val_r2:.4f}")
    return model


if __name__ == "__main__":
    train_tft(n_epochs=5, batch_size=16, hidden_dim=128, num_heads=4, lr=1e-3)