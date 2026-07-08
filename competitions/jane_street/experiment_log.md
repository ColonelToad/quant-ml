# Jane Street Real-Time Market Data Forecasting — Experiment Log

**Metric:** Weighted R² (higher is better)
**Validation set:** Partitions 8-9 (held out throughout)
**Naive baseline:** Lag-1 predictor alone → **0.7943**
**Training data:** date_id ≥ 247, partitions 1-7

---

## Models

| # | Model | Val R² | Params | Key Design Choice | What I Learned |
|---|-------|--------|--------|-------------------|----------------|
| 1 | LightGBM baseline | 0.8789 | — | Walk-forward CV, sample_weight | `responder_6_lag_1` dominates at 440M gain. Weighted R² of 0.88 is real — confirmed by naive lag baseline at 0.794 |
| 2 | GRU | 0.8866 | 229,700 | Static → hidden init, multi-task head r6/r3/r7/r8 | +0.008 over LGBM. Sequential modeling adds genuine signal. Partition 1 consistently harder (earlier regime) |
| 3 | Multitask AutoEncoder | 0.8019 | 99,505 | Bottleneck 64d, reconstruction loss as regularizer | Trails GRU by 0.085 — row-wise misses sequential structure. Real value is as feature extractor into LGBM |
| 4 | TFT | TBD | TBD | VSN, windowed attention (32 steps), static context gates | TBD |
| 5 | Cross-Symbol Transformer | TBD | TBD | Attention across symbols at each time step | TBD |
| 6 | KAN | TBD | TBD | Learnable spline activations | TBD |

---

## Key EDA Findings

**Autocorrelation decay (consistent across all symbols):**
- Lag 1:  ~0.89 — dominant signal
- Lag 5:  ~0.65 — meaningful
- Lag 10: ~0.40 — moderate
- Lag 20: ~0.00 — essentially gone
- Lag 50+: slightly negative (mean reversion)
- **Implication:** attention window of 32 steps captures all useful context

**Target structure:**
- responder_6: clipped ±5, mean ≈ 0, std ≈ 0.87
- Co-train with: r3 (corr 0.45), r7 (0.43), r8 (0.44)
- Drop from co-training: r0/r1/r2 (corr near zero or negative)
- weight: range 0.44–6.01, no zeros — mandatory as sample_weight

**Feature nulls:**
- Structural date cutoff: train from date_id ≥ 247 (all features present)
- Symbol-specific nulls: feature_21/26/27/31 absent for some symbols
- Treatment: fill with 0 + binary indicator flag

---

## Architecture Decisions

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| TFT attention window | 32 steps | Autocorr gone by lag-20 |
| Multi-task targets | r6 + r3/r7/r8 | Corr > 0.43; r0/r1/r2 would hurt |
| AE role | Feature extractor only | Bottleneck → LGBM, not standalone |
| Online learning | Freeze base, fine-tune output head | Fewer params, base preserved |
| Training cutoff | date_id ≥ 247 | Full feature availability |
| Sequence length | 849 steps | 95% of sequences this length |

---

## Ensemble Plan

1. LightGBM — tabular feature interactions
2. GRU — temporal autocorrelation
3. Residual MLP — nonlinear feature combinations
4. AE embeddings → LightGBM — denoised feature representation

Online learning: GRU + TFT fine-tune output head per batch. LGBM static.

---

## Open Questions

- Does TFT's VSN outperform GRU's implicit feature weighting?
- Does cross-symbol attention capture market-wide patterns per-symbol models miss?
- Does AE bottleneck as LGBM input improve over raw features?
- Does KAN find patterns MLP with fixed activations misses?
- What is the optimal ensemble weighting?