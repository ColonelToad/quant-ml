# Jane Street Real-Time Market Data Forecasting — Experiment Log

**Metric:** Weighted R² (higher is better)
**Validation set:** Partitions 8-9 (held out throughout)
**Naive baseline:** Lag-1 predictor alone → **0.7943**
**Training data:** date_id ≥ 247, partitions 1-7

---

## Models

| # | Model | Val R² | Params | Key Design Choice | What I Learned |
|---|-------|--------|--------|-------------------|----------------|
| 1 | LightGBM baseline | -0.4480 | — | Walk-forward CV, sample_weight | Heavily overfit to historical regimes. Catastrophically failed on hidden validation partition shifts (-0.4480), highlighting why non-linear sequential modeling is vital for stationary domain changes. |
| 2 | GRU (Stateless vs Stateful) | 0.8521 | 229,700 | Static → hidden init, multi-task head r6/r3/r7/r8 | Dropped to **0.8487** under stateless step-by-step evaluation due to step-wise "amnesia". Reclaiming statefulness via per-symbol hidden state propagation pushed tracking to **0.8521**. |
| 3 | Multitask AutoEncoder | 0.8019 | 99,505 | Bottleneck 64d, reconstruction loss as regularizer | Trails GRU by 0.085 — row-wise misses sequential structure. Real value is as feature extractor into LGBM |
| 4 | TFT | 0.8534 | 850,248 | VSN, windowed attention (32 steps), static context gates | High performance offline, but incredibly computationally heavy and complex to maintain in real-time constraint environments. |
| 5 | Cross-Symbol Transformer | 0.8396 | 26,500 | Attention across symbols at each time step | Massive success. Became the structural anchor of the ensemble. By shifting the attention window from time to spatial (cross-symbol), it effectively bypassed the statefulness tracking requirement. |
| 6 | KAN | TBD | TBD | Learnable spline activations | TBD |
| **7** | **Final Blended Ensemble** | **0.8651** | — | SLSQP optimized blend (3x CST seeds + Stateful GRU) | The ultimate winner. Blending spatial (CST) and temporal (GRU) inductive biases out-performed any standalone framework, yielding a robust **0.8651** R² score. |

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

## Architecture & Production Decisions

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| TFT attention window | 32 steps | Autocorr gone by lag-20 |
| Multi-task targets | r6 + r3/r7/r8 | Corr > 0.43; r0/r1/r2 would hurt |
| Streaming Engine | NumPy Vectorization | Polars/Pandas `.filter()` reallocates memory on every single time slice, causing catastrophic CPU OOM crashes across a 327k stream. |
| Tensor Management | Explicit Detach + Clone | PyTorch slice views secretly preserve full-batch computational graphs. Explicit `.detach().clone()` on hidden states isolates VRAM footprint. |
| State Robustness | Per-Symbol Zero-Padding | Missing symbols in streaming batches break tensor stacking (`torch.stack`). Masking and backing with static zeros guarantees loop stability. |
| Sequence length | 849 steps | 95% of sequences this length |

---

## Ensemble Plan

1. **Cross-Symbol Transformer Sub-Ensemble (3 Seeds):** Spatial features capturing cross-asset momentum.
2. **Jane Street GRU:** Temporal features capturing intraday momentum.
3. **Weights:** Optimized using SciPy SLSQP bounds (CST heavily favored at ~88% overall mass, GRU providing stabilizing ~12% context).

Online learning: GRU fine-tunes output head dynamically using revealed lag payloads.

---

## Open Questions — Answered

- **Does TFT's VSN outperform GRU's implicit feature weighting?** 
  * TFT performed admirably but fell short of the stateful GRU + CST ensemble, while presenting an unviable computational footprint for live real-time latency thresholds.
- **Does cross-symbol attention capture market-wide patterns per-symbol models miss?** 
  * Yes. The Cross-Symbol Transformer (CST) single-handedly saved the ensemble from the LightGBM baseline collapse, carrying an 88% weight attribution in the optimized blend.
- **How do you keep memory bounded during live PyTorch streams?** 
  * By migrating data frames out of the loop into contiguous C-NumPy blocks, computing index boundaries analytically, and clearing graph history using detached tensor clones.
- **What is the optimal ensemble weighting?** 
  * A stark division of labor: 29.15% (CST Seed 42), 29.56% (CST Seed 100), 29.34% (CST Seed 999), and 11.94% (Stateful GRU).