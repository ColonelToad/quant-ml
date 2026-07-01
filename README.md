# quant-ml

A personal quantitative ML framework built across three Kaggle competitions as preparation for live Numerai competition.

## Structure

```
quant-ml/
  core/               # Shared primitives — CV, eval metrics, book features, feature engineering
  competitions/
    optiver_volatility/   # Optiver Realized Volatility Prediction (2021) — order book, RMSPE
    optiver_close/        # Optiver Trading at the Close (2023) — closing auction, sequential buckets
    jane_street/          # Jane Street Real-Time Market Data Forecasting (2024) — streaming API
  numerai/            # Live Numerai Classic — era-based, Spearman IC, neutralization
  data/               # Local data only — gitignored, never committed
```

## Competition Sequence

1. **Optiver Volatility** — Learn order book feature extraction, realized vol, RMSPE optimization
2. **Optiver Close** — Sequential bucket prediction, closing auction microstructure
3. **Jane Street Real-Time** — Streaming inference API, anonymized responders
4. **Numerai** — Live, sustained, era-based cross-sectional prediction

## Core Modules

| Module | Purpose |
|---|---|
| `core/book.py` | WAP, spread, imbalance, log returns, realized vol |
| `core/cv.py` | Purged K-Fold, walk-forward splits |
| `core/eval.py` | IC, ICIR, Sharpe, RMSPE, max drawdown |
| `core/features.py` | Z-score, rank normalize, neutralization, lag features |

## Setup

```bash
pip install -r requirements.txt
```

Data is not included. Download from Kaggle and place under `data/<competition-name>/`.
