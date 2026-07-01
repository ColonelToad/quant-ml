"""
competitions/jane_street/scripts/features.py
--------------------------------------------
Feature engineering for Jane Street Real-Time Market Data Forecasting (2024).

Data: train.parquet, lags.parquet — anonymized market responders
Target: responder_6 (primary), with responders 0-8 available
Grain: (date_id, time_id, symbol_id)

Key difference: streaming inference via kaggle_evaluation API.
Model must handle a time-series API — no batch prediction over full test set.
"""

# TODO: implement after completing both Optiver competitions
