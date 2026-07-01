"""
competitions/optiver_close/scripts/features.py
----------------------------------------------
Feature engineering for Optiver Trading at the Close (2023).

Data: train.csv — real NASDAQ closing auction data, 10-minute sequential buckets
Target: 60-second future price movement relative to synthetic index
Grain: (stock_id, date_id, seconds_in_bucket)

Key difference from Optiver Volatility: sequential bucket structure matters here.
Predictions must respect bucket ordering — no future bucket leakage.
"""

# TODO: implement after completing optiver_volatility
