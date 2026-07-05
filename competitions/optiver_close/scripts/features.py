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

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "optiver-close-trading"
CACHE_DIR = Path(__file__).resolve().parents[1]


def load_train_polars() -> pl.DataFrame:
    """Load train.csv as Polars DataFrame. Polars is lazy by default."""
    return pl.read_csv(
        DATA_DIR / "train.csv",
        null_values=["NaN", "nan"],
        schema_overrides={
            "far_price": pl.Float64,
            "near_price": pl.Float64
        }
    )


def build_features_polars(df: pl.DataFrame) -> pl.DataFrame:
    """
    Build features using Polars lazy evaluation.
    All computations are pushed down to Arrow before materializing.
    """
    # Ensure sorted by stock, date, then bucket (critical for lags and rolling)
    df = df.sort(["stock_id", "date_id", "seconds_in_bucket"])

    # Per-bucket spread and imbalance
    df = df.with_columns(
        spread=(pl.col("ask_price") - pl.col("bid_price")).alias("spread"),
        mid_price=((pl.col("bid_price") + pl.col("ask_price")) / 2).alias("mid_price"),
        # Imbalance as a signed ratio
        imbalance_ratio=(
            (pl.col("imbalance_size") * 
             (2 * pl.col("imbalance_buy_sell_flag") - 1)) / 
            (pl.col("imbalance_size") + 1e-8)
        ).alias("imbalance_ratio"),
    )

    # Log returns of WAP within each (stock, date)
    # Critical: shift by 1 to get lag-1, nullify group boundaries
    df = df.with_columns(
        log_wap=pl.col("wap").log(),
    ).with_columns(
        log_ret_wap=(
            pl.col("log_wap")
            .diff()
            .over(["stock_id", "date_id"])
        ).alias("log_ret_wap"),
    )

    # Nullify the first bucket of each day (no prior data)
    df = df.with_columns(
        log_ret_wap=pl.when(
            pl.col("seconds_in_bucket") == 0
        ).then(None).otherwise(pl.col("log_ret_wap")),
    )

    # Cumulative features within each day (up to current bucket, not including)
    # Vol: sqrt of sum of squared returns up to (but not including) current bucket
    df = df.with_columns(
        cumsum_sq_ret=(
            (pl.col("log_ret_wap") ** 2)
            .cum_sum()
            .over(["stock_id", "date_id"])
        ).alias("cumsum_sq_ret"),
    )

    # Shift to remove current bucket from cumulative
    df = df.with_columns(
        cumsum_sq_ret_prior=pl.col("cumsum_sq_ret").shift(1).over(["stock_id", "date_id"]),
    )

    df = df.with_columns(
        rv_so_far=pl.col("cumsum_sq_ret_prior").sqrt().alias("rv_so_far"),
    )

    # Imbalance trend: has buy pressure been building or fading?
    df = df.with_columns(
        imbalance_sum_prior=(
            pl.col("imbalance_size")
            .shift(1)
            .cum_sum()
            .over(["stock_id", "date_id"])
        ).alias("imbalance_sum_prior"),
    )

    # Bucket-level aggregates
    df = df.with_columns(
        bucket_matched_sum=pl.col("matched_size").sum().over(
            ["stock_id", "date_id", "seconds_in_bucket"]
        ),
    )

    # Lag-1 features: previous bucket's state (respects causality)
    df = df.with_columns(
        spread_lag1=pl.col("spread").shift(1).over(["stock_id", "date_id"]),
        imbalance_size_lag1=pl.col("imbalance_size").shift(1).over(["stock_id", "date_id"]),
        matched_size_lag1=pl.col("matched_size").shift(1).over(["stock_id", "date_id"]),
        wap_lag1=pl.col("wap").shift(1).over(["stock_id", "date_id"]),
    )

    # Price drift from previous bucket (directional signal)
    # Let it be null, then fill with 0 after is_first_bucket is added
    df = df.with_columns(
        wap_drift=(
            pl.when(pl.col("seconds_in_bucket") == 0)
            .then(pl.lit(None))
            .otherwise(pl.col("wap") / pl.col("wap_lag1") - 1)
        ).alias("wap_drift"),
    )

    # Rolling features: mean and std over last 5 buckets (respecting causality)
    df = df.with_columns(
        spread_roll5_mean=pl.col("spread").rolling_mean(5).over(["stock_id", "date_id"]),
        imbalance_roll5_mean=pl.col("imbalance_size").rolling_mean(5).over(["stock_id", "date_id"]),
    )

    # --- Phase 2: Feature Expansion ---

    # 1. Cross-Stock Features (Market-wide signal)
    # Calculate market-wide average drift per bucket
    df = df.with_columns(
        global_wap_drift=pl.col("wap_drift").mean().over(["date_id", "seconds_in_bucket"])
    )
    
    # Relative drift: how is this specific stock moving compared to the market average?
    df = df.with_columns(
        relative_wap_drift=(pl.col("wap_drift") - pl.col("global_wap_drift")).alias("relative_wap_drift")
    )

    # 2. Imbalance Momentum
    df = df.with_columns(
        # Raw change in size
        imbalance_size_momentum=(
            pl.col("imbalance_size") - pl.col("imbalance_size").shift(1).over(["stock_id", "date_id"])
        ).alias("imbalance_size_momentum"),
        
        # Change in the signed ratio (shows direction flips)
        imbalance_ratio_momentum=(
            pl.col("imbalance_ratio") - pl.col("imbalance_ratio").shift(1).over(["stock_id", "date_id"])
        ).alias("imbalance_ratio_momentum"),
    )

    # 3. Far/Near Price Spread (with Flagging approach)
    df = df.with_columns(
        far_near_spread=(pl.col("far_price") - pl.col("near_price")).alias("far_near_spread"),
        # Create explicit flag for early buckets where auction cross isn't broadcast yet
        is_auction_cross_missing=pl.col("far_price").is_null().cast(pl.Int8)
    )

    df = df.with_columns(
        imbalance_size_rank=pl.col("imbalance_size").rank().over(["date_id", "seconds_in_bucket"]),
        matched_size_rank=pl.col("matched_size").rank().over(["date_id", "seconds_in_bucket"]),
        spread_rank=pl.col("spread").rank().over(["date_id", "seconds_in_bucket"])
    )

    # --- Null Fill and Flagging Block ---
    
    # 1. Add is_first_bucket flag
    df = df.with_columns(
        is_first_bucket=pl.col("seconds_in_bucket").eq(0).cast(pl.Int8)
    )

    # 2. Fill lag features with current bucket value where null
    df = df.with_columns(
        spread_lag1=pl.col("spread_lag1").fill_null(pl.col("spread")),
        imbalance_size_lag1=pl.col("imbalance_size_lag1").fill_null(pl.col("imbalance_size")),
        matched_size_lag1=pl.col("matched_size_lag1").fill_null(pl.col("matched_size")),
        wap_lag1=pl.col("wap_lag1").fill_null(pl.col("wap")),
    )

    # 3. Fill early bucket structural nulls with 0.0
    zero_fill_cols = [
        "rv_so_far", 
        "wap_drift", 
        "global_wap_drift", 
        "relative_wap_drift",
        "imbalance_size_momentum", 
        "imbalance_ratio_momentum",
        "far_near_spread"
    ]
    df = df.with_columns([
        pl.col(c).fill_null(0.0) for c in zero_fill_cols
    ])

    # 4. Rolling fills — early buckets where window isn't full yet
    df = df.with_columns(
        spread_roll5_mean=pl.col("spread_roll5_mean").fill_null(pl.col("spread")),
        imbalance_roll5_mean=pl.col("imbalance_roll5_mean").fill_null(pl.col("imbalance_size")),
    )

    # 5. Rolling fills — early buckets where window isn't full yet
    df = df.with_columns(
        spread_roll5_mean=pl.col("spread_roll5_mean").fill_null(pl.col("spread")),
        imbalance_roll5_mean=pl.col("imbalance_roll5_mean").fill_null(pl.col("imbalance_size")),
    )

    # Select and return clean features
    feature_cols = [
        "stock_id", "date_id", "seconds_in_bucket", "time_id",
        "is_first_bucket",
        "imbalance_size", "imbalance_buy_sell_flag",
        "spread", "imbalance_ratio",
        "rv_so_far", "wap_drift",
        "spread_lag1", "imbalance_size_lag1", "matched_size_lag1",
        "spread_roll5_mean", "imbalance_roll5_mean",
        "reference_price", "matched_size", "near_price", "far_price",
        "bid_price", "bid_size", "ask_price", "ask_size", "wap",
        "target", "global_wap_drift", "relative_wap_drift",
        "imbalance_size_momentum", "imbalance_ratio_momentum",
        "far_near_spread", "is_auction_cross_missing",
        "imbalance_size_rank", "matched_size_rank", "spread_rank"
    ]

    # Resolve schema to avoid PerformanceWarning
    schema_cols = df.collect_schema().names()
    return df.select([c for c in feature_cols if c in schema_cols])


def main():
    """Load, process, and cache features."""
    print("Loading data with Polars...")
    df_lazy = load_train_polars().lazy()

    print("Building features...")
    df_feats = build_features_polars(df_lazy)

    print("Materializing and saving...")
    df_result = df_feats.collect()

    cache_path = CACHE_DIR / "features_cache.parquet"
    df_result.write_parquet(cache_path)
    print(f"Saved to {cache_path}")
    print(f"Shape: {df_result.shape}")
    print(f"Null counts:\n{df_result.null_count()}")


if __name__ == "__main__":
    main()