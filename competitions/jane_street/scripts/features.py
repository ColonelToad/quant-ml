"""
competitions/jane_street/scripts/features.py
--------------------------------------------
Feature pipeline for Jane Street Real-Time Market Data Forecasting.

Key facts established from EDA:
- 47M rows total; train on date_id >= 247 (full feature availability)
- 39 symbols, 1699 dates, up to 968 time_ids per date
- time_ids are contiguous within a date (gap=1 always) — safe to shift by position
- 79 features: 12 static per (date,symbol), 63 time-varying
- feature_21/26/27/31: symbol-specific structural nulls — fill + indicator
- feature_39 and others: sparse partial nulls — fill with 0
- Targets: responder_6 (primary), responder_3/7/8 (auxiliary, corr > 0.43)
- weight: must be used as sample_weight in all models
- Responders already normalized, clipped at ±5
"""

import polars as pl
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "js-real-time"
CACHE_DIR = Path(__file__).resolve().parents[1]

# Feature categories from EDA
STATIC_FEATURES = [
    "feature_09", "feature_10", "feature_11", "feature_20",
    "feature_22", "feature_23", "feature_24", "feature_25",
    "feature_28", "feature_29", "feature_30", "feature_61",
]

SYMBOL_NULL_FEATURES = [
    "feature_21", "feature_26", "feature_27", "feature_31",
]

# All 79 features
ALL_FEATURES = [f"feature_{i:02d}" for i in range(79)]

# Time-varying = all features minus static and symbol-null
TIME_VARYING_FEATURES = [
    f for f in ALL_FEATURES
    if f not in STATIC_FEATURES and f not in SYMBOL_NULL_FEATURES
]

# Targets
PRIMARY_TARGET = "responder_6"
AUX_TARGETS = ["responder_3", "responder_7", "responder_8"]
ALL_TARGETS = [PRIMARY_TARGET] + AUX_TARGETS

# Lag columns provided at inference time
LAG_COLS = [f"responder_{i}_lag_1" for i in range(9)]

TRAIN_DATE_CUTOFF = 247  # First date with all features present


def load_partition(partition_id: int) -> pl.DataFrame:
    path = DATA_DIR / "train.parquet" / f"partition_id={partition_id}"
    return pl.read_parquet(path)


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Full feature pipeline. Input is raw partition data.
    Returns feature matrix with all engineered columns.
    """
    # Sort — critical for all shift-based lag operations
    df = df.sort(["date_id", "symbol_id", "time_id"])

    # 1. Handle symbol-specific structural nulls
    #    Fill with 0, add binary indicator per feature
    null_indicators = []
    for feat in SYMBOL_NULL_FEATURES:
        indicator_col = f"{feat}_missing"
        null_indicators.append(
            pl.col(feat).is_null().cast(pl.Int8).alias(indicator_col)
        )
    df = df.with_columns(null_indicators)
    df = df.with_columns([
        pl.col(f).fill_null(0.0) for f in SYMBOL_NULL_FEATURES
    ])

    # 2. Fill remaining sparse nulls with 0
    #    (feature_39 and others with partial nulls)
    sparse_null_feats = [
        f for f in ALL_FEATURES
        if f not in SYMBOL_NULL_FEATURES
    ]
    df = df.with_columns([
        pl.col(f).fill_null(0.0) for f in sparse_null_feats
    ])

    # 3. Lag-1 responder features
    #    Computed from training data; at inference time these
    #    come from lags.parquet provided by the API
    grp = ["date_id", "symbol_id"]
    lag_exprs = [
        pl.col(f"responder_{i}")
        .shift(1)
        .over(grp)
        .alias(f"responder_{i}_lag_1")
        for i in range(9)
    ]
    df = df.with_columns(lag_exprs)

    # 4. Rolling statistics on primary target lag
    #    Roll over time_id within (date, symbol)
    df = df.with_columns([
        pl.col("responder_6_lag_1")
        .rolling_mean(5, min_samples=1)
        .over(grp)
        .alias("r6_lag_roll5_mean"),

        pl.col("responder_6_lag_1")
        .rolling_std(5, min_samples=2)
        .over(grp)
        .alias("r6_lag_roll5_std"),
    ])

    # 5. Time-varying feature lags (lag-1 of top features)
    #    Only lag a subset — lagging all 63 is excessive
    #    Use tag groups to pick representatives
    key_tv_features = [
        "feature_00", "feature_01", "feature_05", "feature_06",
        "feature_12", "feature_15", "feature_39", "feature_42",
    ]
    tv_lag_exprs = [
        pl.col(f)
        .shift(1)
        .over(grp)
        .alias(f"{f}_lag_1")
        for f in key_tv_features
    ]
    df = df.with_columns(tv_lag_exprs)

    # 6. Cross-symbol features at each (date, time) step
    #    Market-level mean and std of key features
    #    These capture what the "market" is doing at each moment
    cross_grp = ["date_id", "time_id"]
    cross_exprs = [
        pl.col("responder_6_lag_1")
        .mean()
        .over(cross_grp)
        .alias("market_r6_lag_mean"),

        pl.col("responder_6_lag_1")
        .std()
        .over(cross_grp)
        .alias("market_r6_lag_std"),

        pl.col("feature_00")
        .mean()
        .over(cross_grp)
        .alias("market_f00_mean"),

        pl.col("feature_05")
        .mean()
        .over(cross_grp)
        .alias("market_f05_mean"),
    ]
    df = df.with_columns(cross_exprs)

    # 7. Symbol-relative features
    #    How does this symbol deviate from the market at this moment?
    df = df.with_columns([
        (pl.col("responder_6_lag_1") - pl.col("market_r6_lag_mean"))
        .alias("r6_lag_vs_market"),

        (pl.col("feature_00") - pl.col("market_f00_mean"))
        .alias("f00_vs_market"),
    ])

    # 8. Time position within day — useful for GRU and LightGBM
    #    Normalized to [0, 1] so the model knows where in the day it is
    max_time = df.select(pl.col("time_id").max()).item()
    df = df.with_columns(
        (pl.col("time_id") / max_time).alias("time_of_day")
    )

    return df


def get_feature_cols(df: pl.DataFrame) -> list:
    """Return all engineered feature column names for model input."""
    exclude = set(
        ["date_id", "time_id", "symbol_id", "weight"] +
        [f"responder_{i}" for i in range(9)] +
        ALL_FEATURES  # raw features included via engineering
    )
    # Include raw features + engineered features
    raw = ALL_FEATURES + SYMBOL_NULL_FEATURES
    
    # This automatically captures 'feature_21_missing', 'feature_26_missing', etc.
    engineered = [c for c in df.columns if c not in exclude and c not in raw]
    
    # Just return ALL_FEATURES + engineered
    return ALL_FEATURES + engineered


def build_all_partitions(
    force_rebuild: bool = False,
    date_cutoff: int = TRAIN_DATE_CUTOFF,
) -> None:
    """
    Process all partitions, filter to date_cutoff, cache per-partition.
    Keeps memory bounded — never loads full dataset simultaneously.
    """
    cache_dir = CACHE_DIR / "feature_cache"
    cache_dir.mkdir(exist_ok=True)

    for pid in range(10):
        cache_path = cache_dir / f"partition_{pid}.parquet"
        if cache_path.exists() and not force_rebuild:
            print(f"Partition {pid}: cached, skipping")
            continue

        print(f"Partition {pid}: loading...")
        df = load_partition(pid)

        # Filter to usable date range
        df = df.filter(pl.col("date_id") >= date_cutoff)

        if len(df) == 0:
            print(f"Partition {pid}: no rows after cutoff, skipping")
            continue

        print(f"Partition {pid}: building features ({len(df):,} rows)...")
        df = build_features(df)

        df.write_parquet(cache_path)
        print(f"Partition {pid}: saved to {cache_path}")

    print("All partitions processed.")


if __name__ == "__main__":
    print("Smoke test on partition 2 (first with full features)...")
    df = load_partition(2)
    df = df.filter(pl.col("date_id") >= TRAIN_DATE_CUTOFF)
    df = build_features(df)

    feat_cols = get_feature_cols(df)
    print(f"Shape: {df.shape}")
    print(f"Feature columns: {len(feat_cols)}")
    print(f"Null counts in engineered features:")
    print(df.select(feat_cols).null_count()
          .transpose(include_header=True, header_name="feature", column_names=["nulls"])
          .filter(pl.col("nulls") > 0))
    print(f"\nSample row:")
    print(df.select(["date_id", "time_id", "symbol_id",
                     "responder_6_lag_1", "market_r6_lag_mean",
                     "r6_lag_vs_market", "time_of_day"]).head(5))