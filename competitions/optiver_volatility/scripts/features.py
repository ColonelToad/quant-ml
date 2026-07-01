"""
competitions/optiver_volatility/scripts/features.py
----------------------------------------------------
Ultra-Low Memory Feature engineering for Optiver Realized Volatility.
"""

import numpy as np
import pandas as pd
import sys
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.book import (
    weighted_average_price, bid_ask_spread, 
    order_imbalance, depth_ratio, total_depth, wap_combined
)

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "optiver-volatility-pred"
CACHE_DIR = DATA_DIR / "feature_cache_v2"


def optimize_memory(df):
    """Downcasts float64 to float32 to instantly halve RAM usage."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    return df


def load_book(stock_id=None):
    path = DATA_DIR / "book_train.parquet"
    if stock_id is not None:
        df = pd.read_parquet(path / f"stock_id={stock_id}")
        df["stock_id"] = stock_id
        return optimize_memory(df)
    return optimize_memory(pd.read_parquet(path))


def load_trades(stock_id=None):
    path = DATA_DIR / "trade_train.parquet"
    if stock_id is not None:
        df = pd.read_parquet(path / f"stock_id={stock_id}")
        df["stock_id"] = stock_id
        return optimize_memory(df)
    return optimize_memory(pd.read_parquet(path))


def build_book_features(book):
    grp = ["stock_id", "time_id"]

    # Primitives
    book["wap1"] = weighted_average_price(book["bid_price1"], book["ask_price1"], book["bid_size1"], book["ask_size1"])
    book["wap2"] = weighted_average_price(book["bid_price2"], book["ask_price2"], book["bid_size2"], book["ask_size2"])
    book["wap_combined"] = wap_combined(book["wap1"], book["wap2"])
    book["spread1"] = bid_ask_spread(book["bid_price1"], book["ask_price1"])
    book["spread2"] = bid_ask_spread(book["bid_price2"], book["ask_price2"])
    book["imbalance1"] = order_imbalance(book["bid_size1"], book["ask_size1"])
    book["imbalance2"] = order_imbalance(book["bid_size2"], book["ask_size2"])
    book["bid_depth_ratio"] = depth_ratio(book["bid_size1"], book["bid_size2"])
    book["ask_depth_ratio"] = depth_ratio(book["ask_size1"], book["ask_size2"])
    book["total_depth"] = total_depth(book["bid_size1"], book["bid_size2"], book["ask_size1"], book["ask_size2"])
    
    # MEMORY TRICK: Sort and Diff instead of Groupby Transform
    book.sort_values(grp + ["seconds_in_bucket"], inplace=True)
    
    # Calculate global diffs
    book["log_ret_wap1"] = np.log(book["wap1"]).diff()
    book["log_ret_wap2"] = np.log(book["wap2"]).diff()
    book["log_ret_combined"] = np.log(book["wap_combined"]).diff()
    
    # Nullify the boundary rows where a new time_id starts
    is_first_row = book["time_id"] != book["time_id"].shift(1)
    book.loc[is_first_row, ["log_ret_wap1", "log_ret_wap2", "log_ret_combined"]] = np.nan

    # Fast Vectorized Half-Window Splits
    book["sq_ret_wap1"] = book["log_ret_wap1"] ** 2
    book["sq_ret_wap2"] = book["log_ret_wap2"] ** 2
    book["sq_ret_combined"] = book["log_ret_combined"] ** 2
    book["sq_ret_wap1_h1"] = book["sq_ret_wap1"].where(book["seconds_in_bucket"] < 300)
    book["sq_ret_wap1_h2"] = book["sq_ret_wap1"].where(book["seconds_in_bucket"] >= 300)

    # Cython aggregation
    agg = book.groupby(grp).agg(
        rv_wap1=("sq_ret_wap1", "sum"),
        rv_wap2=("sq_ret_wap2", "sum"),
        rv_combined=("sq_ret_combined", "sum"),
        rv_first_half=("sq_ret_wap1_h1", "sum"),
        rv_second_half=("sq_ret_wap1_h2", "sum"),
        spread1_mean=("spread1", "mean"),
        spread1_std=("spread1", "std"),
        spread2_mean=("spread2", "mean"),
        imbalance1_mean=("imbalance1", "mean"),
        imbalance1_std=("imbalance1", "std"),
        imbalance2_mean=("imbalance2", "mean"),
        bid_depth_ratio_mean=("bid_depth_ratio", "mean"),
        ask_depth_ratio_mean=("ask_depth_ratio", "mean"),
        total_depth_mean=("total_depth", "mean"),
        total_depth_std=("total_depth", "std"),
        wap1_mean=("wap1", "mean"),
        wap2_mean=("wap2", "mean"),
        wap1_first=("wap1", "first"),
        wap1_last=("wap1", "last"),
        book_update_count=("wap1", "count"),
    ).reset_index()

    # Free memory of the massive book dataframe
    del book
    gc.collect()

    vols = ["rv_wap1", "rv_wap2", "rv_combined", "rv_first_half", "rv_second_half"]
    agg[vols] = np.sqrt(agg[vols])

    agg["wap1_drift"] = np.log(agg["wap1_last"] / agg["wap1_first"])
    agg["rv_accel"] = agg["rv_second_half"] / (agg["rv_first_half"] + 1e-8)
    agg.drop(columns=["wap1_first", "wap1_last"], inplace=True)
    
    return agg


def build_trade_features(trades):
    agg = trades.groupby(["stock_id", "time_id"]).agg(
        trade_size_sum=("size", "sum"),
        trade_size_mean=("size", "mean"),
        trade_size_std=("size", "std"),
        trade_count=("order_count", "sum"),
        trade_price_std=("price", "std"),
        trade_price_mean=("price", "mean"),
    ).reset_index()
    agg["log_trade_volume"] = np.log1p(agg["trade_size_sum"])
    return agg


def build_features(stock_id=None):
    book = load_book(stock_id)
    trades = load_trades(stock_id)
    
    book_feats = build_book_features(book)
    trade_feats = build_trade_features(trades)
    
    df = book_feats.merge(trade_feats, on=["stock_id", "time_id"], how="left")
    return optimize_memory(df)


def build_all_features_cached(force_rebuild=False):
    CACHE_DIR.mkdir(exist_ok=True)
    final_file = CACHE_DIR.parent / "features_cache_v2.parquet"
    
    if final_file.exists() and not force_rebuild:
        print("Loading compiled cached features...")
        return pd.read_parquet(final_file)

    book_dir = DATA_DIR / "book_train.parquet"
    stock_ids = sorted([int(p.name.split("=")[1]) for p in book_dir.iterdir() if p.is_dir()])
    print(f"Building features for {len(stock_ids)} stocks...")
    
    for i, sid in enumerate(stock_ids):
        chunk_path = CACHE_DIR / f"features_{sid}.parquet"
        if chunk_path.exists() and not force_rebuild:
            continue
            
        if i % 10 == 0:
            print(f"  Processing stock {i}/{len(stock_ids)} (stock_id={sid})...")
            
        df = build_features(stock_id=sid)
        df.to_parquet(chunk_path, index=False)
        
        # Aggressive GC
        del df
        gc.collect()
        
    print("Compiling chunks...")
    chunks = [pd.read_parquet(p) for p in sorted(CACHE_DIR.glob("features_*.parquet"))]
    result = pd.concat(chunks, ignore_index=True)
    result.to_parquet(final_file, index=False)
    print(f"Cached to {final_file}")
    return result

if __name__ == "__main__":
    print("Smoke test: stock_id=0")
    df = build_features(stock_id=0)
    print(f"Shape: {df.shape}")