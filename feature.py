# Run this in your Jupyter Notebook or final training script
import pandas as pd
from pathlib import Path

cache_dir = Path('/home/tolugenius/Projects/quant-ml/data/optiver-volatility-pred/feature_cache')

print("Loading cached chunks...")
# Load all parquet files into a list
chunks = [pd.read_parquet(p) for p in sorted(cache_dir.glob("features_*.parquet"))]

print("Combining...")
result = pd.concat(chunks, ignore_index=True)

print(f"\nShape: {result.shape}")
print(f"\nNull counts:\n{result.isnull().sum()}")