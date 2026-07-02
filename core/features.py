"""
core/features.py
----------------
General-purpose feature engineering utilities.
Neutralization is Numerai-specific but useful for any cross-sectional strategy.
"""

import numpy as np
import pandas as pd

def zscore(series: pd.Series, window: int = None) -> pd.Series:
    """
    Z-score normalize a series. If window is provided, rolling z-score.
    Always apply cross-sectionally (group by time) for market prediction tasks.
    """
    if window:
        mu = series.rolling(window).mean()
        sigma = series.rolling(window).std()
    else:
        mu = series.mean()
        sigma = series.std()
    return (series - mu) / (sigma + 1e-8)


def rank_normalize(series: pd.Series) -> pd.Series:
    """Rank then scale to [0, 1]. Robust to outliers; used heavily in Numerai."""
    return series.rank(pct=True)


def neutralize(
    predictions: pd.Series,
    features: pd.DataFrame,
    proportion: float = 1.0,
) -> pd.Series:
    """
    Neutralize predictions against a set of features using linear projection.
    Removes the component of predictions explained by the given features.
    Standard practice in Numerai to reduce factor exposure.

    Args:
        predictions: Raw model predictions (should be rank-normalized first).
        features: DataFrame of features to neutralize against.
        proportion: How much to neutralize (1.0 = full, 0.5 = half).

    Returns:
        Neutralized predictions.
    """
    scores = predictions.values.reshape(-1, 1)
    feats = features.values
    feats = np.hstack([feats, np.ones((feats.shape[0], 1))])  # add intercept
    exposures = feats @ np.linalg.pinv(feats) @ scores
    neutralized = scores - proportion * exposures
    return pd.Series(neutralized.flatten(), index=predictions.index)


def lag_features(df: pd.DataFrame, cols: list, lags: list) -> pd.DataFrame:
    """
    Add lagged versions of specified columns. Groups by stock/id if 'stock_id'
    or 'symbol' present; otherwise applies globally.
    """
    group_col = next((c for c in ['stock_id', 'symbol', 'asset_id'] if c in df.columns), None)
    out = df.copy()
    for col in cols:
        for lag in lags:
            new_col = f"{col}_lag{lag}"
            if group_col:
                out[new_col] = out.groupby(group_col)[col].shift(lag)
            else:
                out[new_col] = out[col].shift(lag)
    return out
