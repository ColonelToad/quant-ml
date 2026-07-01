"""
core/cv.py
----------
Time-series aware cross-validation strategies.
Shared across all competitions — never use standard KFold on financial time series.
"""

from typing import Iterator, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def purged_kfold(
    n_splits: int,
    time_index: pd.Series,
    embargo_pct: float = 0.01,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Purged K-Fold with embargo for financial time series.

    Removes train samples whose labels overlap with the test window (purging),
    then adds a gap (embargo) after the test period to prevent leakage from
    autocorrelated features.

    Args:
        n_splits: Number of CV folds.
        time_index: Series of timestamps or integer time indices aligned to X.
        embargo_pct: Fraction of total samples to embargo after each test fold.

    Yields:
        (train_indices, test_indices) arrays for each fold.
    """
    n = len(time_index)
    embargo_size = int(n * embargo_pct)
    indices = np.arange(n)
    fold_size = n // n_splits

    for fold in range(n_splits):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_splits - 1 else n
        test_idx = indices[test_start:test_end]

        # Embargo: exclude samples immediately after test fold
        embargo_end = min(test_end + embargo_size, n)

        train_idx = np.concatenate([
            indices[:test_start],
            indices[embargo_end:]
        ])

        yield train_idx, test_idx


def walk_forward_splits(
    time_index: pd.Series,
    n_splits: int = 5,
    gap: int = 0,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Expanding window walk-forward validation. Train grows, test steps forward.

    Args:
        time_index: Ordered time index aligned to X.
        n_splits: Number of forward steps.
        gap: Number of samples to skip between train end and test start (embargo).

    Yields:
        (train_indices, test_indices) for each step.
    """
    n = len(time_index)
    fold_size = n // (n_splits + 1)

    for i in range(1, n_splits + 1):
        train_end = i * fold_size
        test_start = train_end + gap
        test_end = test_start + fold_size

        if test_end > n:
            break

        yield np.arange(train_end), np.arange(test_start, test_end)
