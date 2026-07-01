"""
core/eval.py
------------
Evaluation metrics shared across competitions and Numerai.
Financial ML cares about rank correlation and risk-adjusted returns, not just MSE.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def information_coefficient(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Spearman rank correlation between predictions and targets.
    The primary signal quality metric for cross-sectional strategies (Numerai, Ubiquant).
    IC > 0.05 is generally considered meaningful; IC > 0.10 is strong.
    """
    ic, _ = spearmanr(y_true, y_pred)
    return ic


def icir(ics: pd.Series) -> float:
    """
    IC Information Ratio: mean(IC) / std(IC).
    Measures consistency of the signal across time — more important than raw IC.
    """
    return ics.mean() / ics.std() if ics.std() > 0 else 0.0


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio from a returns series."""
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Percentage Error.
    Primary metric for Optiver Realized Volatility Prediction.
    """
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown of a cumulative return series."""
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    return drawdown.min()
