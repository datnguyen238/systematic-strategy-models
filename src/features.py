from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_vol(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Rolling daily volatility estimate (std of log returns).
    """
    vol = returns.rolling(window).std(ddof=1)
    return vol


def rolling_sum(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Rolling sum of log returns.
    """
    return returns.rolling(window).sum()


def zscore_prices(closes: pd.DataFrame, window: int, min_std: float = 1e-8) -> pd.DataFrame:
    """
    z = (P - rolling_mean(P)) / rolling_std(P)
    """
    mu = closes.rolling(window).mean()
    sd = closes.rolling(window).std(ddof=1).clip(lower=min_std)
    z = (closes - mu) / sd
    return z


def resample_rebalance_dates(index: pd.DatetimeIndex, rule: str) -> pd.DatetimeIndex:
    """
    Return rebalance dates according to pandas resample rule, restricted to trading days in index.
    """
    rb = pd.Series(1, index=index).resample(rule).last()
    return rb.index.intersection(index)

