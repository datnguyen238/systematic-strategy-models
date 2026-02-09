# src/backtest/costs.py
from __future__ import annotations

import pandas as pd


def turnover_series(positions: pd.DataFrame) -> pd.Series:
    return positions.diff().abs().sum(axis=1).fillna(0.0)


def trading_costs_bps(
    positions: pd.DataFrame,
    cost_bps: float,
    trade_mask: pd.Series | None = None,
) -> pd.Series:
    """
    If trade_mask is provided, only charge transaction costs on dates where trade_mask[t] == True.
    """
    if cost_bps < 0:
        raise ValueError("cost_bps must be >= 0")

    turnover = turnover_series(positions)

    if trade_mask is not None:
        trade_mask = trade_mask.reindex(turnover.index).fillna(False)
        turnover = turnover.where(trade_mask, 0.0)

    return (cost_bps / 10000.0) * turnover
