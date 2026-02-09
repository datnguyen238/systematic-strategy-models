# src/backtest/engine.py
from __future__ import annotations

from dataclasses import dataclass
from operator import pos
import pandas as pd

from .costs import trading_costs_bps, turnover_series


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    gross_returns: pd.Series
    costs: pd.Series
    turnover: pd.Series
    positions: pd.DataFrame
    gross_exposure: pd.Series



def backtest_positions(
    positions: pd.DataFrame,
    asset_returns: pd.DataFrame,
    cost_bps: float = 0.0,
    trade_mask: pd.Series | None = None,
) -> BacktestResult:

    """
    positions: weights by date (rows) and asset (cols). Weights can be long/short.
    asset_returns: log returns by date and asset.
    Assumption: positions at t-1 are held during day t (enter at close, earn next day's close-to-close).
    """
    # Align indices/columns
    common_idx = positions.index.intersection(asset_returns.index)
    common_cols = positions.columns.intersection(asset_returns.columns)

    if len(common_idx) < 2 or len(common_cols) == 0:
        raise ValueError("Not enough overlapping data between positions and returns.")

    pos = positions.loc[common_idx, common_cols].copy()
    gross_exposure = pos.abs().sum(axis=1).rename("gross_exposure")

    rets = asset_returns.loc[common_idx, common_cols].copy()

    # No look-ahead: shift positions by 1 day
    pos_held = pos.shift(1).fillna(0.0)

    gross = (pos_held * rets).sum(axis=1)

    costs = trading_costs_bps(pos, cost_bps=cost_bps, trade_mask=trade_mask)
    costs = costs.reindex(common_idx).fillna(0.0)

    net = gross - costs

    equity = (net.fillna(0.0)).cumsum().apply(lambda x: float(x))
    equity.name = "equity_log"

    return BacktestResult(
        equity=equity,
        returns=net.rename("net_log_return"),
        gross_returns=gross.rename("gross_log_return"),
        costs=costs.rename("cost"),
        turnover=turnover_series(pos).rename("turnover"),
        positions=pos,
        gross_exposure=gross_exposure,
    )

