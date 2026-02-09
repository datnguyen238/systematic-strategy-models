# src/backtest/metrics.py
from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_annualized(log_returns: pd.Series, periods_per_year: int = 252) -> float:
    r = log_returns.dropna()
    if len(r) < 2:
        return float("nan")
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float((mu / sd) * np.sqrt(periods_per_year))


def cagr_from_log_returns(log_returns: pd.Series, periods_per_year: int = 252) -> float:
    r = log_returns.dropna()
    if len(r) == 0:
        return float("nan")
    total_log = r.sum()
    years = len(r) / periods_per_year
    if years <= 0:
        return float("nan")
    return float(np.exp(total_log) ** (1 / years) - 1)


def max_drawdown_from_log_equity(equity_log: pd.Series) -> float:
    """
    equity_log is cumulative log return.
    Convert to equity curve in price space: exp(equity_log)
    """
    eq = np.exp(equity_log.dropna())
    if len(eq) == 0:
        return float("nan")
    running_max = eq.cummax()
    dd = (eq / running_max) - 1.0
    return float(dd.min())


def summary_metrics(result, periods_per_year: int = 252) -> dict:
    r = result.returns
    return {
        "Sharpe": sharpe_annualized(r, periods_per_year),
        "CAGR": cagr_from_log_returns(r, periods_per_year),
        "MaxDD": max_drawdown_from_log_equity(result.equity),
        "AvgTurnover": float(result.turnover.mean()),
        "TotalCost": float(result.costs.sum()),
        "AvgGrossExposure": float(result.gross_exposure.mean()),
        "PctInvested": float((result.gross_exposure > 1e-6).mean()),

    }
