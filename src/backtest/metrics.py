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


def trade_stats(returns: pd.Series, positions: pd.DataFrame) -> dict:
    """
    Episode-level stats using contiguous invested periods.
    """
    gross = positions.abs().sum(axis=1)
    active = gross > 1e-8
    starts = active & (~active.shift(1, fill_value=False))
    ends = active & (~active.shift(-1, fill_value=False))

    start_idx = list(starts[starts].index)
    end_idx = list(ends[ends].index)
    if not start_idx or not end_idx:
        return {
            "HitRate": float("nan"),
            "AvgWin": float("nan"),
            "AvgLoss": float("nan"),
            "Payoff": float("nan"),
            "AvgHoldDays": float("nan"),
            "NumTrades": 0.0,
        }

    n = min(len(start_idx), len(end_idx))
    trade_pnl: list[float] = []
    hold_days: list[int] = []

    for i in range(n):
        s = start_idx[i]
        e = end_idx[i]
        if e < s:
            continue
        r = returns.loc[s:e]
        trade_pnl.append(float(r.sum()))
        hold_days.append(int(len(r)))

    if len(trade_pnl) == 0:
        return {
            "HitRate": float("nan"),
            "AvgWin": float("nan"),
            "AvgLoss": float("nan"),
            "Payoff": float("nan"),
            "AvgHoldDays": float("nan"),
            "NumTrades": 0.0,
        }

    pnl = pd.Series(trade_pnl, dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    avg_win = float(wins.mean()) if len(wins) else float("nan")
    avg_loss = float(losses.mean()) if len(losses) else float("nan")
    payoff = float(avg_win / abs(avg_loss)) if np.isfinite(avg_win) and np.isfinite(avg_loss) and avg_loss != 0 else float("nan")

    return {
        "HitRate": float((pnl > 0).mean()),
        "AvgWin": avg_win,
        "AvgLoss": avg_loss,
        "Payoff": payoff,
        "AvgHoldDays": float(np.mean(hold_days)),
        "NumTrades": float(len(pnl)),
    }


def summary_metrics(result, periods_per_year: int = 252) -> dict:
    r = result.returns
    out = {
        "Sharpe": sharpe_annualized(r, periods_per_year),
        "CAGR": cagr_from_log_returns(r, periods_per_year),
        "MaxDD": max_drawdown_from_log_equity(result.equity),
        "AvgTurnover": float(result.turnover.mean()),
        "TotalCost": float(result.costs.sum()),
        "AvgGrossExposure": float(result.gross_exposure.mean()),
        "PctInvested": float((result.gross_exposure > 1e-6).mean()),

    }
    out.update(trade_stats(result.returns, result.positions))
    return out
