# src/experiments/walkforward.py
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from src.backtest.engine import backtest_positions
from src.backtest.metrics import summary_metrics


@dataclass(frozen=True)
class WalkForwardConfig:
    train_years: int = 3
    test_years: int = 1
    cost_bps: float = 10.0
    min_test_days: int = 200


def _year_slices(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    years = sorted(set(index.year))
    return [pd.Timestamp(year=y, month=1, day=1) for y in years]


def run_walkforward(
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    cfg: WalkForwardConfig,
) -> pd.DataFrame:
    """
    Walk-forward evaluation for a precomputed positions/returns set.
    This is stability testing across time, not parameter fitting.
    """
    idx = returns.index.intersection(positions.index)
    positions = positions.loc[idx]
    returns = returns.loc[idx]

    years = sorted(set(idx.year))
    rows = []

    for start_year in years:
        train_start = pd.Timestamp(start_year, 1, 1)
        train_end = pd.Timestamp(start_year + cfg.train_years, 1, 1)
        test_end = pd.Timestamp(start_year + cfg.train_years + cfg.test_years, 1, 1)

        test_idx = idx[(idx >= train_end) & (idx < test_end)]
        if len(test_idx) < cfg.min_test_days:
            continue

        pos_test = positions.loc[test_idx]
        ret_test = returns.loc[test_idx]

        res = backtest_positions(pos_test, ret_test, cost_bps=cfg.cost_bps)
        m = summary_metrics(res)

        avg_gross = float(res.positions.abs().sum(axis=1).mean())
        pct_invested = float((res.positions.abs().sum(axis=1) > 1e-6).mean())


        rows.append(
            {
                "train_start": train_start.date(),
                "train_end": (train_end - pd.Timedelta(days=1)).date(),
                "test_start": test_idx.min().date(),
                "test_end": test_idx.max().date(),
                "Sharpe": m["Sharpe"],
                "CAGR": m["CAGR"],
                "MaxDD": m["MaxDD"],
                "AvgTurnover": m["AvgTurnover"],
                "TotalCost": m["TotalCost"],
                "AvgGrossExposure": avg_gross,
                "PctInvested": pct_invested,
            }
        )

    return pd.DataFrame(rows)

def summarize_walkforward(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_folds": 0}
    
    sharpe_defined = df["Sharpe"].notna()
    no_trade = (df["AvgTurnover"] == 0) & (df["TotalCost"] == 0) & (df["CAGR"] == 0)

    return {
        "n_folds": int(len(df)),
        "Sharpe_mean": float(df["Sharpe"].mean()),
        "Sharpe_median": float(df["Sharpe"].median()),
        "Sharpe_pct_positive": float((df["Sharpe"] > 0).mean()),
        "CAGR_mean": float(df["CAGR"].mean()),
        "MaxDD_worst": float(df["MaxDD"].min()),
        "AvgTurnover_mean": float(df["AvgTurnover"].mean()),
        "TotalCost_sum": float(df["TotalCost"].sum()),
        "Sharpe_defined_folds": int(sharpe_defined.sum()),
        "NoTrade_folds": int(no_trade.sum()),
    }
