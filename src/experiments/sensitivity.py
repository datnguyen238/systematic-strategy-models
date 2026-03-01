# src/experiments/sensitivity.py
from __future__ import annotations

import itertools
import pandas as pd

from src.backtest.engine import backtest_positions
from src.backtest.metrics import summary_metrics
from src.strategies.mean_reversion import MeanRevConfig, generate_positions_mean_reversion


def meanrev_sensitivity(
    closes: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: float = 10.0,
    cost_grid: list[float] | None = None,
    grids: dict | None = None,
) -> pd.DataFrame:
    if grids is None:
        grids = {
            "signal_mode": ["return_z", "price_ma_z"],
            "z_window": [20, 40, 60],
            "entry_z": [2.0, 2.5, 3.0, 4.0],
            "max_hold": [3, 5, 10],
        }
    if cost_grid is None:
        cost_grid = [cost_bps]

    keys = list(grids.keys())
    combos = list(itertools.product(*[grids[k] for k in keys]))

    rows = []
    for vals in combos:
        params = dict(zip(keys, vals))

        cfg = MeanRevConfig(
            signal_mode=str(params["signal_mode"]),
            z_window=int(params["z_window"]),
            entry_z=float(params["entry_z"]),
            exit_z=0.25,
            stop_z=max(4.0, float(params["entry_z"]) + 1.0),
            max_hold=int(params["max_hold"]),
            use_vol_filter=True,
            vol_percentile=0.80,
            use_trend_filter=True,
            trend_window=50,
            trend_threshold=0.03,
            vol_window=20,
            target_vol=0.10,
            max_gross_leverage=1.0,
        )

        pos = generate_positions_mean_reversion(closes, returns, cfg)
        if pos is None:
            raise RuntimeError("generate_positions_mean_reversion returned None — mean_reversion.py is not returning positions.")

        for cbps in cost_grid:
            res = backtest_positions(pos, returns, cost_bps=float(cbps))
            m = summary_metrics(res)

            rows.append(
                {
                    **params,
                    "cost_bps": float(cbps),
                    "Sharpe": m["Sharpe"],
                    "CAGR": m["CAGR"],
                    "MaxDD": m["MaxDD"],
                    "AvgTurnover": m["AvgTurnover"],
                    "TotalCost": m["TotalCost"],
                    "HitRate": m["HitRate"],
                    "AvgWin": m["AvgWin"],
                    "AvgLoss": m["AvgLoss"],
                    "Payoff": m["Payoff"],
                    "AvgHoldDays": m["AvgHoldDays"],
                }
            )

    df = pd.DataFrame(rows).sort_values(["Sharpe", "CAGR"], ascending=False).reset_index(drop=True)
    
    df["SharpeRank"] = df["Sharpe"].rank(ascending=False, method="min")
    df["CAGRank"] = df["CAGR"].rank(ascending=False, method="min")
    df["TurnoverRank"] = df["AvgTurnover"].rank(ascending=True, method="min")

    return df
