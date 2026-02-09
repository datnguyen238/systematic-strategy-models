# src/experiments/run_tsmom_cost_sensitivity.py
from __future__ import annotations

import pandas as pd

from src.data import DataConfig, fetch_prices, close_prices, compute_returns
from src.strategies.tsmom import TSMOMConfig, generate_positions_tsmom
from src.backtest.engine import backtest_positions
from src.backtest.metrics import summary_metrics


def main() -> None:
    tickers = ["SPY", "TLT", "GLD", "HYG", "EFA", "DBC", "VNQ", "UUP"]
    cfg = DataConfig(tickers=tickers, start="2006-01-01", end="2026-01-01")

    prices = fetch_prices(cfg)
    closes = close_prices(prices)
    rets = compute_returns(closes)

    s_cfg = TSMOMConfig(
        lookback=126,
        vol_window=20,
        target_vol=0.10,
        rebalance="W-FRI",
        max_gross_leverage=1.0,
        mom_threshold=0.10,  # freeze
    )

    pos = generate_positions_tsmom(rets, s_cfg)

    rows = []
    for cost_bps in [0.0, 2.0, 5.0, 10.0, 20.0]:
        res = backtest_positions(pos, rets, cost_bps=cost_bps)
        m = summary_metrics(res)
        rows.append({"cost_bps": cost_bps, **m})

    df = pd.DataFrame(rows)
    df.to_csv("data/processed/tsmom_cost_sensitivity.csv", index=False)
    print(df.to_string(index=False))
    print("\nSaved: data/processed/tsmom_cost_sensitivity.csv")


if __name__ == "__main__":
    main()
