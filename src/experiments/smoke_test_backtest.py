# src/experiments/smoke_test_backtest.py
from __future__ import annotations

import pandas as pd

from src.data import DataConfig, fetch_prices, close_prices, compute_returns
from src.backtest.engine import backtest_positions
from src.backtest.metrics import summary_metrics


def equal_weight_buy_and_hold(returns: pd.DataFrame) -> pd.DataFrame:
    # Fully invested, equal-weight, no leverage
    n = returns.shape[1]
    w = pd.DataFrame(1.0 / n, index=returns.index, columns=returns.columns)
    return w


def main() -> None:
    tickers = ["SPY", "TLT", "GLD", "HYG", "EFA", "DBC", "VNQ", "UUP"]
    cfg = DataConfig(tickers=tickers, start="2006-01-01", end="2026-01-01")

    prices = fetch_prices(cfg)
    closes = close_prices(prices)
    rets = compute_returns(closes)

    positions = equal_weight_buy_and_hold(rets)

    for cost_bps in [0.0, 5.0, 10.0, 20.0]:
        res = backtest_positions(positions, rets, cost_bps=cost_bps)
        m = summary_metrics(res)
        print(f"\n=== Buy&Hold EW | cost_bps={cost_bps} ===")
        for k, v in m.items():
            print(f"{k:>12}: {v:.4f}")


if __name__ == "__main__":
    main()
