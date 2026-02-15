# src/experiments/run_tsmom_cost_sensitivity.py
from __future__ import annotations

import pandas as pd

from src.data import DataConfig, fetch_prices, close_prices, compute_returns
from src.strategies.tsmom import TSMOMConfig, generate_positions_tsmom, rebalance_mask
from src.backtest.engine import backtest_positions
from src.backtest.metrics import summary_metrics


def main() -> None:
    tickers = ["SPY", "TLT", "GLD", "HYG", "EFA", "DBC", "VNQ", "UUP"]
    cfg = DataConfig(tickers=tickers, start="2006-01-01", end="2026-01-01")

    prices = fetch_prices(cfg)
    closes = close_prices(prices)
    rets = compute_returns(closes)

    s_cfg = TSMOMConfig(
        signal_mode="monthly_12_1",
        lookback=12,
        skip_recent=1,
        signal_lag=1,
        hold_rebalances=1,
        vol_window=63,
        use_ewm_vol=True,
        ewma_decay=0.94,
        target_vol=0.10,
        rebalance="ME",
        max_gross_leverage=1.0,
        mom_threshold=0.00,
        equal_weight_active=True,
    )

    pos = generate_positions_tsmom(rets, s_cfg)
    mask = rebalance_mask(rets.index, s_cfg.rebalance)

    rows = []
    for cost_bps in [0.0, 2.0, 5.0, 10.0, 20.0]:
        res = backtest_positions(pos, rets, cost_bps=cost_bps, trade_mask=mask)
        m = summary_metrics(res)
        rows.append({"cost_bps": cost_bps, **m})

    df = pd.DataFrame(rows)
    df.to_csv("data/processed/tsmom_cost_sensitivity.csv", index=False)
    print(df.to_string(index=False))
    print("\nSaved: data/processed/tsmom_cost_sensitivity.csv")


if __name__ == "__main__":
    main()
