# src/experiments/smoke_test_tsmom.py
from __future__ import annotations

from src.data import DataConfig, fetch_prices, close_prices, compute_returns
from src.backtest.engine import backtest_positions
from src.backtest.metrics import summary_metrics
from src.strategies.tsmom import TSMOMConfig, generate_positions_tsmom, rebalance_mask

# turnover on trade days only
from src.backtest.costs import turnover_series




def main() -> None:
    tickers = ["SPY", "TLT", "GLD", "HYG", "EFA", "DBC", "VNQ", "UUP"]
    cfg = DataConfig(tickers=tickers, start="2006-01-01", end="2026-01-01")

    prices = fetch_prices(cfg)
    closes = close_prices(prices)
    rets = compute_returns(closes)

    for thr in [0.00, 0.05, 0.10]:
        s_cfg = TSMOMConfig(
            lookback=126,
            vol_window=20,
            target_vol=0.10,
            rebalance="W-FRI",
            max_gross_leverage=1.0,
            mom_threshold=thr,
        )

        positions = generate_positions_tsmom(rets, s_cfg)
        mask = rebalance_mask(rets.index, s_cfg.rebalance)

        turn = turnover_series(positions)
        trade_turn = turn[mask]

        print(f"\n\n===== TSMOM threshold={thr:.2f} =====")
        print("--- Turnover diagnostics ---")
        print("Total days:", len(turn))
        print("Trade days:", int(mask.sum()))
        print("Trade-day fraction:", float(mask.mean()))
        print("Avg turnover (all days):", float(turn.mean()))
        print("Avg turnover (trade days):", float(trade_turn.mean()))
        print("Turnover quantiles (trade days):")
        print(trade_turn.quantile([0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]).to_string())

        for cost_bps in [0.0, 5.0, 10.0, 20.0]:
            res = backtest_positions(positions, rets, cost_bps=cost_bps, trade_mask=mask)
            m = summary_metrics(res)
            print(f"\n=== cost_bps={cost_bps} ===")
            for k, v in m.items():
                print(f"{k:>12}: {v:.4f}")


if __name__ == "__main__":
    main()
