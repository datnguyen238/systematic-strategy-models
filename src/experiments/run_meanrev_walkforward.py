# src/experiments/run_meanrev_walkforward.py
from __future__ import annotations

from src.data import DataConfig, fetch_prices, close_prices, compute_returns
from src.strategies.mean_reversion import MeanRevConfig, generate_positions_mean_reversion
from src.experiments.walkforward import WalkForwardConfig, run_walkforward, summarize_walkforward


def main() -> None:
    tickers = ["SPY", "TLT", "GLD", "HYG", "EFA", "DBC", "VNQ", "UUP"]
    cfg = DataConfig(tickers=tickers, start="2006-01-01", end="2026-01-01")

    prices = fetch_prices(cfg)
    closes = close_prices(prices)
    rets = compute_returns(closes)

    # Freeze a single baseline MR config for reporting (return-shock MR + vol filter)
    s_cfg = MeanRevConfig(
        z_window=10,
        entry_z=2.0,
        exit_z=0.0,
        max_hold=5,
        use_vol_filter=True,
        vol_percentile=0.80,
        vol_window=20,
        target_vol=0.10,
        max_gross_leverage=1.0,
    )

    pos = generate_positions_mean_reversion(closes, rets, s_cfg)

    wf_cfg = WalkForwardConfig(train_years=3, test_years=1, cost_bps=10.0)
    df = run_walkforward(pos, rets, wf_cfg)

    print(df.to_string(index=False))
    df.to_csv("data/processed/meanrev_walkforward.csv", index=False)
    print("\nSaved: data/processed/meanrev_walkforward.csv")


    print("\n--- Walk-forward summary ---")
    summary = summarize_walkforward(df)
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
