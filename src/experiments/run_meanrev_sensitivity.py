# src/experiments/run_meanrev_sensitivity.py
from __future__ import annotations

from src.data import DataConfig, fetch_prices, close_prices, compute_returns
from src.experiments.sensitivity import meanrev_sensitivity


def main() -> None:
    tickers = ["SPY", "TLT", "GLD", "HYG", "EFA", "DBC", "VNQ", "UUP"]
    cfg = DataConfig(tickers=tickers, start="2006-01-01", end="2026-01-01")

    prices = fetch_prices(cfg)
    closes = close_prices(prices)
    rets = compute_returns(closes)

    df = meanrev_sensitivity(closes, rets, cost_bps=10.0)

    print(df.head(15).to_string(index=False))

    # save for notebook/README
    df.to_csv("data/processed/meanrev_sensitivity.csv", index=False)
    print("\nSaved: data/processed/meanrev_sensitivity.csv")


if __name__ == "__main__":
    main()
