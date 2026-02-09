# src/experiments/smoke_test_data.py
from __future__ import annotations

from src.data import DataConfig, fetch_prices, close_prices, compute_returns, outlier_report


def main() -> None:
    tickers = ["SPY", "TLT", "GLD", "HYG", "EFA", "DBC", "VNQ", "UUP"]
    cfg = DataConfig(
        tickers=tickers,
        start="2006-01-01",
        end="2026-01-01",
        interval="1d",
        auto_adjust=True,
    )

    prices = fetch_prices(cfg)
    closes = close_prices(prices)
    rets = compute_returns(closes)

    print("Closes shape:", closes.shape)
    print("Returns shape:", rets.shape)
    print("Date range:", rets.index.min().date(), "->", rets.index.max().date())

    rep = outlier_report(rets, z_thresh=12.0)
    print("Outlier count:", len(rep))
    if len(rep) > 0:
        print(rep.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
