# Systematic Strategy Models

This repository evaluates two systematic strategies on a diversified ETF universe:
- Mean Reversion (`src/strategies/mean_reversion.py`)
- Time-Series Momentum (TSMOM) (`src/strategies/tsmom.py`)

## Paper Information

The current `TSMOM` implementation is configured as a research-style "paper baseline" aligned with:

1. Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012). *Time series momentum*. Journal of Financial Economics, 104(2), 228-250.  
   DOI: `10.1016/j.jfineco.2011.11.003`

## How Paper Concepts Map to Code

In `src/strategies/tsmom.py`:

- Monthly `12-1` style signal:
  - `signal_mode="monthly_12_1"`
  - `lookback=12`
  - `skip_recent=1`
- Lagged information set (no look-ahead):
  - `signal_lag=1`
  - volatility estimate shifted by lag before sizing
- Volatility scaling / risk targeting:
  - `target_vol=0.10`
  - `use_ewm_vol=True`
  - `ewma_decay=0.94`
- Month-end rebalance:
  - `rebalance="ME"`
- Overlapping portfolio construction (Jegadeesh-Titman style):
  - `hold_rebalances > 1` enables overlap averaging

## Important Note

This code is an ETF-based research approximation inspired by the papers above, not a full exact replication of every paper dataset, instrument set, or portfolio construction detail.
