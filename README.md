# Systematic Strategy Models

## Project Summary
This is a small quant research project where I tested two trading ideas on a basket of ETFs:
- Mean Reversion: bet that large short-term moves may bounce back.
- Time-Series Momentum (TSMOM): bet that medium-term trends may continue.

The main goal was to build a clean backtesting pipeline and compare these two ideas under realistic transaction costs.

## What I Built
- Data pipeline to download/cache ETF prices.
- Feature engineering for returns, volatility, and signals.
- Strategy modules for Mean Reversion and TSMOM.
- Backtest engine with:
  - no look-ahead position handling
  - turnover-based trading costs
  - common performance metrics (Sharpe, CAGR, drawdown)
- Experiment scripts for sensitivity tests and walk-forward evaluation.

## Simple Result (Plain English)
- Mean Reversion did not hold up well in this ETF universe.
- TSMOM looked better before costs, but performance dropped a lot when costs increased.
- Takeaway: strategy quality and trading frictions matter as much as raw signal ideas.

## Project Structure
- `src/data.py`: data download, cache, close prices, returns.
- `src/features.py`: rolling volatility and helper features.
- `src/strategies/mean_reversion.py`: mean reversion signal + position logic.
- `src/strategies/tsmom.py`: momentum signal + position logic.
- `src/backtest/engine.py`: converts positions + returns into PnL.
- `src/backtest/costs.py`: turnover and transaction cost model.
- `src/backtest/metrics.py`: performance statistics.
- `src/experiments/`: runnable scripts for tests and reports.

## How To Run
```bash
./venv/bin/python -m src.experiments.smoke_test_data
./venv/bin/python -m src.experiments.run_meanrev_sensitivity
./venv/bin/python -m src.experiments.run_meanrev_walkforward
./venv/bin/python -m src.experiments.run_tsmom_cost_sensitivity
```

Outputs are saved in `data/processed/`.

## Paper Reference
This project is inspired by:
- Moskowitz, Ooi, Pedersen (2012), *Time Series Momentum*  
  DOI: `10.1016/j.jfineco.2011.11.003`

## Scope Note
This is an educational/research build using ETFs, not a production trading system.
