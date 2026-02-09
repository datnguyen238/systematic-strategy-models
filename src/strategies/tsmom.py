# src/strategies/tsmom.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TSMOMConfig:
    lookback: int = 126          # ~6 months
    vol_window: int = 20         # 1 month
    target_vol: float = 0.10     # 10% annualized target vol per asset sleeve
    rebalance: str = "W-FRI"     # weekly rebalance (Friday)
    max_gross_leverage: float = 1.0  # cap sum |weights|
    min_vol: float = 1e-6        # numerical floor
    mom_threshold: float = 0.00  # in log-return units of trailing sum; start 0.00 then test 0.05, 0.10


def generate_positions_tsmom(
    asset_returns: pd.DataFrame,
    cfg: TSMOMConfig,
) -> pd.DataFrame:
    """
    Time-Series Momentum:
      signal_i(t) = sign(sum_{j=1..lookback} r_i(t-j))
      raw_weight_i(t) = signal_i(t) * (target_daily_vol / est_daily_vol_i(t))
    Then apply:
      - rebalance schedule (positions held constant between rebalances)
      - gross leverage cap (sum |w_i| <= max_gross_leverage)
    Returns weights by date and asset.
    """
    rets = asset_returns.copy().sort_index()

    # Momentum signal: trailing sum of log returns
    mom = rets.rolling(cfg.lookback).sum()
    
    thr = cfg.mom_threshold
    signal = (mom > thr).astype(float) - (mom < -thr).astype(float)
    signal = signal.fillna(0.0)


    # Vol estimate: rolling std of daily log returns
    vol = rets.rolling(cfg.vol_window).std(ddof=1)
    vol = vol.clip(lower=cfg.min_vol).fillna(np.nan)

    # Convert annual target vol to daily target vol
    target_daily_vol = cfg.target_vol / np.sqrt(252.0)

    raw = signal * (target_daily_vol / vol)
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Rebalance only on schedule (e.g., weekly Fridays)
    # 1) create rebalance dates index
    rb = pd.Series(1, index=raw.index).resample(cfg.rebalance).last()
    rb_dates = rb.index.intersection(raw.index)

    positions = pd.DataFrame(0.0, index=raw.index, columns=raw.columns)
    if len(rb_dates) == 0:
        return positions

    # Set weights on rebalance dates; forward-fill until next rebalance
    positions.loc[rb_dates] = raw.loc[rb_dates]
    positions = positions.ffill().fillna(0.0)

    # Gross leverage cap each day: sum |w_i| <= max_gross_leverage
    gross = positions.abs().sum(axis=1)
    scale = (cfg.max_gross_leverage / gross).clip(upper=1.0)
    positions = positions.mul(scale, axis=0)

    return positions


def rebalance_mask(index: pd.DatetimeIndex, rule: str) -> pd.Series:
    """
    Boolean Series indexed by trading days: True on rebalance days that exist in index.
    """
    rb = pd.Series(1, index=index).resample(rule).last()
    rb_dates = rb.index.intersection(index)
    mask = pd.Series(False, index=index)
    mask.loc[rb_dates] = True
    return mask
