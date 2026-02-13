# src/strategies/mean_reversion.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.features import zscore_prices, rolling_vol


@dataclass(frozen=True)
class MeanRevConfig:
    z_window: int = 20  # How many days to compute z-score over
    entry_z: float = 2.0 # Enter trade if |z| > 2.0σ (2 standard deviations)
    exit_z: float = 0.0   # Exit when z swings back to 0
    max_hold: int = 5 # Max 5 days in trade before forced exit

    vol_window: int = 20 # Volatility estimation window
    target_vol: float = 0.10 # Target 10% annualized portfolio vol
    max_gross_leverage: float = 1.0
    min_vol: float = 1e-6

    use_vol_filter: bool = True

    # Mean reversion performs poorly in high-volatility regimes.
    vol_percentile: float = 0.80  # disable trading when vol is above this percentile


def zscore_returns(returns: pd.DataFrame, window: int, min_std: float = 1e-8) -> pd.DataFrame:
    mu = returns.rolling(window).mean()
    sd = returns.rolling(window).std(ddof=1).clip(lower=min_std)
    return (returns - mu) / sd


def generate_positions_mean_reversion(
    closes: pd.DataFrame,
    returns: pd.DataFrame,
    cfg: MeanRevConfig,
) -> pd.DataFrame:
    """
    Mean reversion on price z-score:
      - Enter long when z < -entry_z
      - Enter short when z > +entry_z
      - Exit when z crosses exit band (default 0) OR holding period exceeds max_hold
    Sizing:
      - Vol targeting per asset: w_i ~ target_daily_vol / est_daily_vol_i
      - Cap gross leverage each day
    """
    # align indices/columns
    common_idx = closes.index.intersection(returns.index)
    common_cols = closes.columns.intersection(returns.columns)
    if len(common_idx) < 2 or len(common_cols) == 0:
        raise ValueError("Not enough overlapping data between closes and returns.")

    closes = closes.loc[common_idx, common_cols].sort_index()
    rets = returns.loc[common_idx, common_cols].sort_index()

    z = zscore_returns(rets, window=cfg.z_window)

    vol = rolling_vol(rets, window=cfg.vol_window).clip(lower=cfg.min_vol)

    # Vol regime filter: disable entries on high-vol days (proxy for "expectations changed")
    if cfg.use_vol_filter:
        # portfolio-level vol proxy = average vol across assets
        port_vol = vol.mean(axis=1)
        threshold = port_vol.quantile(cfg.vol_percentile)
        allow_trade = port_vol <= threshold
    else:
        allow_trade = pd.Series(True, index=closes.index)


    target_daily_vol = cfg.target_vol / np.sqrt(252.0)
    scale = (target_daily_vol / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    positions = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)

    for col in closes.columns:
        pos_dir = 0.0
        hold = 0

        for dt in closes.index:
            zt = float(z.at[dt, col]) if pd.notna(z.at[dt, col]) else np.nan

            # Manage existing position
            if pos_dir != 0.0:
                hold += 1

                # Exit conditions
                exit_hit = (pos_dir > 0.0 and zt >= cfg.exit_z) or (pos_dir < 0.0 and zt <= -cfg.exit_z)
                time_stop = hold >= cfg.max_hold

                if exit_hit or time_stop:
                    pos_dir = 0.0
                    hold = 0

            # Enter if flat
            if pos_dir == 0.0 and not np.isnan(zt) and bool(allow_trade.loc[dt]):
                if zt <= -cfg.entry_z:
                    pos_dir = +1.0
                    hold = 0
                elif zt >= cfg.entry_z:
                    pos_dir = -1.0
                    hold = 0

            positions.at[dt, col] = pos_dir

    # apply vol targeting
    positions = positions * scale

    # cap gross leverage
    gross = positions.abs().sum(axis=1)
    scale_cap = (cfg.max_gross_leverage / gross).clip(upper=1.0).fillna(1.0)
    positions = positions.mul(scale_cap, axis=0).fillna(0.0)

    return positions
