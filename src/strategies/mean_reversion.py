from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.features import rolling_vol


@dataclass(frozen=True)
class MeanRevConfig:
    signal_mode: str = "return_z"  # "return_z", "price_ma_z"
    z_window: int = 20
    entry_z: float = 2.0
    exit_z: float = 0.25
    stop_z: float = 4.0
    max_hold: int = 5

    vol_window: int = 20
    target_vol: float = 0.10
    max_gross_leverage: float = 1.0
    min_vol: float = 1e-6

    use_vol_filter: bool = True
    vol_percentile: float = 0.80

    use_trend_filter: bool = True
    trend_window: int = 50
    trend_threshold: float = 0.03


def zscore_returns(returns: pd.DataFrame, window: int, min_std: float = 1e-8) -> pd.DataFrame:
    mu = returns.rolling(window).mean()
    sd = returns.rolling(window).std(ddof=1).clip(lower=min_std)
    return (returns - mu) / sd


def zscore_price_deviation(closes: pd.DataFrame, window: int, min_std: float = 1e-8) -> pd.DataFrame:
    ma = closes.rolling(window).mean()
    dev = closes / ma - 1.0
    mu = dev.rolling(window).mean()
    sd = dev.rolling(window).std(ddof=1).clip(lower=min_std)
    return (dev - mu) / sd


def _build_signal(closes: pd.DataFrame, returns: pd.DataFrame, cfg: MeanRevConfig) -> pd.DataFrame:
    if cfg.signal_mode == "return_z":
        return zscore_returns(returns, window=cfg.z_window)
    if cfg.signal_mode == "price_ma_z":
        return zscore_price_deviation(closes, window=cfg.z_window)
    raise ValueError("signal_mode must be one of {'return_z', 'price_ma_z'}")


def generate_positions_mean_reversion(
    closes: pd.DataFrame,
    returns: pd.DataFrame,
    cfg: MeanRevConfig,
) -> pd.DataFrame:
    """
    Mean reversion:
      - Enter long when z < -entry_z
      - Enter short when z > +entry_z
      - Exit when z reverts near 0, stop_z breached, or max_hold reached
    Sizing:
      - Per-asset vol targeting
      - Gross leverage cap each day
    """
    common_idx = closes.index.intersection(returns.index)
    common_cols = closes.columns.intersection(returns.columns)
    if len(common_idx) < 2 or len(common_cols) == 0:
        raise ValueError("Not enough overlapping data between closes and returns.")

    closes = closes.loc[common_idx, common_cols].sort_index()
    rets = returns.loc[common_idx, common_cols].sort_index()

    z = _build_signal(closes, rets, cfg)

    vol = rolling_vol(rets, window=cfg.vol_window).clip(lower=cfg.min_vol)

    if cfg.use_vol_filter:
        portfolio_vol = vol.mean(axis=1)
        vol_cut = portfolio_vol.quantile(cfg.vol_percentile)
        allow_vol = portfolio_vol <= vol_cut
    else:
        allow_vol = pd.Series(True, index=closes.index)

    if cfg.use_trend_filter:
        trend_ma = closes.rolling(cfg.trend_window).mean()
        trend_strength = (closes / trend_ma - 1.0).abs()
        allow_trend = trend_strength <= cfg.trend_threshold
    else:
        allow_trend = pd.DataFrame(True, index=closes.index, columns=closes.columns)

    target_daily_vol = cfg.target_vol / np.sqrt(252.0)
    vol_scale = (target_daily_vol / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    positions = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)

    for col in closes.columns:
        pos_dir = 0.0
        hold_days = 0

        for dt in closes.index:
            zt = float(z.at[dt, col]) if pd.notna(z.at[dt, col]) else np.nan

            if pos_dir != 0.0:
                hold_days += 1
                exit_hit = (pos_dir > 0.0 and zt >= cfg.exit_z) or (pos_dir < 0.0 and zt <= -cfg.exit_z)
                stop_hit = abs(zt) >= cfg.stop_z if not np.isnan(zt) else False
                time_stop = hold_days >= cfg.max_hold

                if exit_hit or stop_hit or time_stop:
                    pos_dir = 0.0
                    hold_days = 0

            trade_ok = bool(allow_vol.loc[dt]) and bool(allow_trend.at[dt, col])
            if pos_dir == 0.0 and not np.isnan(zt) and trade_ok:
                if zt <= -cfg.entry_z:
                    pos_dir = +1.0
                    hold_days = 0
                elif zt >= cfg.entry_z:
                    pos_dir = -1.0
                    hold_days = 0

            positions.at[dt, col] = pos_dir

    positions = positions * vol_scale

    gross = positions.abs().sum(axis=1)
    gross_cap = (cfg.max_gross_leverage / gross.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
    positions = positions.mul(gross_cap, axis=0).fillna(0.0)

    return positions
