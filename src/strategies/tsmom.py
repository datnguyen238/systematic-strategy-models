# src/strategies/tsmom.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TSMOMConfig:
    signal_mode: str = "monthly_12_1"  # "monthly_12_1" (paper baseline) or "daily"
    lookback: int = 12                  # months in monthly mode, days in daily mode
    skip_recent: int = 1                # months in monthly mode, days in daily mode
    signal_lag: int = 1                 # lag to avoid look-ahead (t-1 information set)
    hold_rebalances: int = 1            # h-month overlapping portfolios when >1

    vol_window: int = 63                # daily observations for rolling vol fallback
    use_ewm_vol: bool = True
    ewma_decay: float = 0.94            # RiskMetrics-like daily EWMA decay
    annualization: int = 261

    target_vol: float = 0.10            # annualized target vol per sleeve
    rebalance: str = "ME"              # month-end rebalance for paper baseline
    max_gross_leverage: float = 1.0
    min_vol: float = 1e-6
    mom_threshold: float = 0.00
    equal_weight_active: bool = True


def generate_positions_tsmom(
    asset_returns: pd.DataFrame,
    cfg: TSMOMConfig,
) -> pd.DataFrame:
    """
    Time-Series Momentum:
      signal_i(t) = sign(trailing return over lookback horizon)
      raw_weight_i(t) = signal_i(t) * (target_daily_vol / vol_{i,t-1})
    Then apply:
      - rebalance schedule (positions held constant between rebalances)
      - optional overlapping portfolios when hold_rebalances > 1
      - optional equal-weight averaging across active instruments
      - gross leverage cap (sum |w_i| <= max_gross_leverage)
    Returns weights by date and asset.
    """
    ret_df = asset_returns.copy().sort_index()
    if ret_df.empty:
        return pd.DataFrame(index=ret_df.index, columns=ret_df.columns).fillna(0.0)

    _validate_config(cfg)

    signal_df = _compute_signal(ret_df, cfg)

    # t-1 volatility estimate to avoid look-ahead bias.
    vol_df = _estimate_daily_vol(ret_df, cfg).shift(cfg.signal_lag)
    vol_df = vol_df.clip(lower=cfg.min_vol).fillna(np.nan)

    target_daily_vol = cfg.target_vol / np.sqrt(float(cfg.annualization))

    raw_wgt_df = signal_df * (target_daily_vol / vol_df)
    raw_wgt_df = raw_wgt_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Diversified factor construction: average equally across active instruments.
    if cfg.equal_weight_active:
        active_flag_df = (raw_wgt_df.abs() > 0).astype(float)
        active_count = active_flag_df.sum(axis=1).replace(0.0, np.nan)
        raw_wgt_df = raw_wgt_df.div(active_count, axis=0).fillna(0.0)

    rebalance_day_mask = rebalance_mask(raw_wgt_df.index, cfg.rebalance)
    rebalance_dates = rebalance_day_mask.index[rebalance_day_mask]

    pos_df = pd.DataFrame(0.0, index=raw_wgt_df.index, columns=raw_wgt_df.columns)
    if len(rebalance_dates) == 0:
        return pos_df

    rebalance_wgt_df = raw_wgt_df.loc[rebalance_dates]

    # h-month overlapping portfolios (Jegadeesh-Titman style averaging).
    if cfg.hold_rebalances > 1:
        overlap_wgt_sum = rebalance_wgt_df.copy() * 0.0
        for lag in range(cfg.hold_rebalances):
            overlap_wgt_sum = overlap_wgt_sum + rebalance_wgt_df.shift(lag).fillna(0.0)
        rebalance_wgt_df = overlap_wgt_sum / float(cfg.hold_rebalances)

    pos_df.loc[rebalance_dates] = rebalance_wgt_df
    pos_df = pos_df.ffill().fillna(0.0)

    gross_exposure = pos_df.abs().sum(axis=1)
    gross_scale = (cfg.max_gross_leverage / gross_exposure.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
    pos_df = pos_df.mul(gross_scale, axis=0)

    return pos_df


def rebalance_mask(index: pd.DatetimeIndex, rule: str) -> pd.Series:
    """
    Boolean Series indexed by trading days: True on rebalance days that exist in index.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex")
    if len(index) == 0:
        return pd.Series(dtype=bool, index=index)

    idx_marker = pd.Series(np.arange(len(index)), index=index)
    rebalance_dates = idx_marker.groupby(pd.Grouper(freq=rule)).tail(1).index
    rebalance_day_mask = pd.Series(False, index=index)
    rebalance_day_mask.loc[rebalance_dates] = True
    return rebalance_day_mask


def _validate_config(cfg: TSMOMConfig) -> None:
    if cfg.lookback <= 0:
        raise ValueError("lookback must be > 0")
    if cfg.skip_recent < 0:
        raise ValueError("skip_recent must be >= 0")
    if cfg.signal_lag < 0:
        raise ValueError("signal_lag must be >= 0")
    if cfg.hold_rebalances <= 0:
        raise ValueError("hold_rebalances must be > 0")
    if cfg.vol_window <= 1:
        raise ValueError("vol_window must be > 1")
    if not (0.0 < cfg.ewma_decay < 1.0):
        raise ValueError("ewma_decay must be in (0, 1)")
    if cfg.annualization <= 0:
        raise ValueError("annualization must be > 0")


def _compute_signal(ret_df: pd.DataFrame, cfg: TSMOMConfig) -> pd.DataFrame:
    signal_cut = cfg.mom_threshold

    if cfg.signal_mode == "monthly_12_1":
        # For ETF data we use total returns as an excess-return proxy.
        monthly_ret_df = ret_df.resample(cfg.rebalance).sum(min_count=1)
        total_lag = cfg.skip_recent + cfg.signal_lag
        trailing_ret_df = monthly_ret_df.shift(total_lag).rolling(cfg.lookback).sum()
        monthly_signal_df = (trailing_ret_df > signal_cut).astype(float) - (trailing_ret_df < -signal_cut).astype(float)
        signal_df = monthly_signal_df.reindex(ret_df.index, method="ffill")
    elif cfg.signal_mode == "daily":
        total_lag = cfg.skip_recent + cfg.signal_lag
        trailing_ret_df = ret_df.shift(total_lag).rolling(cfg.lookback).sum()
        signal_df = (trailing_ret_df > signal_cut).astype(float) - (trailing_ret_df < -signal_cut).astype(float)
    else:
        raise ValueError("signal_mode must be one of {'monthly_12_1', 'daily'}")

    return signal_df.fillna(0.0)


def _estimate_daily_vol(ret_df: pd.DataFrame, cfg: TSMOMConfig) -> pd.DataFrame:
    if cfg.use_ewm_vol:
        # EWMA of squared demeaned returns (simple GARCH-like estimate).
        ewma_alpha = 1.0 - cfg.ewma_decay
        ewma_mean_df = ret_df.ewm(alpha=ewma_alpha, adjust=False).mean()
        centered_sq_df = (ret_df - ewma_mean_df).pow(2)
        ewma_var_df = centered_sq_df.ewm(alpha=ewma_alpha, adjust=False).mean()
        return np.sqrt(ewma_var_df)

    return ret_df.rolling(cfg.vol_window).std(ddof=1)
