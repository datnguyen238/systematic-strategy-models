from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class DataConfig:
    tickers: list[str]
    start: str  # "YYYY-MM-DD"
    end: str    # "YYYY-MM-DD"
    interval: str = "1d"
    auto_adjust: bool = True  # adjusted prices (splits/dividends) -> consistent returns
    cache_dir: str = "data/processed"


def _cache_path(cfg: DataConfig) -> Path:
    tickers_slug = "-".join(cfg.tickers)
    fname = f"prices_{tickers_slug}_{cfg.start}_{cfg.end}_{cfg.interval}_adj{int(cfg.auto_adjust)}.parquet"
    return Path(cfg.cache_dir) / fname


def fetch_prices(cfg: DataConfig, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch OHLCV data via yfinance and cache to parquet.
    Returns a MultiIndex columns DataFrame like:
      (Open, Ticker), (High, Ticker), ... (Close, Ticker), (Volume, Ticker)
    """
    cache_path = _cache_path(cfg)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force_refresh:
        df = pd.read_parquet(cache_path)
        # Ensure datetime index
        df.index = pd.to_datetime(df.index)
        return df

    df = yf.download(
        tickers=cfg.tickers,
        start=cfg.start,
        end=cfg.end,
        interval=cfg.interval,
        auto_adjust=cfg.auto_adjust,
        group_by="column",
        threads=True,
        progress=False,
    )

    if df is None or df.empty:
        raise RuntimeError("yfinance returned empty data. Check tickers/date range.")

    # yfinance returns:
    # - single ticker: columns like ["Open","High",...]
    # - multi ticker: columns is MultiIndex (PriceField, Ticker)
    if not isinstance(df.columns, pd.MultiIndex):
        # normalize to MultiIndex for consistency
        df.columns = pd.MultiIndex.from_product([df.columns, [cfg.tickers[0]]])

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    df.to_parquet(cache_path)
    return df


def close_prices(prices_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Extract close prices into a 2D DataFrame: rows=dates, cols=tickers.
    """
    if not isinstance(prices_ohlcv.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns (Field, Ticker).")
    if "Close" not in prices_ohlcv.columns.get_level_values(0):
        raise ValueError("No 'Close' field found.")

    closes = prices_ohlcv["Close"].copy()
    closes = closes.dropna(how="all")
    return closes


def compute_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """
    Log returns from close prices. Safer for aggregation and vol estimation.
    """
    closes = closes.sort_index()
    rets = np.log(closes).diff()
    rets = rets.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return rets


def outlier_report(returns: pd.DataFrame, z_thresh: float = 10.0) -> pd.DataFrame:
    """
    Identify extreme return observations that can fake mean reversion.
    Flags points where |r - median| / MAD > z_thresh (robust z-score).
    """
    # robust z using median absolute deviation (MAD)
    med = returns.median(axis=0, skipna=True)
    mad = (returns.sub(med, axis=1)).abs().median(axis=0, skipna=True)
    mad = mad.replace(0.0, np.nan)

    robust_z = returns.sub(med, axis=1).abs().div(mad, axis=1)
    flagged = robust_z > z_thresh

    rows = []
    for tkr in returns.columns:
        idx = flagged.index[flagged[tkr].fillna(False)]
        for dt in idx:
            rows.append(
                {
                    "date": dt,
                    "ticker": tkr,
                    "return": float(returns.loc[dt, tkr]),
                    "robust_z": float(robust_z.loc[dt, tkr]),
                }
            )

    rep = pd.DataFrame(rows).sort_values(["ticker", "date"])
    return rep
