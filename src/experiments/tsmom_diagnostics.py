from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.engine import backtest_positions
from src.data import DataConfig, close_prices, compute_returns, fetch_prices
from src.strategies.tsmom import TSMOMConfig, generate_positions_tsmom


def _ols_coef_tstat(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(df) < 5:
        return float("nan"), float("nan")

    X = np.column_stack([np.ones(len(df)), df["x"].to_numpy()])
    yv = df["y"].to_numpy()

    beta = np.linalg.lstsq(X, yv, rcond=None)[0]
    resid = yv - X @ beta
    dof = len(df) - X.shape[1]
    if dof <= 0:
        return float(beta[1]), float("nan")

    sigma2 = (resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(cov), 1e-18, np.inf))
    t = beta[1] / se[1]
    return float(beta[1]), float(t)


def lag_persistence_table(monthly_rets: pd.DataFrame, max_lag: int = 12) -> pd.DataFrame:
    """
    Section 3 style check: does lagged k-month return forecast next month return?
    We pool by averaging coefficients across instruments.
    """
    rows = []
    y = monthly_rets.shift(-1)

    for k in range(1, max_lag + 1):
        x = monthly_rets.rolling(k).sum().shift(1)
        coefs = []
        tstats = []

        for col in monthly_rets.columns:
            b, t = _ols_coef_tstat(y[col], x[col])
            if np.isfinite(b):
                coefs.append(b)
            if np.isfinite(t):
                tstats.append(t)

        rows.append(
            {
                "lag_months": k,
                "avg_beta": float(np.nanmean(coefs)) if coefs else float("nan"),
                "avg_tstat": float(np.nanmean(tstats)) if tstats else float("nan"),
                "n_assets": int(len(coefs)),
            }
        )

    return pd.DataFrame(rows)


def trend_smile_regression(tsmom_monthly: pd.Series, market_monthly: pd.Series) -> dict:
    """
    Convexity check: r_tsmom = a + b*market + c*market^2.
    Positive c is the 'trend smile' signature.
    """
    df = pd.concat([tsmom_monthly.rename("y"), market_monthly.rename("m")], axis=1).dropna()
    if len(df) < 10:
        return {"alpha": float("nan"), "beta_mkt": float("nan"), "beta_mkt2": float("nan"), "n_obs": len(df)}

    m = df["m"].to_numpy()
    X = np.column_stack([np.ones(len(df)), m, m * m])
    y = df["y"].to_numpy()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    return {
        "alpha": float(beta[0]),
        "beta_mkt": float(beta[1]),
        "beta_mkt2": float(beta[2]),
        "n_obs": int(len(df)),
    }


def avg_pairwise_corr(df: pd.DataFrame) -> float:
    c = df.corr()
    tri = c.where(np.triu(np.ones(c.shape), 1).astype(bool)).stack()
    if len(tri) == 0:
        return float("nan")
    return float(tri.mean())


def main() -> None:
    tickers = ["SPY", "TLT", "GLD", "HYG", "EFA", "DBC", "VNQ", "UUP"]
    data_cfg = DataConfig(tickers=tickers, start="2006-01-01", end="2026-01-01")

    prices = fetch_prices(data_cfg)
    closes = close_prices(prices)
    rets = compute_returns(closes)

    strat_cfg = TSMOMConfig(
        signal_mode="monthly_12_1",
        lookback=12,
        skip_recent=1,
        signal_lag=1,
        hold_rebalances=1,
        use_ewm_vol=True,
        ewma_decay=0.94,
        rebalance="ME",
        equal_weight_active=True,
    )

    pos = generate_positions_tsmom(rets, strat_cfg)
    bt = backtest_positions(pos, rets, cost_bps=0.0)

    monthly_rets = rets.resample("ME").sum(min_count=1)
    tsmom_monthly = bt.returns.resample("ME").sum(min_count=1)
    mkt_monthly = monthly_rets.mean(axis=1)

    persistence = lag_persistence_table(monthly_rets, max_lag=12)
    smile = trend_smile_regression(tsmom_monthly, mkt_monthly)

    # Passive correlations: long-only asset return structure.
    passive_corr = avg_pairwise_corr(monthly_rets)

    # TS correlations: return contribution per sleeve from held weights.
    sleeves = pos.shift(1).fillna(0.0) * rets
    sleeves_m = sleeves.resample("ME").sum(min_count=1)
    tsmom_corr = avg_pairwise_corr(sleeves_m)

    print("\n=== Lag Persistence (avg across assets) ===")
    print(persistence.to_string(index=False))

    print("\n=== Trend Smile Regression ===")
    for k, v in smile.items():
        if isinstance(v, float):
            print(f"{k:>10}: {v:.6f}")
        else:
            print(f"{k:>10}: {v}")

    print("\n=== Correlation Structure ===")
    print(f"Passive avg pairwise corr : {passive_corr:.4f}")
    print(f"TSMOM avg pairwise corr   : {tsmom_corr:.4f}")


if __name__ == "__main__":
    main()
