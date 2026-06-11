"""Volatility features: realized vol, ATR-like, beta, idiosyncratic vol."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_volatility_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
    market_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Add volatility and risk features.

    Parameters
    ----------
    df : DataFrame with columns ticker, date, open, high, low, close, volume,
         and ideally log_ret_1d from price_action.
    windows : look-back windows (default [5, 20, 60, 120]).
    market_returns : Series indexed by date with market daily returns for beta calc.
    """
    windows = windows or [5, 20, 60, 120]
    df = df.sort_values(["ticker", "date"]).copy()
    grouped = df.groupby("ticker")

    # Ensure log return exists
    if "log_ret_1d" not in df.columns:
        df["log_ret_1d"] = np.log(df["close"] / grouped["close"].shift(1))

    # Realized volatility (annualised)
    for w in windows:
        df[f"realized_vol_{w}d"] = (
            grouped["log_ret_1d"]
            .transform(lambda x: x.rolling(w, min_periods=max(w // 2, 2)).std())
            * np.sqrt(252)
        )

    # ATR-like: average true range / close
    prev_close = grouped["close"].shift(1)
    true_range = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    df["true_range"] = true_range
    for w in [5, 20]:
        df[f"atr_{w}d"] = grouped["true_range"].transform(
            lambda x: x.rolling(w, min_periods=max(w // 2, 2)).mean()
        )
        df[f"atr_pct_{w}d"] = df[f"atr_{w}d"] / df["close"]

    # Downside deviation
    for w in [20, 60]:
        df[f"downside_vol_{w}d"] = (
            grouped["log_ret_1d"]
            .transform(lambda x: x.clip(upper=0).rolling(w, min_periods=max(w // 2, 2)).std())
            * np.sqrt(252)
        )

    # Beta and idiosyncratic vol (if market returns provided)
    if market_returns is not None:
        df = df.set_index("date", drop=False)
        df["mkt_ret"] = market_returns
        df = df.reset_index(drop=True)
        # Re-group after adding mkt_ret column
        grp2 = df.groupby("ticker")
        for w in [60, 120]:
            beta = grp2.apply(
                lambda g: g["log_ret_1d"].rolling(w, min_periods=w // 2).cov(g["mkt_ret"])
                / g["mkt_ret"].rolling(w, min_periods=w // 2).var(),
                include_groups=False,
            )
            df[f"beta_{w}d"] = beta.values if hasattr(beta, "values") else np.nan
        # Idiosyncratic vol = residual vol after removing market component
        if "beta_60d" in df.columns:
            resid = df["log_ret_1d"] - df.get("beta_60d", 1) * df.get("mkt_ret", 0)
            df["idio_vol_60d"] = (
                resid.groupby(df["ticker"])
                .transform(lambda x: x.rolling(60, min_periods=30).std())
                * np.sqrt(252)
            )

    return df
