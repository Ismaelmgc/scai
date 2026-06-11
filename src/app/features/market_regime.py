"""Market-regime features: VIX proxy, market breadth, macro proxies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_market_regime_features(
    market_df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Compute market-level regime indicators.

    Parameters
    ----------
    market_df : DataFrame with columns date, close (market index like SPY),
                and optionally volume.  One row per date.

    Returns a DataFrame indexed by date with regime columns that can be
    broadcast-joined to every ticker.
    """
    df = market_df.sort_values(date_col).copy()

    # Market returns
    df["mkt_ret_1d"] = df["close"].pct_change(1)
    df["mkt_ret_5d"] = df["close"].pct_change(5)
    df["mkt_ret_20d"] = df["close"].pct_change(20)

    # Market volatility (VIX proxy): 20-day realised vol annualised
    df["mkt_vol_20d"] = df["mkt_ret_1d"].rolling(20, min_periods=10).std() * np.sqrt(252)
    df["mkt_vol_60d"] = df["mkt_ret_1d"].rolling(60, min_periods=30).std() * np.sqrt(252)

    # Volatility regime bucketing
    df["vol_regime"] = pd.cut(
        df["mkt_vol_20d"],
        bins=[0, 0.10, 0.18, 0.30, float("inf")],
        labels=["low", "normal", "elevated", "high"],
    )

    # Market trend (SMA cross)
    df["mkt_sma_50"] = df["close"].rolling(50, min_periods=25).mean()
    df["mkt_sma_200"] = df["close"].rolling(200, min_periods=100).mean()
    df["mkt_trend_bull"] = (df["mkt_sma_50"] > df["mkt_sma_200"]).astype(int)

    # Drawdown from rolling max
    df["mkt_cummax"] = df["close"].cummax()
    df["mkt_drawdown"] = df["close"] / df["mkt_cummax"] - 1

    # Breadth placeholder – requires advance/decline data; use return dispersion as proxy
    # If volume available, use volume z-score
    if "volume" in df.columns:
        df["mkt_volume_z"] = (
            (df["volume"] - df["volume"].rolling(60, min_periods=30).mean())
            / df["volume"].rolling(60, min_periods=30).std()
        )

    return df
