"""Liquidity features: dollar volume, turnover, Amihud, spread proxy, ADV."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_liquidity_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
    max_participation: float = 0.05,
) -> pd.DataFrame:
    """Add liquidity and executability features.

    Parameters
    ----------
    df : DataFrame with columns ticker, date, open, high, low, close, volume.
    windows : averaging windows (default [5, 20, 60]).
    max_participation : max fraction of volume for capacity estimate.
    """
    windows = windows or [5, 20, 60]
    df = df.sort_values(["ticker", "date"]).copy()
    grouped = df.groupby("ticker")

    # Dollar volume
    df["dollar_volume"] = df["close"] * df["volume"]

    # Average daily volume (shares and USD)
    for w in windows:
        df[f"adv_{w}d"] = grouped["volume"].transform(
            lambda x: x.rolling(w, min_periods=max(w // 2, 2)).mean()
        )
        df[f"adv_usd_{w}d"] = grouped["dollar_volume"].transform(
            lambda x: x.rolling(w, min_periods=max(w // 2, 2)).mean()
        )

    # Turnover (requires shares_outstanding; proxy with volume/adv ratio)
    if "shares_outstanding" in df.columns:
        df["turnover"] = df["volume"] / df["shares_outstanding"]
    else:
        # Volume relative to own 60d average as proxy
        df["volume_ratio_60d"] = df["volume"] / df.get("adv_60d", df["volume"])

    # Amihud illiquidity: |return| / dollar_volume  (lower = more liquid)
    if "log_ret_1d" not in df.columns:
        df["log_ret_1d"] = np.log(df["close"] / grouped["close"].shift(1))
    df["amihud_daily"] = df["log_ret_1d"].abs() / df["dollar_volume"].replace(0, np.nan)
    for w in [20, 60]:
        df[f"amihud_{w}d"] = grouped["amihud_daily"].transform(
            lambda x: x.rolling(w, min_periods=max(w // 2, 2)).mean()
        )

    # Spread proxy: (high - low) / ((high + low) / 2)  (Corwin-Schultz inspired)
    df["spread_proxy"] = (df["high"] - df["low"]) / ((df["high"] + df["low"]) / 2)
    for w in [20]:
        df[f"spread_proxy_{w}d"] = grouped["spread_proxy"].transform(
            lambda x: x.rolling(w, min_periods=10).mean()
        )

    # Capacity estimate: how many USD could you trade at max_participation rate
    adv_col = f"adv_usd_{windows[-1]}d"
    if adv_col in df.columns:
        df["capacity_usd"] = df[adv_col] * max_participation

    # Liquidity score (composite, higher = better)
    # Normalised within cross-section later; here raw inverse Amihud
    if "amihud_20d" in df.columns:
        df["liquidity_score_raw"] = 1 / (1 + df["amihud_20d"] * 1e6)

    return df
