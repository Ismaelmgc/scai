"""Price-action features: returns, gaps, overnight returns."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_price_action_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add price-action features per ticker.

    Expects columns: ticker, date, open, high, low, close, volume.
    All features are computed using only past data (shift where needed).

    Parameters
    ----------
    df : DataFrame sorted by (ticker, date)
    windows : look-back windows in trading days (default [1,5,20,60,120,252])
    """
    windows = windows or [1, 5, 20, 60, 120, 252]
    df = df.sort_values(["ticker", "date"]).copy()
    grouped = df.groupby("ticker")

    # Simple returns over various horizons
    for w in windows:
        df[f"ret_{w}d"] = grouped["close"].pct_change(w)

    # Log returns
    df["log_ret_1d"] = np.log(df["close"] / grouped["close"].shift(1))

    # Overnight return (open vs previous close)
    df["overnight_ret"] = df["open"] / grouped["close"].shift(1) - 1

    # Intraday return
    df["intraday_ret"] = df["close"] / df["open"] - 1

    # Gap (open vs previous high/low)
    df["gap_up"] = (df["open"] > grouped["high"].shift(1)).astype(int)
    df["gap_down"] = (df["open"] < grouped["low"].shift(1)).astype(int)

    # High-low range relative to close
    df["hl_range_pct"] = (df["high"] - df["low"]) / df["close"]

    # Close position within day's range
    hl_diff = df["high"] - df["low"]
    df["close_position"] = np.where(hl_diff > 0, (df["close"] - df["low"]) / hl_diff, 0.5)

    # Reversal features (short-term return following longer-term)
    for short_w, long_w in [(1, 5), (5, 20), (20, 60)]:
        s = f"ret_{short_w}d"
        long_col = f"ret_{long_w}d"
        if s in df.columns and long_col in df.columns:
            df[f"reversal_{short_w}v{long_w}"] = df[long_col] - df[s]

    return df
