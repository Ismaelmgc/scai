"""Lagged/rolling-difference features shared by both ML products.

Extracted verbatim from the retired V2 research pipeline (scripts/archived/
run_smallcap_pipeline.py); the small-cap daily pipeline and the liquidcap feature
build both call it, so its output is part of each model's feature contract —
change it only together with a retrain.
"""
from __future__ import annotations

import pandas as pd


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged momentum and mean-reversion features.

    These capture autoregressive patterns the model can exploit.
    """
    df = df.sort_values(["ticker", "date"]).copy()
    grouped = df.groupby("ticker")

    # Lagged returns — model can learn momentum/reversion from recent history
    for col in ["ret_1d", "ret_5d", "ret_20d"]:
        if col in df.columns:
            for lag in [1, 5, 10]:
                df[f"{col}_lag{lag}"] = grouped[col].shift(lag)

    # Lagged volume ratio
    if "volume_ratio" in df.columns:
        for lag in [1, 5]:
            df[f"volume_ratio_lag{lag}"] = grouped["volume_ratio"].shift(lag)

    # Lagged RSI
    if "rsi_14" in df.columns:
        for lag in [1, 5]:
            df[f"rsi_14_lag{lag}"] = grouped["rsi_14"].shift(lag)

    # Momentum acceleration (2nd derivative)
    if "ret_5d" in df.columns:
        df["momentum_accel_5d"] = grouped["ret_5d"].diff()
    if "ret_20d" in df.columns:
        df["momentum_accel_20d"] = grouped["ret_20d"].diff()

    # Return dispersion across recent windows (vol of vol proxy)
    if all(c in df.columns for c in ["ret_1d", "ret_5d", "ret_20d"]):
        df["return_dispersion"] = df[["ret_1d", "ret_5d", "ret_20d"]].std(axis=1)

    # Mean reversion signal: distance from rolling mean
    if "close" in df.columns:
        for w in [10, 30]:
            sma = grouped["close"].transform(lambda x: x.rolling(w, min_periods=w//2).mean())
            df[f"mean_rev_{w}d"] = (df["close"] - sma) / sma

    # Volume trend
    if "volume" in df.columns:
        vol_sma_5 = grouped["volume"].transform(lambda x: x.rolling(5, min_periods=3).mean())
        vol_sma_20 = grouped["volume"].transform(lambda x: x.rolling(20, min_periods=10).mean())
        df["volume_trend"] = vol_sma_5 / vol_sma_20.clip(lower=1)

    return df
