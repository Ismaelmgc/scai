"""Momentum & trend features: moving averages, z-scores, breakouts."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_momentum_features(
    df: pd.DataFrame,
    ma_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add trend and momentum features.

    Parameters
    ----------
    df : DataFrame with ticker, date, close, volume (and ideally ret_* columns).
    ma_windows : moving-average periods (default [5, 10, 20, 50, 200]).
    """
    ma_windows = ma_windows or [5, 10, 20, 50, 200]
    df = df.sort_values(["ticker", "date"]).copy()
    grouped = df.groupby("ticker")

    # Simple moving averages and price relative to SMA
    for w in ma_windows:
        sma_col = f"sma_{w}"
        df[sma_col] = grouped["close"].transform(
            lambda x: x.rolling(w, min_periods=max(w // 2, 2)).mean()
        )
        df[f"close_vs_sma_{w}"] = df["close"] / df[sma_col] - 1

    # EMA
    for w in [12, 26]:
        df[f"ema_{w}"] = grouped["close"].transform(lambda x: x.ewm(span=w, adjust=False).mean())

    # MACD
    if "ema_12" in df.columns and "ema_26" in df.columns:
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = grouped["macd"].transform(lambda x: x.ewm(span=9, adjust=False).mean())
        df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Rolling z-score of returns
    for w in [20, 60]:
        ret_col = f"ret_{w}d" if f"ret_{w}d" in df.columns else None
        if ret_col:
            mu = grouped[ret_col].transform(lambda x: x.rolling(w, min_periods=w // 2).mean())
            sigma = grouped[ret_col].transform(lambda x: x.rolling(w, min_periods=w // 2).std())
            df[f"zscore_ret_{w}d"] = (df[ret_col] - mu) / sigma.replace(0, np.nan)

    # 52-week high/low proximity
    df["high_52w"] = grouped["high"].transform(lambda x: x.rolling(252, min_periods=60).max())
    df["low_52w"] = grouped["low"].transform(lambda x: x.rolling(252, min_periods=60).min())
    df["pct_from_52w_high"] = df["close"] / df["high_52w"] - 1
    df["pct_from_52w_low"] = df["close"] / df["low_52w"] - 1

    # Breakout flags
    df["breakout_20d_high"] = (df["close"] >= grouped["high"].transform(
        lambda x: x.rolling(20, min_periods=10).max()
    )).astype(int)
    df["breakdown_20d_low"] = (df["close"] <= grouped["low"].transform(
        lambda x: x.rolling(20, min_periods=10).min()
    )).astype(int)

    # RSI (14-day)
    delta = grouped["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.groupby(df["ticker"]).transform(lambda x: x.ewm(span=14, adjust=False).mean())
    avg_loss = loss.groupby(df["ticker"]).transform(lambda x: x.ewm(span=14, adjust=False).mean())
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Volume momentum
    df["volume_sma_20"] = grouped["volume"].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"].replace(0, np.nan)

    # ── Bollinger Bands ────────────────────────────────────
    bb_sma = grouped["close"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    bb_std = grouped["close"].transform(lambda x: x.rolling(20, min_periods=10).std())
    df["bb_upper"] = bb_sma + 2 * bb_std
    df["bb_lower"] = bb_sma - 2 * bb_std
    bb_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_pctb"] = (df["close"] - df["bb_lower"]) / bb_range  # %B position
    df["bb_width"] = bb_range / bb_sma.replace(0, np.nan)       # bandwidth

    # ── Stochastic Oscillator (%K, %D) ─────────────────────
    low_14 = grouped["low"].transform(lambda x: x.rolling(14, min_periods=7).min())
    high_14 = grouped["high"].transform(lambda x: x.rolling(14, min_periods=7).max())
    denom = (high_14 - low_14).replace(0, np.nan)
    df["stoch_k"] = 100 * (df["close"] - low_14) / denom
    df["stoch_d"] = grouped["stoch_k"].transform(lambda x: x.rolling(3, min_periods=1).mean())

    # ── Williams %R ────────────────────────────────────────
    df["williams_r"] = -100 * (high_14 - df["close"]) / denom

    # ── CCI (Commodity Channel Index) ──────────────────────
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_grouped = typical_price.groupby(df["ticker"])
    tp_mean = tp_grouped.transform(lambda x: x.rolling(20, min_periods=10).mean())
    tp_mad = tp_grouped.transform(
        lambda x: x.rolling(20, min_periods=10).apply(
            lambda w: np.mean(np.abs(w - w.mean())), raw=True)
    )
    df["cci_20"] = (typical_price - tp_mean) / (0.015 * tp_mad.replace(0, np.nan))

    # ── ADX (Average Directional Index) ────────────────────
    high_diff = grouped["high"].diff()
    low_diff = -grouped["low"].diff()
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)

    # Use ATR from true range if available, else compute
    if "true_range" in df.columns:
        tr = df["true_range"]
    else:
        prev_close = grouped["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)

    # Smoothed averages (Wilder's 14-period)
    _plus_dm_s = pd.Series(plus_dm, index=df.index)
    _minus_dm_s = pd.Series(minus_dm, index=df.index)
    atr_14 = tr.groupby(df["ticker"]).transform(lambda x: x.ewm(span=14, adjust=False).mean())
    plus_di = 100 * _plus_dm_s.groupby(df["ticker"]).transform(
        lambda x: x.ewm(span=14, adjust=False).mean()
    ) / atr_14.replace(0, np.nan)
    minus_di = 100 * _minus_dm_s.groupby(df["ticker"]).transform(
        lambda x: x.ewm(span=14, adjust=False).mean()
    ) / atr_14.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx_14"] = dx.groupby(df["ticker"]).transform(lambda x: x.ewm(span=14, adjust=False).mean())
    df["plus_di_14"] = plus_di
    df["minus_di_14"] = minus_di

    # ── Donchian Channels ──────────────────────────────────
    df["donchian_upper_20"] = grouped["high"].transform(
        lambda x: x.rolling(20, min_periods=10).max())
    df["donchian_lower_20"] = grouped["low"].transform(
        lambda x: x.rolling(20, min_periods=10).min())
    don_range = (df["donchian_upper_20"] - df["donchian_lower_20"]).replace(0, np.nan)
    df["donchian_position"] = (df["close"] - df["donchian_lower_20"]) / don_range

    # ── Ichimoku Cloud ─────────────────────────────────────
    high_9 = grouped["high"].transform(lambda x: x.rolling(9, min_periods=5).max())
    low_9 = grouped["low"].transform(lambda x: x.rolling(9, min_periods=5).min())
    df["ichi_tenkan"] = (high_9 + low_9) / 2

    high_26 = grouped["high"].transform(lambda x: x.rolling(26, min_periods=13).max())
    low_26 = grouped["low"].transform(lambda x: x.rolling(26, min_periods=13).min())
    df["ichi_kijun"] = (high_26 + low_26) / 2

    df["ichi_tenkan_kijun"] = ((df["ichi_tenkan"] - df["ichi_kijun"])
                               / df["close"].replace(0, np.nan))
    senkou_a = (df["ichi_tenkan"] + df["ichi_kijun"]) / 2
    senkou_b_high = grouped["high"].transform(lambda x: x.rolling(52, min_periods=26).max())
    senkou_b_low = grouped["low"].transform(lambda x: x.rolling(52, min_periods=26).min())
    senkou_b = (senkou_b_high + senkou_b_low) / 2
    df["ichi_cloud_thickness"] = (senkou_a - senkou_b) / df["close"].replace(0, np.nan)
    df["ichi_price_vs_cloud"] = (df["close"] - senkou_a) / df["close"].replace(0, np.nan)

    return df
