"""Alpha features: high-value statistical and interaction features for prediction.

Includes:
- Rolling autocorrelation (trend persistence / mean-reversion detection)
- Rolling skewness (tail risk / lottery effect)
- Max drawdown in window (crash risk)
- Volume-price divergence (confirmation/divergence signal)
- Volatility of volatility (regime stability)
- Feature interaction terms (known alpha combinations)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_alpha_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add high-value alpha features per ticker."""
    df = df.sort_values(["ticker", "date"]).copy()
    grouped = df.groupby("ticker")

    # --- Rolling autocorrelation of returns ---
    # Positive → trending, Negative → mean-reverting
    if "log_ret_1d" in df.columns:
        ret = df["log_ret_1d"]
        ret_lag = grouped["log_ret_1d"].shift(1)
        for w in [20, 60]:
            df[f"ret_autocorr_{w}d"] = (
                ret.rolling(w, min_periods=w // 2)
                .corr(ret_lag)
            )

    # --- Rolling skewness of returns ---
    # Left-skewed → crash risk, right-skewed → lottery-ticket
    if "log_ret_1d" in df.columns:
        for w in [20, 60]:
            df[f"ret_skew_{w}d"] = grouped["log_ret_1d"].transform(
                lambda x: x.rolling(w, min_periods=w // 2).skew()
            )

    # --- Rolling kurtosis ---
    # High kurtosis → fat tails → higher risk
    if "log_ret_1d" in df.columns:
        df["ret_kurtosis_60d"] = grouped["log_ret_1d"].transform(
            lambda x: x.rolling(60, min_periods=30).kurt()
        )

    # --- Max drawdown in rolling window ---
    if "close" in df.columns:
        for w in [20, 60]:
            roll_max = grouped["close"].transform(
                lambda x: x.rolling(w, min_periods=w // 2).max()
            )
            dd_series = df["close"] / roll_max - 1
            df[f"max_dd_{w}d"] = dd_series.groupby(df["ticker"]).transform(
                lambda x: x.rolling(w, min_periods=w // 2).min()
            )

    # --- Volume-price divergence ---
    # Price up but volume down = bearish divergence
    if "ret_5d" in df.columns and "volume" in df.columns:
        vol_chg_5d = grouped["volume"].pct_change(5)
        # Divergence = price direction vs volume direction disagreement
        df["vol_price_div_5d"] = np.sign(df["ret_5d"]) * np.sign(vol_chg_5d)
        # Continuous version: correlation of price and volume changes
        df["vol_price_corr_20d"] = (
            df["ret_5d"].rolling(20, min_periods=10)
            .corr(vol_chg_5d)
        )

    # --- Volatility of volatility ---
    if "realized_vol_20d" in df.columns:
        df["vol_of_vol_60d"] = grouped["realized_vol_20d"].transform(
            lambda x: x.rolling(60, min_periods=20).std()
        )
        # Normalized: vol_of_vol / mean_vol
        vol_mean = grouped["realized_vol_20d"].transform(
            lambda x: x.rolling(60, min_periods=20).mean()
        )
        df["vol_of_vol_ratio"] = df["vol_of_vol_60d"] / vol_mean.replace(0, np.nan)

    # --- Information ratio (rolling Sharpe proxy) ---
    if "log_ret_1d" in df.columns:
        for w in [20, 60]:
            roll_mean = grouped["log_ret_1d"].transform(
                lambda x: x.rolling(w, min_periods=w // 2).mean()
            )
            roll_std = grouped["log_ret_1d"].transform(
                lambda x: x.rolling(w, min_periods=w // 2).std()
            )
            df[f"info_ratio_{w}d"] = (roll_mean / roll_std.replace(0, np.nan)) * np.sqrt(252)

    # --- Relative strength (price momentum normalized) ---
    # Annualized log-return
    if "close" in df.columns:
        for w in [60, 120]:
            df[f"price_roc_smooth_{w}d"] = grouped["close"].transform(
                lambda x: np.log(x / x.shift(w))
            ) / w * 252

    # --- Feature interactions (known alpha combinations) ---
    _add_interaction_features(df)

    return df


def _add_interaction_features(df: pd.DataFrame) -> None:
    """Add multiplicative interaction features between known alpha factors."""
    # Momentum × Volatility: momentum signal quality depends on vol regime
    if "ret_20d" in df.columns and "realized_vol_20d" in df.columns:
        vol = df["realized_vol_20d"].replace(0, np.nan)
        df["mom_vol_interaction"] = df["ret_20d"] / vol  # risk-adjusted momentum

    # RSI × Volume: RSI reversal signals are stronger with volume confirmation
    if "rsi_14" in df.columns and "volume_ratio" in df.columns:
        # RSI distance from neutral (50)
        rsi_extreme = (df["rsi_14"] - 50).abs() / 50
        df["rsi_vol_interaction"] = rsi_extreme * df["volume_ratio"]

    # Momentum vs mean-reversion quality
    if "ret_5d" in df.columns and "ret_60d" in df.columns:
        # Short-term reversal in context of long-term trend
        df["reversal_quality"] = -df["ret_5d"] * np.sign(df["ret_60d"])

    # Volatility regime × momentum direction
    if "realized_vol_20d" in df.columns and "realized_vol_60d" in df.columns:
        df["vol_regime_change"] = df["realized_vol_20d"] / df["realized_vol_60d"].replace(0, np.nan)

    # ADX × momentum: strong trend + momentum alignment
    # Note: momentum.py creates "adx_14", not "adx" — intentionally kept as "adx"
    # to avoid activating this feature which was never part of the trained model.
    if "adx" in df.columns and "ret_20d" in df.columns:
        df["trend_strength_mom"] = df["adx"] * np.sign(df["ret_20d"])

    # Liquidity-adjusted momentum
    if "ret_20d" in df.columns and "amihud_20d" in df.columns:
        amihud = df["amihud_20d"].replace(0, np.nan)
        # High liquidity momentum is more reliable
        df["liquid_momentum"] = df["ret_20d"] / np.log1p(amihud * 1e6)

    # Distance from 52w high × volume
    if "pct_from_52w_high" in df.columns and "volume_ratio" in df.columns:
        df["breakout_vol_confirm"] = df["pct_from_52w_high"] * df["volume_ratio"]
