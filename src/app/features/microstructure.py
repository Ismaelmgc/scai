"""Microstructure features: VWAP deviation, volume profile, spread proxies.

These features capture order-flow and execution quality signals
that are particularly informative for small-cap stocks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_microstructure_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add microstructure-based features.

    Parameters
    ----------
    df : DataFrame with ticker, date, open, high, low, close, volume,
         and optionally vwap, transactions.
    windows : Rolling windows (default [5, 20]).
    """
    windows = windows or [5, 20]
    df = df.sort_values(["ticker", "date"]).copy()
    grouped = df.groupby("ticker")

    # ── VWAP deviation ──────────────────────────────────────
    # If VWAP is available, measure how far close deviates from it
    if "vwap" in df.columns:
        df["vwap_deviation"] = (df["close"] - df["vwap"]) / df["vwap"].replace(0, np.nan)
        for w in windows:
            df[f"vwap_dev_avg_{w}d"] = grouped["vwap_deviation"].transform(
                lambda x: x.rolling(w, min_periods=max(w // 2, 2)).mean()
            )
        # Persistent VWAP deviation suggests accumulation/distribution
        df["vwap_dev_zscore"] = grouped["vwap_deviation"].transform(
            lambda x: (x - x.rolling(20, min_periods=10).mean())
            / x.rolling(20, min_periods=10).std().replace(0, np.nan)
        )

    # ── Volume profile ──────────────────────────────────────
    # Volume concentration: what fraction of recent volume came on up days
    if "log_ret_1d" not in df.columns:
        df["log_ret_1d"] = np.log(df["close"] / grouped["close"].shift(1))

    df["up_volume"] = np.where(df["log_ret_1d"] > 0, df["volume"], 0)
    df["down_volume"] = np.where(df["log_ret_1d"] < 0, df["volume"], 0)

    for w in windows:
        up_sum = grouped["up_volume"].transform(
            lambda x: x.rolling(w, min_periods=max(w // 2, 2)).sum()
        )
        total_sum = grouped["volume"].transform(
            lambda x: x.rolling(w, min_periods=max(w // 2, 2)).sum()
        )
        df[f"up_volume_pct_{w}d"] = up_sum / total_sum.replace(0, np.nan)

    # Volume-weighted return: do big-volume days move price more?
    df["abs_ret_x_vol"] = df["log_ret_1d"].abs() * df["volume"]
    for w in [20]:
        df[f"vol_weighted_impact_{w}d"] = (
            grouped["abs_ret_x_vol"].transform(
                lambda x: x.rolling(w, min_periods=10).mean()
            )
            / grouped["volume"].transform(
                lambda x: x.rolling(w, min_periods=10).mean()
            ).replace(0, np.nan)
        )

    # ── Transactions-based features ─────────────────────────
    if "transactions" in df.columns:
        # Average trade size (volume / transactions)
        df["avg_trade_size"] = df["volume"] / df["transactions"].replace(0, np.nan)
        for w in [20]:
            df[f"avg_trade_size_{w}d"] = grouped["avg_trade_size"].transform(
                lambda x: x.rolling(w, min_periods=10).mean()
            )
        # Trade size relative to recent average (institutional vs retail proxy)
        df["trade_size_ratio"] = (
            df["avg_trade_size"]
            / grouped["avg_trade_size"].transform(
                lambda x: x.rolling(20, min_periods=10).mean()
            ).replace(0, np.nan)
        )

    # ── Enhanced spread proxy (Corwin-Schultz) ──────────────
    # Beta = sum of (ln(H/L))^2 over 2 consecutive days
    log_hl = np.log(df["high"] / df["low"].replace(0, np.nan))
    log_hl_sq = log_hl ** 2
    df["_log_hl_sq"] = log_hl_sq

    # 2-day high-low
    h2 = pd.concat([df["high"], grouped["high"].shift(1)], axis=1).max(axis=1)
    l2 = pd.concat([df["low"], grouped["low"].shift(1)], axis=1).min(axis=1)
    log_hl_2d = np.log(h2 / l2.replace(0, np.nan))

    # Corwin-Schultz spread estimate
    beta_cs = df["_log_hl_sq"] + grouped["_log_hl_sq"].shift(1)
    gamma_cs = log_hl_2d ** 2
    alpha_cs = (np.sqrt(2 * beta_cs) - np.sqrt(beta_cs)) / (3 - 2 * np.sqrt(2))
    alpha_cs = alpha_cs - np.sqrt(gamma_cs / (3 - 2 * np.sqrt(2)))
    df["cs_spread"] = 2 * (np.exp(alpha_cs.clip(lower=0)) - 1) / (1 + np.exp(alpha_cs.clip(lower=0)))
    df["cs_spread"] = df["cs_spread"].clip(lower=0, upper=0.2)  # cap at 20%

    for w in [20]:
        df[f"cs_spread_{w}d"] = grouped["cs_spread"].transform(
            lambda x: x.rolling(w, min_periods=10).mean()
        )

    df = df.drop(columns=["_log_hl_sq"], errors="ignore")

    # ── On-Balance Volume (OBV) trend ───────────────────────
    df["obv_sign"] = np.sign(df["log_ret_1d"]).fillna(0)
    df["obv_volume"] = df["obv_sign"] * df["volume"]
    df["obv"] = df.groupby("ticker")["obv_volume"].cumsum()
    grp2 = df.groupby("ticker")
    for w in [20, 60]:
        obv_sma = grp2["obv"].transform(
            lambda x: x.rolling(w, min_periods=max(w // 2, 2)).mean()
        )
        df[f"obv_vs_sma_{w}d"] = (df["obv"] - obv_sma) / obv_sma.abs().replace(0, np.nan)

    # Drop intermediate columns
    df = df.drop(columns=["up_volume", "down_volume", "abs_ret_x_vol",
                           "obv_sign", "obv_volume", "log_hl_sq"],
                 errors="ignore")

    return df
