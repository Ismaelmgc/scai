"""Insider-buying features (price-orthogonal small-cap signal).

Open-market insider purchases (Form 4 code 'P') predict positive abnormal
returns, especially cluster buys in small-caps. Orthogonal to price/volume and,
unlike short interest, not entangled with the squeeze effect.

Two PIT-safe features over a trailing window (default 90 calendar days):
  - insider_buy_count_90d: number of insider buy filings (cluster intensity)
  - insider_buy_value_log90d: log1p of total $ bought

POINT-IN-TIME SAFETY: Form 4 is public on its FILING_DATE; availability is lagged
to filing_date + 1 business day. The trailing window is computed with a
cumulative-sum + two-merge_asof trick (no look-ahead): count in (T-window, T] =
cum(<=T) - cum(<=T-window), each via a backward as-of merge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAG_BDAYS = 1
WINDOW_DAYS = 90

INSIDER_FEATURES = ["insider_buy_count_90d", "insider_buy_value_log90d"]


def add_insider_features(
    features: pd.DataFrame,
    insider: pd.DataFrame,
    window_days: int = WINDOW_DAYS,
    lag_bdays: int = LAG_BDAYS,
) -> tuple[pd.DataFrame, list[str]]:
    """Merge PIT-safe trailing-window insider-buy features. Returns (df, cols)."""
    ins = insider.copy()
    ins["filing_date"] = pd.to_datetime(ins["filing_date"])
    ins["available_date"] = ins["filing_date"] + pd.tseries.offsets.BusinessDay(lag_bdays)
    ins = ins.sort_values(["ticker", "available_date"])
    # Per-ticker cumulative sums (monotonic → window sum = cum_hi - cum_lo).
    ins["cum_n"] = ins.groupby("ticker")["n_buy_filings"].cumsum()
    ins["cum_v"] = ins.groupby("ticker")["buy_value"].cumsum()
    keyed = ins[["available_date", "ticker", "cum_n", "cum_v"]].sort_values("available_date")

    feat = features.copy().reset_index(drop=True)
    feat["_rowid"] = np.arange(len(feat))
    feat["date"] = pd.to_datetime(feat["date"])

    # Cumulative as of T (inclusive).
    hi = pd.merge_asof(
        feat.sort_values("date"), keyed,
        left_on="date", right_on="available_date", by="ticker", direction="backward",
    )[["_rowid", "cum_n", "cum_v"]]

    # Cumulative as of T - window.
    lo_src = feat[["_rowid", "ticker", "date"]].copy()
    lo_src["dm"] = lo_src["date"] - pd.Timedelta(days=window_days)
    lo = pd.merge_asof(
        lo_src.sort_values("dm"), keyed,
        left_on="dm", right_on="available_date", by="ticker", direction="backward",
    )[["_rowid", "cum_n", "cum_v"]].rename(columns={"cum_n": "cum_n_lo", "cum_v": "cum_v_lo"})

    feat = feat.merge(hi, on="_rowid", how="left").merge(lo, on="_rowid", how="left")
    cnt = feat["cum_n"].fillna(0) - feat["cum_n_lo"].fillna(0)
    val = feat["cum_v"].fillna(0) - feat["cum_v_lo"].fillna(0)
    feat["insider_buy_count_90d"] = cnt.clip(lower=0)
    feat["insider_buy_value_log90d"] = np.log1p(val.clip(lower=0))

    feat = feat.drop(columns=["_rowid", "cum_n", "cum_v", "cum_n_lo", "cum_v_lo"])
    return feat, list(INSIDER_FEATURES)
