"""Meta-learning features — Error-Aware Features (Option 3).

The model's own prediction errors become features, enabling self-correction.
All features are strictly anti-leakage: only use signals with date < current_date.

Features:
1. model_error_ticker_5: Rolling mean error for this ticker (last 5 signals)
2. model_error_sector_20d: Rolling mean error for the sector (last 20 days)
3. model_hit_rate_30d: % of top-8 picks with positive actual return (30d rolling)
4. model_ic_rolling_20d: Rolling rank correlation between predicted and actual (20d)
5. model_error_vol_20d: Volatility of model errors (20d rolling)

Usage:
    from app.features.meta_features import compute_meta_features
    meta_df = compute_meta_features(signal_history, current_date, sector_map)
    # Returns: DataFrame(ticker, meta feature columns) for current_date
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from app.utils import get_logger

log = get_logger(__name__)


def compute_meta_features(
    signal_history: pd.DataFrame,
    current_date: str | date | pd.Timestamp,
    sector_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Compute meta-learning features from past signal history.

    Parameters
    ----------
    signal_history : DataFrame with columns:
        signal_date, ticker, v2_score, actual_ret_20d, outcome_filled
    current_date : Point-in-time cutoff (strictly < this date).
    sector_map : {ticker: sector} mapping for sector-level error features.

    Returns
    -------
    DataFrame with columns: ticker + meta feature columns.
    Only includes tickers that appear in past signals.
    """
    current_date = pd.Timestamp(current_date)
    hist = signal_history.copy()
    hist["signal_date"] = pd.to_datetime(hist["signal_date"])

    # Anti-leakage: outcome (signal_date + 20 trading days ≈ 30 calendar days) must
    # be known AS OF current_date. `outcome_filled` is True for any historical signal
    # at parquet-build time, so we ALSO require signal_date + 30 days < current_date.
    # Without this, meta-features at T leak prices in [T, T+20].
    outcome_cutoff = current_date - pd.Timedelta(days=30)
    past = hist[
        (hist["signal_date"] < outcome_cutoff)
        & (hist["outcome_filled"] == True)  # noqa: E712
    ].copy()

    if past.empty or len(past) < 5:
        return pd.DataFrame(columns=["ticker"])

    past["error"] = past["v2_score"] - past["actual_ret_20d"]
    past = past.sort_values("signal_date")

    # ── 1. Per-ticker rolling error (last 5 signals for that ticker) ──
    ticker_errors = {}
    for ticker, grp in past.groupby("ticker"):
        recent = grp.tail(5)
        ticker_errors[ticker] = {
            "model_error_ticker_5": recent["error"].mean(),
        }
    ticker_df = pd.DataFrame.from_dict(ticker_errors, orient="index")
    ticker_df.index.name = "ticker"
    ticker_df = ticker_df.reset_index()

    # ── 2. Sector-level rolling error (last 20 days) ──
    if sector_map:
        past["sector"] = past["ticker"].map(sector_map)
        # Window: last 30 calendar days of usable signals (relative to outcome_cutoff)
        cutoff_20d = outcome_cutoff - pd.Timedelta(days=30)  # ~20 trading days
        recent_20d = past[past["signal_date"] >= cutoff_20d]
        sector_error = recent_20d.groupby("sector")["error"].mean()
        # Map back to tickers
        ticker_df["model_error_sector_20d"] = ticker_df["ticker"].map(
            lambda t: sector_error.get(sector_map.get(t, ""), np.nan)
        )
    else:
        ticker_df["model_error_sector_20d"] = np.nan

    # ── 3. Global hit rate (top-8 picks, 30d rolling) ──
    cutoff_30d = outcome_cutoff - pd.Timedelta(days=45)  # ~30 trading days
    recent_30d = past[past["signal_date"] >= cutoff_30d]
    # Consider only top-ranked signals (rank < 8)
    top_signals = (recent_30d[recent_30d["rank"] < 8]
                   if "rank" in recent_30d.columns else recent_30d)
    hit_rate = (top_signals["actual_ret_20d"] > 0).mean() if len(top_signals) > 0 else np.nan
    ticker_df["model_hit_rate_30d"] = hit_rate  # same for all tickers (global)

    # ── 4. Rolling IC (rank correlation, last 20 days) ──
    cutoff_ic = outcome_cutoff - pd.Timedelta(days=30)
    recent_ic = past[past["signal_date"] >= cutoff_ic]
    if len(recent_ic) >= 10:
        ic_val, _ = spearmanr(recent_ic["v2_score"], recent_ic["actual_ret_20d"])
        if np.isnan(ic_val):
            ic_val = 0.0
    else:
        ic_val = np.nan
    ticker_df["model_ic_rolling_20d"] = ic_val  # same for all tickers (global)

    # ── 5. Error volatility (last 20 days) ──
    error_vol = recent_ic["error"].std() if len(recent_ic) >= 5 else np.nan
    ticker_df["model_error_vol_20d"] = error_vol  # same for all tickers (global)

    log.info("meta_features_computed",
             date=str(current_date.date()),
             tickers=len(ticker_df),
             past_signals=len(past),
             hit_rate=round(hit_rate, 3) if pd.notna(hit_rate) else None)

    return ticker_df


def build_meta_feature_panel(
    signal_history: pd.DataFrame,
    dates: list[pd.Timestamp],
    sector_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build a full panel of meta features for multiple dates.

    Used during training to create a (date, ticker) → meta_features matrix.
    Anti-leakage: each date only uses signals from strictly before that date.

    Returns: DataFrame with columns: date, ticker, meta feature columns.
    """
    panels = []
    hist = signal_history.copy()
    hist["signal_date"] = pd.to_datetime(hist["signal_date"])

    for dt in dates:
        meta = compute_meta_features(hist, dt, sector_map)
        if meta.empty:
            continue
        meta["date"] = dt
        panels.append(meta)

    if not panels:
        return pd.DataFrame(columns=["date", "ticker"])

    result = pd.concat(panels, ignore_index=True)
    log.info("meta_panel_built", rows=len(result), dates=len(panels))
    return result
