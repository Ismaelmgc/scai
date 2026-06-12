"""Sector features: SIC → sector mapping, sector-relative signals, rotation.

Assigns sector labels from SIC codes and computes sector-relative
momentum, mean-reversion, and rotation signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.utils import get_logger

log = get_logger(__name__)

# SIC code → GICS-like sector mapping (first 2 digits of 4-digit SIC)
_SIC_TO_SECTOR = {
    # Agriculture, Forestry, Fishing (01-09) → Materials
    range(100, 1000): "Materials",
    # Mining (10-14) → Energy / Materials
    range(1000, 1400): "Energy",
    range(1400, 1500): "Materials",
    # Construction (15-17) → Industrials
    range(1500, 1800): "Industrials",
    # Manufacturing (20-39) → split
    range(2000, 2100): "Consumer Staples",
    range(2100, 2200): "Consumer Staples",
    range(2200, 2400): "Consumer Discretionary",
    range(2400, 2600): "Industrials",
    range(2600, 2800): "Materials",
    range(2800, 2900): "Healthcare",
    range(2900, 3000): "Energy",
    range(3000, 3200): "Industrials",
    range(3200, 3400): "Materials",
    range(3400, 3600): "Industrials",
    range(3600, 3700): "Technology",
    range(3700, 3800): "Industrials",
    range(3800, 3900): "Healthcare",
    range(3900, 4000): "Industrials",
    # Transportation & Utilities (40-49)
    range(4000, 4500): "Industrials",
    range(4500, 4600): "Industrials",
    range(4600, 4700): "Industrials",
    range(4700, 4900): "Communication Services",
    range(4900, 5000): "Utilities",
    # Wholesale & Retail (50-59)
    range(5000, 5200): "Consumer Discretionary",
    range(5200, 5300): "Consumer Discretionary",
    range(5300, 5400): "Consumer Staples",
    range(5400, 5500): "Consumer Staples",
    range(5500, 6000): "Consumer Discretionary",
    # Finance, Insurance, Real Estate (60-67)
    range(6000, 6100): "Financials",
    range(6100, 6200): "Financials",
    range(6200, 6300): "Financials",
    range(6300, 6400): "Financials",
    range(6400, 6500): "Financials",
    range(6500, 6600): "Real Estate",
    range(6600, 6800): "Financials",
    # Services (70-89)
    range(7000, 7400): "Consumer Discretionary",
    range(7400, 7500): "Industrials",
    range(7500, 7600): "Consumer Discretionary",
    range(7600, 7700): "Consumer Discretionary",
    range(7700, 7800): "Consumer Discretionary",
    range(7800, 8000): "Communication Services",
    range(8000, 8100): "Healthcare",
    range(8100, 8200): "Industrials",
    range(8200, 8300): "Consumer Discretionary",
    range(8300, 8400): "Industrials",
    range(8400, 8500): "Industrials",
    range(8700, 8800): "Technology",
    range(8800, 9000): "Industrials",
    # Public Administration (91-99)
    range(9100, 10000): "Industrials",
}


def sic_to_sector(sic_code: str | int | None) -> str:
    """Map a SIC code to a GICS-like sector name."""
    if sic_code is None or sic_code == "N/A":
        return "Unknown"
    try:
        code = int(str(sic_code)[:4])
    except (ValueError, TypeError):
        return "Unknown"

    for code_range, sector in _SIC_TO_SECTOR.items():
        if code in code_range:
            return sector
    return "Unknown"


def assign_sectors(
    ohlcv: pd.DataFrame,
    universe: list[dict] | pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add a 'sector' column to OHLCV from universe metadata or SIC codes.

    Parameters
    ----------
    ohlcv : OHLCV DataFrame with 'ticker' column.
    universe : List of dicts or DataFrame with 'ticker' and 'sic_code' columns.
    """
    df = ohlcv.copy()

    if universe is not None:
        uni_df = pd.DataFrame(universe) if isinstance(universe, list) else universe.copy()

        if "sic_code" in uni_df.columns:
            uni_df["sector"] = uni_df["sic_code"].apply(sic_to_sector)
            sector_map = dict(zip(uni_df["ticker"], uni_df["sector"], strict=False))
            df["sector"] = df["ticker"].map(sector_map).fillna("Unknown")
        elif "sector" in uni_df.columns:
            sector_map = dict(zip(uni_df["ticker"], uni_df["sector"], strict=False))
            df["sector"] = df["ticker"].map(sector_map).fillna("Unknown")
        else:
            df["sector"] = "Unknown"
    else:
        if "sector" not in df.columns:
            df["sector"] = "Unknown"

    return df


def compute_sector_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add sector-relative and sector rotation features.

    Requires 'sector' and 'log_ret_1d' columns.

    Parameters
    ----------
    df : OHLCV DataFrame with sector column.
    windows : Rolling windows (default [5, 20, 60]).
    """
    windows = windows or [5, 20, 60]
    df = df.sort_values(["ticker", "date"]).copy()

    if "sector" not in df.columns:
        log.warning("no_sector_column", msg="Skipping sector features")
        return df

    if "log_ret_1d" not in df.columns:
        grouped = df.groupby("ticker")
        df["log_ret_1d"] = np.log(df["close"] / grouped["close"].shift(1))

    # ── Sector average returns ──────────────────────────────
    for w in windows:
        # Per-ticker rolling return
        ret_col = f"ret_{w}d" if f"ret_{w}d" in df.columns else None
        if ret_col is None:
            df[f"_tmp_ret_{w}d"] = df.groupby("ticker")["close"].pct_change(w)
            ret_col = f"_tmp_ret_{w}d"

        # Sector mean return (cross-sectional for the same date)
        df[f"sector_ret_{w}d"] = df.groupby(["date", "sector"])[ret_col].transform("mean")

        # Stock return relative to sector
        df[f"ret_vs_sector_{w}d"] = df[ret_col] - df[f"sector_ret_{w}d"]

        # Clean up temp col
        if ret_col.startswith("_tmp_"):
            df = df.drop(columns=[ret_col])

    # ── Sector momentum rank ───────────────────────────────
    # Rank sectors by their average return (sector rotation signal)
    for w in [20, 60]:
        sector_col = f"sector_ret_{w}d"
        if sector_col in df.columns:
            df[f"sector_momentum_rank_{w}d"] = (
                df.groupby("date")[sector_col].rank(pct=True)
            )

    # ── Sector volatility ──────────────────────────────────
    df["sector_vol_20d"] = (
        df.groupby(["sector"])["log_ret_1d"]
        .transform(lambda x: x.rolling(20, min_periods=10).std())
        * np.sqrt(252)
    )

    # ── Sector breadth (% of stocks up in sector today) ────
    df["_is_up"] = (df["log_ret_1d"] > 0).astype(float)
    df["sector_breadth"] = df.groupby(["date", "sector"])["_is_up"].transform("mean")
    df = df.drop(columns=["_is_up"])

    # ── Sector-relative volume ──────────────────────────────
    df["sector_avg_volume"] = df.groupby(["date", "sector"])["volume"].transform("mean")
    df["volume_vs_sector"] = df["volume"] / df["sector_avg_volume"].replace(0, np.nan)

    return df
