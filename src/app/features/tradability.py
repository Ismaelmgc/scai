"""Point-in-time tradability gate.

Delisted/illiquid tickers stay in the stored OHLCV **on purpose** (training
needs them to avoid survivorship bias), but they must never be *selected* at
signal time: live paper trading bought sub-penny zombies (SRNE @ $0.0006 with
$153/day dollar volume) because nothing re-checked tradability after the
universe snapshot aged.

Single source of truth used by BOTH the production pipeline
(``scripts/daily_pipeline.py``) and the walk-forward harness
(``scripts/v3/_v3_harness.py``) so backtests measure the same universe that
production can actually trade.
"""
from __future__ import annotations

import pandas as pd

# Defaults validated by scripts/v3/20_filter_sweep.py (16-fold walk-forward
# threshold sweep). Update there first if these ever change.
MIN_PRICE = 1.50
MIN_ADV_USD = 500_000

# Production guard: features older than this vs. the run date mean the data
# download failed silently — signals must not be generated from stale data.
MAX_STALENESS_DAYS = 4  # calendar days; covers weekend + one holiday


def tradable_mask(
    df: pd.DataFrame,
    min_price: float = MIN_PRICE,
    min_adv_usd: float = MIN_ADV_USD,
) -> pd.Series:
    """Boolean mask of rows that are actually tradable at signal time.

    Uses trailing columns only (``close``, ``adv_usd_20d``) so the mask is
    point-in-time safe. NaN in either column counts as NOT tradable.
    Apply to selection cross-sections only — never to training rows.
    """
    price_ok = df["close"] >= min_price
    adv_ok = df["adv_usd_20d"] >= min_adv_usd
    return (price_ok & adv_ok).fillna(False)


def is_stale(latest_feature_date: pd.Timestamp, today: pd.Timestamp,
             max_staleness_days: int = MAX_STALENESS_DAYS) -> bool:
    """True if the latest feature date is too old to trade on."""
    return (today.normalize() - latest_feature_date.normalize()).days > max_staleness_days
