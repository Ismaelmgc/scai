"""Point-in-time data helpers – prevent temporal leakage."""

from __future__ import annotations

from datetime import date

import pandas as pd


def as_of(df: pd.DataFrame, as_of_date: date | str, date_col: str = "date") -> pd.DataFrame:
    """Return rows available on or before *as_of_date* (point-in-time filter).

    This is the **core leakage-prevention primitive**.  Every feature / label
    lookup in the system must go through this (or an equivalent) filter.
    """
    cutoff = pd.Timestamp(as_of_date)
    return df.loc[df[date_col] <= cutoff].copy()


def latest_as_of(
    df: pd.DataFrame,
    as_of_date: date | str,
    date_col: str = "date",
    group_col: str = "ticker",
) -> pd.DataFrame:
    """Return the *most recent* row per group as of a date."""
    filtered = as_of(df, as_of_date, date_col)
    if filtered.empty:
        return filtered
    idx = filtered.groupby(group_col)[date_col].idxmax()
    return filtered.loc[idx].copy()


def lag_safe_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: list[str],
    date_col: str = "date",
    lag_days: int = 1,
) -> pd.DataFrame:
    """Merge *right* into *left* ensuring at least ``lag_days`` delay.

    This prevents using information that would not be available at
    decision time (e.g. same-day fundamentals for overnight signals).
    """
    right = right.copy()
    right[date_col] = pd.to_datetime(right[date_col]) + pd.Timedelta(days=lag_days)
    return left.merge(right, on=on, how="left", suffixes=("", "_right"))
