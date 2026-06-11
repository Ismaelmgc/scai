"""Cross-sectional features: percentile ranks, sector-relative, winsorisation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rank_within_date(
    df: pd.DataFrame,
    cols: list[str],
    date_col: str = "date",
    suffix: str = "_rank",
) -> pd.DataFrame:
    """Add cross-sectional percentile rank (0-1) for each column per date."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        df[f"{col}{suffix}"] = df.groupby(date_col)[col].rank(pct=True)
    return df


def sector_relative(
    df: pd.DataFrame,
    cols: list[str],
    sector_col: str = "sector",
    date_col: str = "date",
    suffix: str = "_sec_rel",
) -> pd.DataFrame:
    """Compute sector-relative z-score for each column."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        grp = df.groupby([date_col, sector_col])[col]
        mu = grp.transform("mean")
        sigma = grp.transform("std").replace(0, np.nan)
        df[f"{col}{suffix}"] = (df[col] - mu) / sigma
    return df


def winsorise(
    df: pd.DataFrame,
    cols: list[str],
    limits: tuple[float, float] = (0.01, 0.99),
    date_col: str = "date",
) -> pd.DataFrame:
    """Winsorise columns per date at the given quantile limits."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        lo = df.groupby(date_col)[col].transform(lambda x: x.quantile(limits[0]))
        hi = df.groupby(date_col)[col].transform(lambda x: x.quantile(limits[1]))
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def normalise_cross_section(
    df: pd.DataFrame,
    cols: list[str],
    date_col: str = "date",
) -> pd.DataFrame:
    """Z-score normalise columns cross-sectionally per date."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        mu = df.groupby(date_col)[col].transform("mean")
        sigma = df.groupby(date_col)[col].transform("std").replace(0, np.nan)
        df[col] = (df[col] - mu) / sigma
    return df


def compute_cross_sectional_features(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    date_col: str = "date",
    sector_col: str = "sector",
) -> pd.DataFrame:
    """Full cross-sectional post-processing pipeline:
    1. Winsorise extreme values
    2. Add percentile ranks
    3. Add sector-relative z-scores
    """
    if feature_cols is None:
        # Auto-detect numeric feature columns
        exclude = {"ticker", "date", "open", "high", "low", "close", "volume"}
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude
        ]

    df = winsorise(df, feature_cols, date_col=date_col)
    df = rank_within_date(df, feature_cols, date_col=date_col)
    if sector_col in df.columns:
        df = sector_relative(df, feature_cols, sector_col=sector_col, date_col=date_col)
    return df
