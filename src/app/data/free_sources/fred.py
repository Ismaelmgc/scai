"""FRED macro data connector — regime/context features.

Requires free API key from https://fred.stlouisfed.org/docs/api/api_key.html
Set FRED_API_KEY in .env.

These features are NOT for direct model input without validation.
Use as regime filter / gating signal.
"""
from __future__ import annotations

import os
from datetime import date

import pandas as pd

from app.utils import get_logger

log = get_logger(__name__)

# Key macro series for small-cap trading context
MACRO_SERIES = {
    "DFF": "fed_funds_rate",
    "T10Y2Y": "yield_curve_10y2y",
    "BAMLH0A0HYM2": "high_yield_spread",
    "VIXCLS": "vix",
    "DTWEXBGS": "dollar_index",
    "T10YIE": "breakeven_inflation_10y",
    "ICSA": "initial_claims",
}


def download_fred_macro(
    start_date: str = "2019-01-01",
    end_date: str | None = None,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Download macro series from FRED.

    Returns DataFrame: date, series columns (one per macro indicator).
    """
    if api_key is None:
        api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        log.warning("fred_no_api_key",
                     msg="Set FRED_API_KEY in .env (free at https://fred.stlouisfed.org/docs/api/api_key.html)")
        return pd.DataFrame()

    # Fix SSL on macOS — fredapi uses urllib, needs certifi certs
    if "SSL_CERT_FILE" not in os.environ:
        try:
            import certifi
            os.environ["SSL_CERT_FILE"] = certifi.where()
        except ImportError:
            pass

    from fredapi import Fred
    fred = Fred(api_key=api_key)

    if end_date is None:
        end_date = date.today().isoformat()

    series_data: dict[str, pd.Series] = {}
    for series_id, name in MACRO_SERIES.items():
        try:
            data = fred.get_series(
                series_id, observation_start=start_date, observation_end=end_date)
            series_data[name] = data
            log.info("fred_series_ok", series=series_id, name=name, rows=len(data))
        except Exception as e:
            log.warning("fred_series_error", series=series_id, error=str(e))

    if not series_data:
        return pd.DataFrame()

    result = pd.DataFrame(series_data)
    result.index.name = "date"
    result = result.reset_index()
    result["date"] = pd.to_datetime(result["date"])
    # Forward-fill macro data (releases are weekly/monthly, need daily)
    result = result.sort_values("date")
    result = result.ffill()
    result["source"] = "fred"

    log.info("fred_download_complete", series=len(series_data), rows=len(result))
    return result


def compute_macro_features(macro_df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived macro/regime features.

    Features:
    - yield_curve_inverted: binary (T10Y2Y < 0)
    - vix_regime: low/medium/high based on percentiles
    - credit_stress: high_yield_spread z-score
    - rate_direction_3m: 3-month change in fed funds rate
    """
    if macro_df.empty:
        return macro_df

    df = macro_df.copy()

    if "yield_curve_10y2y" in df.columns:
        df["yield_curve_inverted"] = (df["yield_curve_10y2y"] < 0).astype(int)

    if "vix" in df.columns:
        df["vix_regime"] = pd.cut(
            df["vix"],
            bins=[0, 15, 25, 100],
            labels=["low", "medium", "high"],
        )

    if "high_yield_spread" in df.columns:
        rolling_mean = df["high_yield_spread"].rolling(252, min_periods=60).mean()
        rolling_std = df["high_yield_spread"].rolling(252, min_periods=60).std()
        df["credit_stress_zscore"] = (
            (df["high_yield_spread"] - rolling_mean) / rolling_std.replace(0, float("nan")))

    if "fed_funds_rate" in df.columns:
        df["rate_direction_3m"] = df["fed_funds_rate"].diff(63)  # ~3 months of trading days

    return df
