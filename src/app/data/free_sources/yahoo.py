"""Yahoo Finance connector — historical OHLCV backfill.

Uses yfinance for free historical daily data going back 5-20+ years.
NOT a replacement for Massive/Polygon — used for historical extension only.

Limitations:
- No delisted ticker coverage (confirmed: WISH, IRNT, BGFV return 0 rows)
- Adjusted prices only (auto_adjust=True)
- Rate limits are informal (be nice, add delays)
- Data quality vs Massive: median diff 0.000% in overlap period (excellent)
"""
from __future__ import annotations

import time
from datetime import date

import pandas as pd
import yfinance as yf

from app.utils import get_logger

log = get_logger(__name__)


def download_yahoo_ohlcv(
    tickers: list[str],
    start_date: str = "2019-01-01",
    end_date: str | None = None,
    delay: float = 0.3,
    max_tickers: int | None = None,
) -> pd.DataFrame:
    """Download historical OHLCV from Yahoo Finance.

    Parameters
    ----------
    tickers : List of ticker symbols.
    start_date : Earliest date to fetch.
    end_date : Latest date (default: today).
    delay : Seconds between requests (rate limiting).
    max_tickers : Limit number of tickers (for testing).

    Returns
    -------
    DataFrame with columns: date, ticker, open, high, low, close, volume, source.
    """
    if max_tickers:
        tickers = tickers[:max_tickers]
    if end_date is None:
        end_date = date.today().isoformat()

    all_rows: list[pd.DataFrame] = []
    errors: list[str] = []

    for i, ticker in enumerate(tickers):
        try:
            t = yf.Ticker(ticker)
            hist = t.history(start=start_date, end=end_date, auto_adjust=True)
            if hist.empty:
                errors.append(ticker)
                continue

            df = hist.reset_index()
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={"date": "date"})
            # Remove timezone info
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            df["ticker"] = ticker
            df["source"] = "yahoo"
            df["adjusted"] = True

            df = df[["date", "ticker", "open", "high", "low", "close", "volume", "source", "adjusted"]]
            all_rows.append(df)

            if (i + 1) % 20 == 0:
                log.info("yahoo_progress", downloaded=i + 1, total=len(tickers))

        except Exception as e:
            log.warning("yahoo_ticker_error", ticker=ticker, error=str(e))
            errors.append(ticker)

        if delay > 0:
            time.sleep(delay)

    if errors:
        log.warning("yahoo_errors", count=len(errors), tickers=errors[:10])

    if not all_rows:
        return pd.DataFrame()

    result = pd.concat(all_rows, ignore_index=True)
    log.info("yahoo_download_complete",
             tickers=len(tickers) - len(errors),
             rows=len(result),
             date_range=f"{result['date'].min().date()} → {result['date'].max().date()}")
    return result


def reconcile_with_massive(
    yahoo_df: pd.DataFrame,
    massive_df: pd.DataFrame,
    tolerance_pct: float = 0.01,
) -> pd.DataFrame:
    """Compare Yahoo data with Massive/Polygon in overlap period.

    Returns a quality report DataFrame with per-ticker statistics.
    """
    yahoo = yahoo_df.copy()
    massive = massive_df.copy()
    yahoo["date"] = pd.to_datetime(yahoo["date"])
    massive["date"] = pd.to_datetime(massive["date"])

    overlap_start = max(yahoo["date"].min(), massive["date"].min())
    overlap_end = min(yahoo["date"].max(), massive["date"].max())

    yahoo_overlap = yahoo[(yahoo["date"] >= overlap_start) & (yahoo["date"] <= overlap_end)]
    massive_overlap = massive[(massive["date"] >= overlap_start) & (massive["date"] <= overlap_end)]

    merged = pd.merge(
        massive_overlap[["date", "ticker", "close"]],
        yahoo_overlap[["date", "ticker", "close"]],
        on=["date", "ticker"],
        suffixes=("_massive", "_yahoo"),
    )

    if merged.empty:
        return pd.DataFrame()

    merged["close_diff_pct"] = (
        (merged["close_massive"] - merged["close_yahoo"]).abs() / merged["close_massive"]
    )

    report = merged.groupby("ticker").agg(
        overlap_days=("close_diff_pct", "count"),
        median_diff_pct=("close_diff_pct", "median"),
        max_diff_pct=("close_diff_pct", "max"),
        pct_over_tolerance=("close_diff_pct", lambda x: (x > tolerance_pct).mean()),
    ).reset_index()

    report["source_quality_score"] = 1.0 - report["pct_over_tolerance"]
    report["survivorship_risk"] = "high"  # Yahoo doesn't cover delisted
    report["has_delisted_coverage"] = False
    report["source"] = "yahoo"

    return report
