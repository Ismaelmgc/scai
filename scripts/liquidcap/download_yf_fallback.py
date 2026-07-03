"""LIQUIDCAP — yfinance fallback for CURRENT S&P 500 members that the Tiingo
free tier couldn't cover (monthly unique-symbol cap ~500).

Safe division of labor: removed/delisted names MUST come from Tiingo (yfinance
has no delisted coverage) and were downloaded first; current members are
exactly what yfinance serves correctly. Both sources are dividend+split
adjusted (Tiingo adj*, yfinance auto_adjust=True) -> consistent panel.

Appends to data/liquidcap/ohlcv_sp500.parquet, skipping tickers already there.

Usage:
    PYTHONPATH=src python scripts/liquidcap/download_yf_fallback.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from app.data.free_sources.yahoo import download_yahoo_ohlcv  # noqa: E402

DATA = ROOT / "data" / "liquidcap"
OUT = DATA / "ohlcv_sp500.parquet"


def main() -> None:
    t0 = time.time()
    mem = pd.read_parquet(DATA / "membership_sp500.parquet")
    current = set(mem[mem["end"] >= mem["end"].max()]["ticker"])
    panel = pd.read_parquet(OUT)
    done = set(panel.ticker.unique())
    todo = sorted(current - done)
    missing_removed = sorted(set(mem.ticker) - current - done)
    if missing_removed:
        print(f"  WARNING: {len(missing_removed)} REMOVED names missing and yfinance "
              f"can't serve them (survivorship hole): {missing_removed[:15]}")
    if not todo:
        print("  nothing to do — all current members present")
        return
    print(f"  fetching {len(todo)} current members via yfinance")
    df = download_yahoo_ohlcv(todo, start_date="2014-01-01")
    df = df[["date", "ticker", "open", "high", "low", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["close_unadj"] = np.nan  # raw close unavailable via yfinance auto_adjust
    merged = pd.concat([panel, df], ignore_index=True)
    merged.to_parquet(OUT, index=False)
    print(f"  saved: {len(merged):,} rows, {merged.ticker.nunique()} tickers "
          f"({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
