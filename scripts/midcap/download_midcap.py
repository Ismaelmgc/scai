"""Download a mid-cap ($2-10B) universe + 5y adjusted OHLCV for the #6 experiment:
does the production 20d swing edge hold on mid-caps (less arbitraged than large,
less manipulable + lower-cost than small, and short-borrow is feasible there → a
future market-neutral L/S could capture the +5% long-short spread from #5)?

Data sourcing under the FREE Polygon plan (paid expired → only 2y of aggregates):
  - 1 grouped_daily call (free, recent day OK) → liquid candidate pool.
  - market cap + 5y bars from YFINANCE (free, multi-year, auto-adjusted). yfinance
    is only unfit for the AUTOMATED daily cron (IP-blocks on repeat); a one-off
    research backfill is fine. fast_info.market_cap for the cap filter, then
    download_yahoo_ohlcv for history.

Resumable: saves universe_midcap.parquet after discovery, ohlcv_midcap.parquet
incrementally. Outputs under data/midcap/ (gitignored).

Usage:
    PYTHONPATH=src python scripts/midcap/download_midcap.py
"""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from app.data.free_sources.yahoo import download_yahoo_ohlcv  # noqa: E402
from app.data.massive import AggregatesAPI, MassiveClient  # noqa: E402

OUT = ROOT / "data" / "midcap"
UNIV_FP = OUT / "universe_midcap.parquet"
OHLCV_FP = OUT / "ohlcv_midcap.parquet"
CAP_MIN, CAP_MAX = 2_000_000_000, 10_000_000_000
MIN_PRICE, POOL_SIZE = 5.0, 1400
START = "2021-01-01"


def latest_grouped(aggs):
    from datetime import timedelta
    d = date.today() - timedelta(days=1)
    for _ in range(8):
        if d.weekday() < 5:
            bars = aggs.get_grouped_daily(d, adjusted=True)
            if bars:
                return bars
        d -= timedelta(days=1)
    return []


def discover_caps(tickers: list[str]) -> pd.DataFrame:
    import yfinance as yf
    rows, t0 = [], time.time()
    for i, tk in enumerate(tickers, 1):
        try:
            fi = yf.Ticker(tk).fast_info
            cap = getattr(fi, "market_cap", None)
        except Exception:
            cap = None
        if cap and CAP_MIN <= cap <= CAP_MAX:
            rows.append({"ticker": tk, "market_cap": float(cap)})
        if i % 100 == 0:
            print(f"    ... {i}/{len(tickers)} checked, {len(rows)} in $2-10B "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
        time.sleep(0.15)
    return pd.DataFrame(rows).sort_values("market_cap", ascending=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if UNIV_FP.exists():
        uni = pd.read_parquet(UNIV_FP)
        print(f"Universe cached: {len(uni)} mid-caps")
    else:
        client = MassiveClient(calls_per_minute=5)
        bars = latest_grouped(AggregatesAPI(client))
        client.close()
        cand = pd.DataFrame([{"ticker": b.ticker, "close": b.close,
                              "dvol": b.close * b.volume} for b in bars])
        cand = cand[(cand["close"] >= MIN_PRICE) & (cand["dvol"] > 0)]
        pool = cand.sort_values("dvol", ascending=False).head(POOL_SIZE)["ticker"].tolist()
        print(f"{len(bars)} tickers -> {len(pool)} liquid candidates; caps via yfinance...")
        uni = discover_caps(pool)
        uni.to_parquet(UNIV_FP, index=False)
        print(f"Saved {len(uni)} mid-caps ($2-10B) -> {UNIV_FP}")

    existing = pd.read_parquet(OHLCV_FP) if OHLCV_FP.exists() else None
    have = set(existing["ticker"].unique()) if existing is not None else set()
    todo = [t for t in uni["ticker"] if t not in have]
    print(f"Downloading 5y bars (yfinance) for {len(todo)} tickers ({len(have)} present)...")
    if todo:
        new = download_yahoo_ohlcv(todo, start_date=START, end_date=date.today().isoformat())
        if existing is not None and not new.empty:
            new = pd.concat([existing, new], ignore_index=True)
        elif existing is not None:
            new = existing
        new = new.drop_duplicates(["date", "ticker"], keep="last").sort_values(["ticker", "date"])
        new.to_parquet(OHLCV_FP, index=False)
        dts = pd.to_datetime(new["date"])
        print(f"\nSaved OHLCV -> {OHLCV_FP}: {len(new):,} rows, {new['ticker'].nunique()} "
              f"tickers, {dts.min().date()} -> {dts.max().date()}")


if __name__ == "__main__":
    main()
