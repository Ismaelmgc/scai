"""LIQUIDCAP — download daily bars from Tiingo for every PIT S&P 500 member.

Removed/delisted names are downloaded FIRST: they are the anti-survivorship
value and Tiingo is our only free source for them (yfinance can't). Resumable:
tickers already in the output parquet are skipped, progress saved every 50.
Uses dividend+split adjusted fields (adj*) -> total-return-consistent panel;
raw close kept as close_unadj for price filters.

Usage:
    PYTHONPATH=src python scripts/liquidcap/download_tiingo.py [--start 2014-01-01]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "liquidcap"
OUT = DATA / "ohlcv_sp500.parquet"
MISS = DATA / "download_misses.json"

COLS = {"adjOpen": "open", "adjHigh": "high", "adjLow": "low",
        "adjClose": "close", "adjVolume": "volume", "close": "close_unadj"}


def token() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("TIINGO_TOKEN"):
            return line.split("=", 1)[1].strip()
    raise SystemExit("TIINGO_TOKEN not in .env")


def fetch(tkr: str, start: str, tok: str) -> pd.DataFrame | None:
    """Free tier is ~50 req/hour: on 429 wait 15 min and retry (an hourly
    window always rolls within 6 attempts = 90 min). A 429 persisting past
    that means the MONTHLY unique-symbol cap -> stop gracefully."""
    url = f"https://api.tiingo.com/tiingo/daily/{tkr.lower()}/prices"
    for attempt in range(7):
        r = requests.get(url, params={"startDate": start, "token": tok},
                         headers={"Content-Type": "application/json"}, timeout=60)
        if r.status_code == 200:
            js = r.json()
            if not js:
                return None
            df = pd.DataFrame(js)
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
            # select BEFORE renaming: raw open/high/low/volume also exist and
            # would collide with the renamed adj* columns
            df = df[["date"] + list(COLS)].rename(columns=COLS)
            df["ticker"] = tkr
            return df
        if r.status_code == 404:
            return None
        if r.status_code in (429, 403):
            print(f"    {tkr}: HTTP {r.status_code}, backoff 15 min "
                  f"(attempt {attempt + 1}/7)", flush=True)
            time.sleep(900)
        else:
            print(f"    {tkr}: HTTP {r.status_code} {r.text[:80]}", flush=True)
            time.sleep(5)
    raise RuntimeError(f"rate-limited beyond 90 min at {tkr} — likely the "
                       f"monthly symbol cap; run the yfinance fallback for the rest")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2014-01-01")
    args = ap.parse_args()
    tok = token()
    t0 = time.time()

    mem = pd.read_parquet(DATA / "membership_sp500.parquet")
    today_members = set(mem[mem["end"] >= mem["end"].max()]["ticker"])
    removed = sorted(set(mem.ticker) - today_members)
    order = removed + sorted(today_members) + ["SPY"]

    done: set[str] = set()
    frames: list[pd.DataFrame] = []
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        done = set(prev.ticker.unique())
        frames = [prev]
        print(f"  resume: {len(done)} tickers already downloaded")
    misses = json.loads(MISS.read_text()) if MISS.exists() else []

    todo = [t for t in order if t not in done]
    print(f"  downloading {len(todo)} tickers (removed first), start {args.start}")
    new: list[pd.DataFrame] = []
    for i, tkr in enumerate(todo, 1):
        try:
            df = fetch(tkr, args.start, tok)
        except RuntimeError as e:
            print(f"  STOPPING (limits): {e}", flush=True)
            break
        if df is None:
            misses.append(tkr)
        else:
            new.append(df)
        if i % 50 == 0 or i == len(todo):
            if new:
                frames.append(pd.concat(new, ignore_index=True))
                new = []
                pd.concat(frames, ignore_index=True).to_parquet(OUT, index=False)
            MISS.write_text(json.dumps(misses))
            got = sum(len(f) for f in frames)
            print(f"  [{i}/{len(todo)}] rows={got:,} misses={len(misses)} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
        time.sleep(1.0)

    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not panel.empty:
        panel.to_parquet(OUT, index=False)
        spy = panel[panel.ticker == "SPY"]
        if not spy.empty:
            spy.to_parquet(DATA / "spy.parquet", index=False)
        print(f"\n  saved {OUT.name}: {len(panel):,} rows, {panel.ticker.nunique()} tickers, "
              f"{panel.date.min().date()} -> {panel.date.max().date()}")
        print(f"  misses ({len(misses)}): {misses[:20]}{'...' if len(misses) > 20 else ''}")
    print(f"  runtime {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
