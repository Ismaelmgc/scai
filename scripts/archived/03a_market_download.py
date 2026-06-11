"""V3 Step S1.1 — Download market regime indices via Yahoo.

Indices used as regime proxies:
  - SPY: S&P 500 (already downloaded → re-use existing)
  - IWM: Russell 2000 (small-cap proxy, most relevant for us)
  - QQQ: Nasdaq 100 (growth/tech)
  - ^VIX: volatility index
  - ^TNX: 10Y yield (^TNX)
  - HYG: high-yield bond ETF (credit spread proxy)
  - LQD: investment-grade bond ETF
  - TLT: 20+Y Treasury (duration)
  - DX-Y.NYB: US dollar index
  - XLF, XLE, XLV, XLK, XLI: sector ETFs (financials, energy, healthcare, tech, industrials)

Output: data/processed/market_indices.parquet
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import pandas as pd
import yfinance as yf

INDICES = [
    "SPY", "IWM", "QQQ",
    "^VIX", "^TNX", "DX-Y.NYB",
    "HYG", "LQD", "TLT",
    "XLF", "XLE", "XLV", "XLK", "XLI",
]


def main() -> None:
    out = Path("data/processed/market_indices.parquet")
    print(f"Downloading {len(INDICES)} indices from Yahoo (2018-01-01 → today)...")
    frames = []
    for sym in INDICES:
        try:
            df = yf.download(sym, start="2018-01-01", auto_adjust=True,
                             progress=False, threads=False)
            if df is None or df.empty:
                print(f"  {sym}: NO DATA")
                continue
            df = df.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df = df.rename(columns={"Date": "date", "Open": "open", "High": "high",
                                    "Low": "low", "Close": "close", "Volume": "volume"})
            df["ticker"] = sym
            df = df[["date", "ticker", "open", "high", "low", "close", "volume"]]
            frames.append(df)
            print(f"  {sym}: {len(df)} rows  {df.date.min().date()} → {df.date.max().date()}")
            time.sleep(0.2)
        except Exception as e:
            print(f"  {sym}: FAIL {e}")
    all_idx = pd.concat(frames, ignore_index=True)
    all_idx["date"] = pd.to_datetime(all_idx["date"])
    all_idx.to_parquet(out, index=False)
    print(f"\nSaved {out}  shape={all_idx.shape}")


if __name__ == "__main__":
    main()
