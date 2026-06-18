"""Download a mid-large-cap ($2B-$50B) universe + 5y adjusted daily OHLCV.

SEPARATE experiment from the small-cap swing in production — nothing here touches
the production universe/model/pipeline. Goal: test whether the ranker's short-
horizon (overnight) edge, which existed GROSS on small-caps but died on small-cap
spreads (~2%/day round-trip), SURVIVES net of the far tighter large-cap spreads.

Strategy (cheap + correct):
  1. grouped_daily (1 call) for the latest trading day -> whole market close+vol.
  2. Liquidity pre-filter: price >= $5, rank by dollar volume, take top N as the
     candidate pool (every $2B+ name is liquid, so this captures the band).
  3. get_ticker_details per candidate -> keep $2B-$50B, CS/ADRC, non-OTC, drop
     SPACs/funds (SIC + name). ~N calls.
  4. get_custom_bars (ADJUSTED, like production) 2021-01-01->today for survivors
     + SPY benchmark.

Survivorship caveat: discovery is from the CURRENT listing, so this v1 has mild
survivorship bias. For large-caps that bias is far smaller than for small-caps
(they rarely delist to zero; acquisitions happen at a premium). If the net edge
looks real we redo discovery point-in-time. Run while the paid plan is active
(50/min) -> ~1 h. Outputs under data/largecap/.

Usage:
    PYTHONPATH=src python scripts/largecap/download_largecap.py
"""
from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from app.data.massive import AggregatesAPI, MassiveClient, ReferenceAPI  # noqa: E402

OUT_DIR = ROOT / "data" / "largecap"
UNIV_FP = OUT_DIR / "universe_largecap.parquet"
OHLCV_FP = OUT_DIR / "ohlcv_largecap.parquet"

CAP_MIN = 2_000_000_000      # $2B
CAP_MAX = 50_000_000_000     # $50B
MIN_PRICE = 5.0              # large-caps are not penny stocks
POOL_SIZE = 2000             # top-N by dollar volume to check caps on
TRAIN_START = "2021-01-01"   # match production history window

OTC_EXCHANGES = {"OTCBB", "GREY", "PINK", "OTCQB", "OTCQX"}
EXCLUDED_SICS = {"6726", "6722", "6770", "6795"}  # funds, SPACs, trusts
EXCLUDED_NAME_PATTERNS = [
    "acquisition corp", "spac", "blank check", " fund ", "fund,",
    "opportunities fund", "income fund", "total return fund",
    "convertible fund", "municipal fund", " etf", " trust",
]


def latest_trading_day(aggs: AggregatesAPI) -> tuple[date, list]:
    """Step back from yesterday until grouped_daily returns rows."""
    d = date.today() - timedelta(days=1)
    for _ in range(10):
        if d.weekday() < 5:
            bars = aggs.get_grouped_daily(d, adjusted=True)
            if bars:
                return d, bars
        d -= timedelta(days=1)
    raise RuntimeError("no grouped_daily data in the last 10 days")


def discover(ref: ReferenceAPI, pool: list[str]) -> pd.DataFrame:
    rows, checked = [], 0
    t0 = time.time()
    for t in pool:
        detail = ref.get_ticker_details(t)
        checked += 1
        if detail is None or detail.market_cap is None:
            continue
        mcap = detail.market_cap
        if mcap < CAP_MIN or mcap > CAP_MAX:
            continue
        if detail.type not in ("CS", "ADRC"):
            continue
        exch = detail.primary_exchange or ""
        if exch in OTC_EXCHANGES:
            continue
        if (detail.sic_code or "") in EXCLUDED_SICS:
            continue
        name = detail.name or ""
        if any(p in name.lower() for p in EXCLUDED_NAME_PATTERNS):
            continue
        rows.append({
            "ticker": t, "name": name, "market_cap": mcap,
            "sic_code": detail.sic_code or "", "exchange": exch,
            "type": detail.type, "active": detail.active,
        })
        if len(rows) % 25 == 0:
            print(f"    ... {checked} checked, {len(rows)} in band "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
    df = pd.DataFrame(rows).sort_values("market_cap", ascending=False)
    print(f"  Discovered {len(df)} names in ${CAP_MIN/1e9:.0f}B-${CAP_MAX/1e9:.0f}B "
          f"(checked {checked} candidates, {(time.time()-t0)/60:.1f} min)")
    return df


def download_bars(aggs: AggregatesAPI, tickers: list[str], existing) -> pd.DataFrame:
    have = set(existing["ticker"].unique()) if existing is not None else set()
    all_dfs, failed = [], []
    t0 = time.time()
    todo = [t for t in tickers if t not in have]
    print(f"  Downloading bars for {len(todo)} tickers ({len(have)} already present)")
    for i, t in enumerate(todo, 1):
        bars = aggs.get_custom_bars(t, from_date=TRAIN_START,
                                    to_date=date.today().isoformat(), adjusted=True)
        if not bars or len(bars) < 30:
            failed.append(t)
            continue
        all_dfs.append(pd.DataFrame([{
            "date": pd.Timestamp(b.trading_date), "ticker": b.ticker,
            "open": b.open, "high": b.high, "low": b.low,
            "close": b.close, "volume": b.volume,
            "vwap": b.vwap, "transactions": b.transactions,
        } for b in bars]))
        if i % 25 == 0:
            el = (time.time() - t0) / 60
            eta = el / i * (len(todo) - i)
            print(f"    {i}/{len(todo)} ({el:.1f} min, ETA ~{eta:.0f} min)", flush=True)
    new = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    if failed:
        print(f"  WARN {len(failed)} failed/insufficient: {', '.join(failed[:20])}"
              f"{'...' if len(failed) > 20 else ''}")
    if existing is not None and not new.empty:
        new = pd.concat([existing, new], ignore_index=True)
    elif existing is not None:
        new = existing
    return new.drop_duplicates(["date", "ticker"], keep="last").sort_values(
        ["ticker", "date"]).reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = MassiveClient(calls_per_minute=50)
    aggs, ref = AggregatesAPI(client), ReferenceAPI(client)

    # ── Universe (cached if already built) ──
    if UNIV_FP.exists():
        uni = pd.read_parquet(UNIV_FP)
        print(f"Universe cached: {len(uni)} names")
    else:
        day, bars = latest_trading_day(aggs)
        cand = pd.DataFrame([{"ticker": b.ticker, "close": b.close,
                              "dvol": b.close * b.volume} for b in bars])
        cand = cand[(cand["close"] >= MIN_PRICE) & (cand["dvol"] > 0)]
        pool = cand.sort_values("dvol", ascending=False).head(POOL_SIZE)["ticker"].tolist()
        print(f"Latest trading day {day}: {len(bars)} tickers -> "
              f"{len(pool)} liquid candidates (price>=${MIN_PRICE:.0f})")
        uni = discover(ref, pool)
        uni["as_of_date"] = date.today().isoformat()
        uni.to_parquet(UNIV_FP, index=False)
        print(f"Saved universe -> {UNIV_FP}")

    # ── OHLCV (resumable) ──
    existing = pd.read_parquet(OHLCV_FP) if OHLCV_FP.exists() else None
    tickers = sorted(uni["ticker"].tolist()) + ["SPY"]
    ohlcv = download_bars(aggs, tickers, existing)
    ohlcv.to_parquet(OHLCV_FP, index=False)
    client.close()
    print(f"\nSaved OHLCV -> {OHLCV_FP}")
    print(f"  {len(ohlcv):,} rows, {ohlcv['ticker'].nunique()} tickers, "
          f"{ohlcv['date'].min().date()} -> {ohlcv['date'].max().date()}")


if __name__ == "__main__":
    main()
