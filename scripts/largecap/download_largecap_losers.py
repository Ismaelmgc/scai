"""Survivorship probe: download the LOSERS — names that were $2-50B in the past
(2021-2026) but are gone from today's universe (delisted, or fell below $2B).

Today's universe is current-listing, so an overnight LONG backtest never traded
the names that blew up. This script reconstructs that excluded set so we can add
them back and re-test:

  1. PIT snapshots (grouped_daily on past dates) -> liquid candidate pool then.
  2. Candidates not in today's universe -> PIT cap (ticker_details date_=) to keep
     those that were $2-50B THEN (+ CS/ADRC, non-OTC, non-SPAC).
  3. Current status to isolate true losers (exclude winners that grew >$50B):
       delisted / not found      -> 'delisted'  (add)
       current cap < $2B         -> 'fell'      (add)
       current cap > $50B        -> winner      (SKIP, would inflate)
       current cap $2-50B        -> 'missed'    (add; should've been in universe)
  4. Download 5y adjusted OHLCV for the add-set (delisted names return partial
     history up to delisting — exactly the crash we were missing).

Outputs data/largecap/universe_losers.parquet + ohlcv_losers.parquet. Run while
the paid plan is active (50/min). Then combine + retrain to test if the edge
survives the survivorship haircut.

Usage:
    PYTHONPATH=src python scripts/largecap/download_largecap_losers.py
"""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from app.data.massive import AggregatesAPI, MassiveClient, ReferenceAPI  # noqa: E402

OUT = ROOT / "data" / "largecap"
UNIV_FP = OUT / "universe_largecap.parquet"
LOSERS_FP = OUT / "universe_losers.parquet"
OHLCV_FP = OUT / "ohlcv_losers.parquet"

CAP_MIN, CAP_MAX = 2_000_000_000, 50_000_000_000
MIN_PRICE, POOL_SIZE = 5.0, 1800
TRAIN_START = "2021-01-01"
SNAPSHOTS = ["2022-01-03", "2023-08-01"]   # pre-bust peak + later cohort

OTC = {"OTCBB", "GREY", "PINK", "OTCQB", "OTCQX"}
BAD_SIC = {"6726", "6722", "6770", "6795"}
BAD_NAME = ["acquisition corp", "spac", "blank check", " fund ", "fund,",
            "opportunities fund", "income fund", "total return fund",
            "convertible fund", "municipal fund", " etf", " trust"]


def liquid_pool(aggs: AggregatesAPI, snap: str) -> list[str]:
    bars = aggs.get_grouped_daily(snap, adjusted=True)
    if not bars:
        return []
    df = pd.DataFrame([{"ticker": b.ticker, "close": b.close,
                        "dvol": b.close * b.volume} for b in bars])
    df = df[(df["close"] >= MIN_PRICE) & (df["dvol"] > 0)]
    return df.sort_values("dvol", ascending=False).head(POOL_SIZE)["ticker"].tolist()


def ok_meta(d) -> bool:
    if d.type not in ("CS", "ADRC"):
        return False
    if (d.primary_exchange or "") in OTC:
        return False
    if (d.sic_code or "") in BAD_SIC:
        return False
    name = (d.name or "").lower()
    return not any(p in name for p in BAD_NAME)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    client = MassiveClient(calls_per_minute=50)
    aggs, ref = AggregatesAPI(client), ReferenceAPI(client)
    current = set(pd.read_parquet(UNIV_FP)["ticker"])

    if LOSERS_FP.exists():
        losers = pd.read_parquet(LOSERS_FP)
        print(f"Losers universe cached: {len(losers)}")
    else:
        cands: set[str] = set()
        for snap in SNAPSHOTS:
            pool = liquid_pool(aggs, snap)
            cands |= set(pool)
            print(f"  snapshot {snap}: {len(pool)} liquid -> running union {len(cands)}")
        cands -= current
        print(f"  {len(cands)} candidates not in current universe; classifying...")

        rows, checked = [], 0
        t0 = time.time()
        for t in sorted(cands):
            checked += 1
            # PIT cap: was it $2-50B at a snapshot?
            pit_cap = None
            for snap in SNAPSHOTS:
                d = ref.get_ticker_details(t, date_=date.fromisoformat(snap))
                if d and d.market_cap:
                    if CAP_MIN <= d.market_cap <= CAP_MAX and ok_meta(d):
                        pit_cap = d.market_cap
                    break
            if pit_cap is None:
                continue
            # Current status -> isolate true losers (skip winners that grew out)
            cur = ref.get_ticker_details(t)
            if cur is None or not cur.active:
                status, cur_cap = "delisted", (cur.market_cap if cur else None)
            elif cur.market_cap is None:
                status, cur_cap = "delisted", None
            elif cur.market_cap < CAP_MIN:
                status, cur_cap = "fell", cur.market_cap
            elif cur.market_cap > CAP_MAX:
                continue  # winner that grew >$50B — excluding avoids inflation
            else:
                status, cur_cap = "missed", cur.market_cap
            rows.append({"ticker": t, "name": (cur.name if cur else "") or "",
                         "pit_cap": pit_cap, "current_cap": cur_cap,
                         "status": status,
                         "sic_code": (cur.sic_code if cur else "") or ""})
            if len(rows) % 25 == 0:
                print(f"    ... {checked}/{len(cands)} checked, {len(rows)} losers "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
        losers = pd.DataFrame(rows)
        losers.to_parquet(LOSERS_FP, index=False)
        print(f"\nSaved {len(losers)} losers -> {LOSERS_FP}")
        print(losers["status"].value_counts().to_string())

    # ── OHLCV for losers (resumable) ──
    existing = pd.read_parquet(OHLCV_FP) if OHLCV_FP.exists() else None
    have = set(existing["ticker"].unique()) if existing is not None else set()
    todo = [t for t in losers["ticker"] if t not in have]
    print(f"\nDownloading bars for {len(todo)} losers ({len(have)} present)")
    dfs, failed = [], []
    t0 = time.time()
    for i, t in enumerate(todo, 1):
        bars = aggs.get_custom_bars(t, from_date=TRAIN_START,
                                    to_date=date.today().isoformat(), adjusted=True)
        if not bars or len(bars) < 30:
            failed.append(t)
            continue
        dfs.append(pd.DataFrame([{
            "date": pd.Timestamp(b.trading_date), "ticker": b.ticker,
            "open": b.open, "high": b.high, "low": b.low,
            "close": b.close, "volume": b.volume,
            "vwap": b.vwap, "transactions": b.transactions} for b in bars]))
        if i % 25 == 0:
            el = (time.time() - t0) / 60
            eta = el / i * (len(todo) - i)
            print(f"    {i}/{len(todo)} ({el:.1f} min, ETA ~{eta:.0f} min)", flush=True)
    new = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if existing is not None and not new.empty:
        new = pd.concat([existing, new], ignore_index=True)
    elif existing is not None:
        new = existing
    if not new.empty:
        new = new.drop_duplicates(["date", "ticker"], keep="last").sort_values(
            ["ticker", "date"]).reset_index(drop=True)
        new.to_parquet(OHLCV_FP, index=False)
    client.close()
    if failed:
        print(f"  WARN {len(failed)} failed/insufficient")
    print(f"\nSaved OHLCV -> {OHLCV_FP}: {len(new):,} rows, "
          f"{new['ticker'].nunique() if not new.empty else 0} tickers")


if __name__ == "__main__":
    main()
