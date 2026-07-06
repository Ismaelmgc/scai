"""LIQUIDCAP monthly maintenance: keep the universe honest as the index evolves.

The S&P 500 turns over ~20-25 names/year (acquisitions, market-cap changes,
spin-offs). Two things must happen when a name LEAVES the index, or the
anti-survivorship property silently rots:

  1. membership must be refreshed so the daily job stops treating a departed
     name as current.
  2. the departed name's price history must be ARCHIVED into the committed
     delisted slice BEFORE yfinance drops it (yfinance has ~no delisted
     coverage — confirmed WISH/IRNT/BGFV return 0 rows). Otherwise the training
     panel slowly loses the losers and re-introduces survivorship bias — the
     exact failure that sank the mid-cap experiment.

Additions need no action: a new member is "current", so the daily job's
yfinance download already picks up its full history.

Flow:
  - snapshot the CURRENT set from the existing membership
  - refresh membership from live Wikipedia (build_universe_sp500.py)
  - departed = old current − new current
  - download each departed name's full history (yfinance while still available;
    misses are logged for a manual Tiingo backfill) and append to
    ohlcv_delisted.parquet (dedup by date,ticker)
  - sanity-gate the refresh (≈500 current) so a bad Wikipedia parse can't wipe
    the universe

Run monthly (own workflow) or on demand. Commits membership + delisted slice.

Usage:
    PYTHONPATH=src python scripts/liquidcap/maintain_universe.py [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from app.data.free_sources.yahoo import download_yahoo_ohlcv  # noqa: E402

DATA = ROOT / "data" / "liquidcap"
MEMBERSHIP_FP = DATA / "membership_sp500.parquet"
DELISTED_FP = DATA / "ohlcv_delisted.parquet"
START = "2014-01-01"


def current_set(mem: pd.DataFrame) -> set[str]:
    return set(mem[mem["end"] >= mem["end"].max()]["ticker"].unique())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="no writes")
    args = ap.parse_args()
    t0 = time.time()

    old_mem = pd.read_parquet(MEMBERSHIP_FP)
    old_current = current_set(old_mem)
    print(f"  current members before refresh: {len(old_current)}")

    # 1. Refresh membership from live Wikipedia (build_universe_sp500 overwrites
    # MEMBERSHIP_FP). In dry-run we restore the committed file afterwards.
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run([sys.executable,
                    str(ROOT / "scripts/liquidcap/build_universe_sp500.py")],
                   env=env, check=True)
    new_mem = pd.read_parquet(MEMBERSHIP_FP)
    new_current = current_set(new_mem)

    # Sanity gate: a healthy S&P 500 reconstruction has ~500 current members.
    # On a bad/failed parse, roll back to the committed membership.
    if args.dry_run or not (480 <= len(new_current) <= 520):
        old_mem.to_parquet(MEMBERSHIP_FP, index=False)
        if not (480 <= len(new_current) <= 520):
            raise SystemExit(f"  ABORT: refreshed membership has {len(new_current)} "
                             f"current members (expected ~500) — kept old membership")

    departed = sorted(old_current - new_current)
    added = sorted(new_current - old_current)
    print(f"  current after refresh: {len(new_current)}  "
          f"departed: {len(departed)}  added: {len(added)}")
    if added:
        print(f"  + added (auto-covered as current): {added}")
    if not departed:
        print("  no departures — nothing to archive")
        if args.dry_run:
            print("  DRY RUN — membership not changed")
        print(f"  Runtime {(time.time() - t0) / 60:.1f} min")
        return
    print(f"  - departed (archiving before yfinance drops them): {departed}")

    if args.dry_run:
        print("  DRY RUN — would archive departed history + commit; no writes")
        return

    # 2. Archive departed names' full history into the committed delisted slice.
    delisted = pd.read_parquet(DELISTED_FP)
    delisted["date"] = pd.to_datetime(delisted["date"])
    already = set(delisted.ticker.unique())
    to_fetch = [t for t in departed if t not in already]
    fresh = download_yahoo_ohlcv(to_fetch, start_date=START) if to_fetch else pd.DataFrame()
    misses = [t for t in to_fetch if fresh.empty or t not in set(fresh.ticker)]
    if not fresh.empty:
        fresh = fresh[["date", "ticker", "open", "high", "low", "close", "volume"]].copy()
        fresh["date"] = pd.to_datetime(fresh["date"]).dt.tz_localize(None).dt.normalize()
        fresh["close_unadj"] = float("nan")
        delisted = pd.concat([delisted, fresh], ignore_index=True)
        delisted = delisted.drop_duplicates(["date", "ticker"], keep="last")
        delisted.to_parquet(DELISTED_FP, index=False)
        print(f"  archived {fresh.ticker.nunique()} names -> delisted slice "
              f"({len(delisted):,} rows, {delisted.ticker.nunique()} tickers)")
    if misses:
        print(f"  WARNING: yfinance served no data for {misses} — "
              f"MANUAL Tiingo backfill needed (permaTicker) to keep them for training")

    print(f"  Runtime {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
