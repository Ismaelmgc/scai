#!/usr/bin/env python3
"""Monthly small-cap universe refresh.

The daily pipeline only *reads* the universe — it never adds new names or
re-checks delistings, so the tradable set slowly erodes and misses new
entrants. This job, run monthly, keeps the universe current:

  1. Reuses every existing ticker (delisted ones stay for anti-survivorship —
     they are NEVER dropped; the OHLCV history must keep them).
  2. Adds up to MAX_NEW_PER_REFRESH new entrants that are now in the
     $50M-$2B band (via `discover_universe`, point-in-time market caps).
  3. Refreshes the `active`/delisted flag of every ticker from Polygon's
     current listing, so newly delisted names stop being downloaded/selected.

New entrants have no OHLCV yet; the next daily run backfills their full history
automatically (download_ohlcv does a full pull for tickers with no bars).

Writes data/processed/smallcap_universe.parquet (+ refreshes the cached
ticker catalog). Run via the monthly Actions workflow or:

    PYTHONPATH=src python scripts/refresh_universe.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_smallcap_pipeline import discover_universe  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.data.massive import MassiveClient, ReferenceAPI  # noqa: E402
from app.data.store.parquet_store import ParquetStore  # noqa: E402

# How many new tickers to add per monthly run (bounds API calls / runtime).
MAX_NEW_PER_REFRESH = 100
TRAIN_START = "2021-01-01"  # delisted candidates eligible if they left after this


def _current_active_tickers(ref: ReferenceAPI) -> set[str]:
    """Current active US common stocks + ADRs (independent of any cache)."""
    active: set[str] = set()
    for ticker_type in ("CS", "ADRC"):
        for t in ref.list_tickers(
            market="stocks", locale="us", ticker_type=ticker_type,
            active=True, limit=1000,
        ):
            active.add(t.ticker)
    return active


def main() -> None:
    cfg = get_settings()
    store = ParquetStore()

    existing = store.read("smallcap_universe") if store.exists("smallcap_universe") else None
    n_before = len(existing) if existing is not None else 0
    active_before = int((existing["active"] == True).sum()) if existing is not None else 0  # noqa: E712
    print(f"Universe before: {n_before} tickers ({active_before} active)")

    client = MassiveClient(calls_per_minute=50)
    ref = ReferenceAPI(client)

    # 1+2. Reuse existing + add up to MAX_NEW_PER_REFRESH new entrants.
    target = n_before + MAX_NEW_PER_REFRESH
    verified = discover_universe(
        ref, cfg, max_tickers=target, store=store,
        existing_universe=existing, train_start=TRAIN_START,
    )

    # 3. Refresh active/delisted flags from the current listing.
    active_now = _current_active_tickers(ref)
    print(f"  Polygon reports {len(active_now)} active CS/ADR tickers")
    newly_delisted = 0
    for v in verified:
        was_active = bool(v.get("active", True))
        v["active"] = v["ticker"] in active_now
        if was_active and not v["active"]:
            newly_delisted += 1

    uni_df = pd.DataFrame(verified)
    uni_df["as_of_date"] = date.today().isoformat()
    store.write("smallcap_universe", uni_df)
    client.close()

    n_after = len(uni_df)
    active_after = int((uni_df["active"] == True).sum())  # noqa: E712
    existing_set = set(existing["ticker"]) if existing is not None else set()
    n_new = len([t for t in uni_df["ticker"] if t not in existing_set])
    print(
        f"Universe after:  {n_after} tickers ({active_after} active) | "
        f"+{n_new} new entrants | {newly_delisted} newly delisted"
    )
    print("Next daily run will backfill OHLCV for the new entrants.")


if __name__ == "__main__":
    main()
