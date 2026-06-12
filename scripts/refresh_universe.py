#!/usr/bin/env python3
"""Monthly small-cap universe refresh (fixed-size replacement model).

The daily pipeline only *reads* the universe — it never re-checks delistings or
adds new names, so the tradable set slowly erodes. This job, run monthly, keeps
it current WITHOUT growing the active set (the active count drives the daily
OHLCV API calls, which are rate-limited — especially on a free Polygon plan):

  1. Reuses every existing ticker. Delisted ones are KEPT (marked inactive) for
     anti-survivorship — they are never dropped, and being inactive they are not
     downloaded daily, so they cost no API calls.
  2. Refreshes the active/delisted flag of every ticker from Polygon's current
     listing.
  3. Refills the ACTIVE set back up to TARGET_ACTIVE by adding only as many new
     entrants ($50M-$2B band) as were lost to delisting. The active set stays
     ~constant, so daily API usage stays bounded.

New entrants have no OHLCV yet; the next daily run backfills their full history
(download_ohlcv full-pulls tickers with no bars).

Writes data/processed/smallcap_universe.parquet. Run via the monthly Actions
workflow or:

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

# Cap on the ACTIVE (daily-downloaded) set. This bounds daily API calls — lower
# it if you move to a rate-limited / free plan. The active set is refilled up to
# this number but never grown beyond it.
TARGET_ACTIVE = 320
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

    # Current active listing — drives both the flag refresh and the refill budget.
    active_now = _current_active_tickers(ref)
    print(f"  Polygon reports {len(active_now)} active CS/ADR tickers")

    existing_tickers = set(existing["ticker"]) if existing is not None else set()
    still_active = len(existing_tickers & active_now)
    needed = max(0, TARGET_ACTIVE - still_active)
    print(f"  {still_active} existing still active; refilling {needed} to reach "
          f"TARGET_ACTIVE={TARGET_ACTIVE}")

    # Add exactly `needed` new entrants (reusing existing, delisted included).
    target_total = n_before + needed
    verified = discover_universe(
        ref, cfg, max_tickers=target_total, store=store,
        existing_universe=existing, train_start=TRAIN_START,
    )

    # Refresh active/delisted flags from the current listing.
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
    n_new = len([t for t in uni_df["ticker"] if t not in existing_tickers])
    print(
        f"Universe after:  {n_after} tickers ({active_after} active) | "
        f"+{n_new} new entrants | {newly_delisted} newly delisted"
    )
    print("Next daily run will backfill OHLCV for the new entrants.")


if __name__ == "__main__":
    main()
