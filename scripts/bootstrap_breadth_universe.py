"""Bootstrap an EXPANDED small-cap universe for the breadth experiment.

Breadth ablation (scripts/v3/38) showed candidate-pool size is a first-order
driver of top-8 return AND WR, and the curve is still climbing at our current
~526-name pool. The production universe caps the ACTIVE set at 320 (daily API
budget) — but grouped-daily updates make a much larger set sustainable. To test
the UPWARD direction we need OHLCV history for tradeable small-caps we don't yet
track.

This discovers the full $50M-$2B band (reusing the existing 1048 verified names,
anti-survivorship: keeps delisted) and downloads FULL history for the new names to
RESEARCH files only. Production (smallcap_universe / ohlcv_smallcap) is untouched.
Run while Polygon is on the paid plan (50/min); the historical pull is the only
rate-limited step (daily updates would later use grouped-daily = 1 call/day).

Outputs (research-only):
  data/research_breadth/universe_expanded.parquet
  data/research_breadth/ohlcv_new.parquet     (only the newly downloaded names)

Usage:
    PYTHONPATH=src python scripts/bootstrap_breadth_universe.py            # max 3000
    PYTHONPATH=src python scripts/bootstrap_breadth_universe.py 2200
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_smallcap_pipeline import discover_universe, download_ohlcv  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.data.massive import MassiveClient, ReferenceAPI  # noqa: E402
from app.data.massive.aggregates import AggregatesAPI  # noqa: E402
from app.data.store.parquet_store import ParquetStore  # noqa: E402

OUT_DIR = ROOT / "data" / "research_breadth"
TRAIN_START = "2021-06-01"


def main() -> None:
    max_tickers = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    cfg = get_settings()
    store = ParquetStore()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    existing_uni = store.read("smallcap_universe")
    existing_tickers = set(existing_uni["ticker"].astype(str))
    ohlcv = pd.read_parquet(ROOT / "data/processed/ohlcv_smallcap.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    have_ohlcv = set(ohlcv["ticker"].astype(str))
    predict_to = ohlcv["date"].max().strftime("%Y-%m-%d")
    print(f"Existing: {len(existing_tickers)} universe / {len(have_ohlcv)} with OHLCV "
          f"| target max_tickers={max_tickers} | history {TRAIN_START}->{predict_to}")

    client = MassiveClient(calls_per_minute=50)
    ref = ReferenceAPI(client)
    aggs = AggregatesAPI(client)

    # Discover up to max_tickers verified small-caps, reusing the existing verified
    # set (so only NEW candidates cost market-cap verification calls).
    verified = discover_universe(
        ref, cfg, max_tickers=max_tickers, store=store,
        existing_universe=existing_uni, train_start=TRAIN_START,
    )
    uni_df = pd.DataFrame(verified)
    uni_df.to_parquet(OUT_DIR / "universe_expanded.parquet", index=False)
    new_tickers = [t for t in uni_df["ticker"].astype(str) if t not in have_ohlcv]
    delta = len(uni_df) - len(existing_tickers)
    print(f"\nDiscovered {len(uni_df)} verified ({delta:+d} vs existing); "
          f"{len(new_tickers)} need OHLCV history download")

    if not new_tickers:
        print("No new tickers to download.")
        client.close()
        return

    new_ohlcv = download_ohlcv(aggs, new_tickers, TRAIN_START, predict_to, existing_ohlcv=None)
    client.close()

    if new_ohlcv is None or new_ohlcv.empty:
        print("Download returned no data.")
        return
    new_ohlcv["date"] = pd.to_datetime(new_ohlcv["date"])
    new_ohlcv.to_parquet(OUT_DIR / "ohlcv_new.parquet", index=False)
    print(f"\nSaved {len(new_ohlcv):,} bars for {new_ohlcv['ticker'].nunique()} new tickers "
          f"-> {OUT_DIR/'ohlcv_new.parquet'}")
    print(f"  date range: {new_ohlcv['date'].min().date()} -> {new_ohlcv['date'].max().date()}")


if __name__ == "__main__":
    main()
