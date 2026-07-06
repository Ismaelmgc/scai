"""Build the feature matrix for the mid-cap ($2-10B) swing experiment (#6).

Mirrors the production feature build (build_feature_matrix + _add_lag_features) on
the yfinance mid-cap OHLCV. SPY (also yfinance) is the market_df for beta/regime.
No EDGAR (uses the 26 base features). First pass uses "Unknown" sectors (the
yfinance universe has no SIC), so the sector-relative target collapses to a clean
cross-sectional (date-relative) target and 2-3 sector features go degenerate — a
conservative first test; real sectors would only help.

Outputs data/midcap/features_midcap.parquet.

Usage:
    PYTHONPATH=src python scripts/midcap/build_features_midcap.py
"""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_smallcap_pipeline import _add_lag_features  # noqa: E402

from app.data.free_sources.yahoo import download_yahoo_ohlcv  # noqa: E402
from app.features.pipeline import build_feature_matrix  # noqa: E402

DATA = ROOT / "data" / "midcap"
OHLCV_FP = DATA / "ohlcv_midcap.parquet"
SPY_FP = DATA / "spy_midcap.parquet"
FEAT_FP = DATA / "features_midcap.parquet"


def main() -> None:
    t0 = time.time()
    panel = pd.read_parquet(OHLCV_FP)
    panel["date"] = pd.to_datetime(panel["date"])

    if SPY_FP.exists():
        spy = pd.read_parquet(SPY_FP)
    else:
        spy = download_yahoo_ohlcv(["SPY"], start_date="2021-01-01",
                                   end_date=date.today().isoformat())
        spy.to_parquet(SPY_FP, index=False)
    spy["date"] = pd.to_datetime(spy["date"])
    print(f"  panel {len(panel):,} rows / {panel['ticker'].nunique()} tickers; SPY {len(spy):,}")

    universe = [{"ticker": t} for t in panel["ticker"].unique()]  # no SIC -> Unknown sector
    feats = build_feature_matrix(panel, fundamentals=None, market_df=spy,
                                 universe=universe, horizons=[1, 5, 10, 20])
    feats = _add_lag_features(feats)
    feats.to_parquet(FEAT_FP, index=False)
    print(f"\nSaved {FEAT_FP.name}: {len(feats):,} rows, {feats['ticker'].nunique()} tickers, "
          f"{feats['date'].min().date()} -> {feats['date'].max().date()}, {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
