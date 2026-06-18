"""Build the feature matrix for the large-cap experiment universe.

Mirrors the production feature build (build_feature_matrix + _add_lag_features)
but on the large-cap OHLCV, with no fundamentals / EDGAR / sentiment (not
available for this universe and only marginal anyway). SPY is split out as the
market_df; sectors come from the universe SIC codes.

Outputs data/largecap/features_largecap.parquet.

Usage:
    PYTHONPATH=src python scripts/largecap/build_features_largecap.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_smallcap_pipeline import _add_lag_features  # noqa: E402

from app.features.pipeline import build_feature_matrix  # noqa: E402

OUT_DIR = ROOT / "data" / "largecap"
UNIV_FP = OUT_DIR / "universe_largecap.parquet"
OHLCV_FP = OUT_DIR / "ohlcv_largecap.parquet"
FEAT_FP = OUT_DIR / "features_largecap.parquet"


def main() -> None:
    t0 = time.time()
    uni = pd.read_parquet(UNIV_FP)
    ohlcv = pd.read_parquet(OHLCV_FP)
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    spy = ohlcv[ohlcv["ticker"] == "SPY"].copy()
    panel = ohlcv[ohlcv["ticker"] != "SPY"].copy()
    print(f"  panel: {len(panel):,} rows, {panel['ticker'].nunique()} tickers; "
          f"SPY: {len(spy):,} rows")

    verified = uni[["ticker", "sic_code", "name"]].to_dict("records")
    feats = build_feature_matrix(
        panel, fundamentals=None, market_df=spy,
        universe=verified, horizons=[1, 5, 10, 20],
    )
    feats = _add_lag_features(feats)

    feats.to_parquet(FEAT_FP, index=False)
    print(f"\nSaved features -> {FEAT_FP}")
    print(f"  {len(feats):,} rows, {feats['ticker'].nunique()} tickers, "
          f"{feats['date'].min().date()} -> {feats['date'].max().date()}, "
          f"{(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
