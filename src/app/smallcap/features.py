"""Small-cap feature rebuild (production path) + point-in-time EDGAR merge."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.smallcap.model import V2_EDGAR_FEATURES
from app.utils import get_logger

log = get_logger(__name__)


def rebuild_features(ohlcv: pd.DataFrame, cfg) -> pd.DataFrame:
    """Rebuild feature matrix with latest data (Yahoo backfill + Massive)."""
    from app.data.store.parquet_store import ParquetStore
    from app.features.pipeline import build_feature_matrix

    store = ParquetStore()
    uni_df = store.read("smallcap_universe")
    verified_tickers = uni_df.to_dict("records")

    # ── Combine OHLCV: Yahoo backfill (2019-2024) + Massive (2024-present) ──
    ohlcv_combined = ohlcv.copy()
    try:
        if store.exists("ohlcv_smallcap_yahoo"):
            yahoo_ohlcv = store.read("ohlcv_smallcap_yahoo")
            if yahoo_ohlcv is not None and not yahoo_ohlcv.empty:
                yahoo_ohlcv["date"] = pd.to_datetime(yahoo_ohlcv["date"])
                ohlcv_combined["date"] = pd.to_datetime(ohlcv_combined["date"])
                # Concat and deduplicate (Massive takes priority in overlap)
                ohlcv_combined = pd.concat([yahoo_ohlcv, ohlcv_combined], ignore_index=True)
                ohlcv_combined = ohlcv_combined.drop_duplicates(
                    subset=["ticker", "date"], keep="last"
                ).sort_values(["ticker", "date"])
                print(f"  ✓ Yahoo backfill: {len(yahoo_ohlcv):,} rows added "
                      f"({yahoo_ohlcv['date'].min().date()} → {yahoo_ohlcv['date'].max().date()})")
    except Exception as e:
        log.warning("yahoo_backfill_unavailable", error=str(e))

    # Load auxiliary data
    fundamentals = None
    try:
        from app.features.fundamentals import _pivot_fundamentals, compute_fundamental_features
        fund_raw = store.read("smallcap_fundamentals")
        if fund_raw is not None and not fund_raw.empty:
            fund_pivoted = _pivot_fundamentals(fund_raw)
            if not fund_pivoted.empty:
                fundamentals = compute_fundamental_features(fund_pivoted)
    except Exception as e:
        log.warning("fundamentals_unavailable", error=str(e))

    market_df = None
    try:
        spy_data = store.read("smallcap_spy")
        if spy_data is not None and not spy_data.empty:
            market_df = spy_data
    except Exception as e:
        log.warning("spy_unavailable", error=str(e))

    features = build_feature_matrix(
        ohlcv_combined, fundamentals=fundamentals, market_df=market_df,
        universe=verified_tickers, horizons=[1, 5, 10, 20],
    )

    # Add lag features
    from app.features.lags import add_lag_features
    features = add_lag_features(features)

    # Add EDGAR fundamental features (dilution_pct, current_ratio)
    features = _merge_edgar_features(features)

    store.write("features_smallcap", features)
    print(f"  ✓ Features: {len(features):,} rows × {len(features.columns)} cols")
    return features.copy()  # defragment


def _merge_edgar_features(features: pd.DataFrame) -> pd.DataFrame:
    """Merge EDGAR fundamental features (dilution_pct, current_ratio) into feature matrix.

    Uses point-in-time filing dates to prevent look-ahead bias:
    for each (ticker, date), we use the most recent filing before that date.
    """
    try:
        edgar_path = Path("data/edgar_facts.parquet")
        if not edgar_path.exists():
            return features

        from app.data.free_sources.sec_edgar import compute_edgar_features
        facts = pd.read_parquet(edgar_path)
        edgar_feat = compute_edgar_features(facts)
        if edgar_feat.empty:
            return features

        # Keep only columns we need
        keep_cols = ["ticker", "filing_date"] + [
            c for c in V2_EDGAR_FEATURES if c in edgar_feat.columns
        ]
        edgar_feat = edgar_feat[keep_cols].dropna(subset=["filing_date"]).copy()
        edgar_feat["filing_date"] = pd.to_datetime(edgar_feat["filing_date"])

        # Point-in-time merge: for each (ticker, date), use most recent filing <= date
        features["date"] = pd.to_datetime(features["date"])
        edgar_feat = edgar_feat.sort_values("filing_date").reset_index(drop=True)

        # merge_asof requires both sides sorted by the on-key globally
        features_sorted = features.sort_values("date").reset_index(drop=True)

        merged = pd.merge_asof(
            features_sorted,
            edgar_feat,
            left_on="date",
            right_on="filing_date",
            by="ticker",
            direction="backward",
        )
        # Drop the filing_date column
        merged = merged.drop(columns=["filing_date"], errors="ignore")

        first = V2_EDGAR_FEATURES[0]
        n_filled = merged[first].notna().sum() if first in merged.columns else 0
        print(f"  ✓ EDGAR features added ({n_filled:,} rows with data)")
        return merged

    except Exception as e:
        log.warning("edgar_features_failed", error=str(e))
        return features

