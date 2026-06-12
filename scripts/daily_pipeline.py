#!/usr/bin/env python3
"""SCAI Daily Pipeline — Automated paper trading with V3 LambdaRank model.

Model: LightGBM LambdaRank on fwd_ret_20d_sector_rel (V3 Sprint 2 candidate).
Signal: rank by predicted ranking score, top-8 equal-weight.
Retraining: every 7 days, expanding window with sector-enriched universe.

V3 metrics vs V2 baseline (16-fold walk-forward):
  Sharpe   2.94 → 3.71  (+0.77)
  WinRate  44.88% → 49.56%
  SelMed   -2.42% → +0.47%  (top-K median crossed positive)
  +folds   13/16 → 14/16

Usage:
    python scripts/daily_pipeline.py
    python scripts/daily_pipeline.py --force-retrain
    python scripts/daily_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from app.config import get_settings
from app.data import supabase_store
from app.features.tradability import tradable_mask, is_stale
from app.utils import setup_logging, set_global_seed, get_logger

log = get_logger(__name__)

# ── Constants ───────────────────────────────────────────────
RETRAIN_EVERY_DAYS = 7          # Retrain model every N calendar days
PORTFOLIO_PATH = "data/paper_trading/portfolio.json"
PORTFOLIO_PATH_ADAPTIVE = "data/paper_trading/adaptive/portfolio.json"
TRADE_LOG_PATH = "data/paper_trading/trades.parquet"
DAILY_LOG_PATH = "data/paper_trading/daily_log.jsonl"
MODEL_REGISTRY = "data/paper_trading/model_registry.json"
MODEL_PATH = "data/models/smallcap_v3_lambdarank.pkl"

# V3 model config — Sprint 2 candidate (HP-tuned LambdaRank, 16-fold WF validated)
V2_TARGET = "fwd_ret_20d_sector_rel"
V2_RAW_COL = "fwd_ret_20d"
V2_HOLD_DAYS = 20
# Same 26 base features as V2 (pruning was tested and discarded — Sprint 2 task b)
V2_FEATURES = [
    'max_dd_60d', 'vol_of_vol_60d', 'ret_kurtosis_60d', 'avg_trade_size_20d',
    'obv', 'obv_vs_sma_60d', 'amihud_60d', 'downside_vol_60d', 'ema_26',
    'spread_proxy_20d', 'ret_skew_60d', 'ret_252d', 'sma_200', 'sector_ret_60d',
    'realized_vol_120d', 'macd_hist', 'pct_from_52w_low', 'adv_60d',
    'ret_vs_sector_60d', 'vol_of_vol_ratio', 'price_roc_smooth_120d',
    'vwap_dev_avg_20d', 'reversal_20v60', 'vol_regime_change', 'beta_60d',
    'macd_signal',
]
# Number of relevance bins for LambdaRank (per-date qcut)
V3_N_BINS = 16
V2_LGB_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg",
    "num_leaves": 31, "max_depth": 6,
    "learning_rate": 0.05, "min_child_samples": 30,
    "subsample": 0.75, "colsample_bytree": 0.7,
    "reg_lambda": 5.0,
    "lambdarank_truncation_level": 8,
    "label_gain": list(range(V3_N_BINS)),
    "n_jobs": 1, "seed": 42, "verbose": -1,
}
V3_NUM_BOOST_ROUND = 600
V2_TOP_K = 8
V2_REBALANCE_FREQ = 5  # days between rebalances

# EDGAR fundamental features (validated IC: dilution_pct=-0.029***, current_ratio=-0.021***)
V2_EDGAR_FEATURES = [
    'dilution_pct',
    'current_ratio',
]

# Meta-learning: REMOVED 2026-05-22 (A/B test showed no gain when leak-free)
V2_META_FEATURES: list[str] = []
SIGNAL_HISTORY_BACKTEST = "data/paper_trading/signal_history_backtest.parquet"


def _already_ran_today() -> bool:
    """Check if the pipeline already ran today (idempotency guard)."""
    p = Path(DAILY_LOG_PATH)
    if not p.exists():
        return False
    today_str = date.today().isoformat()
    # Read last few lines efficiently
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("date") == today_str and not entry.get("dry_run"):
                    return True
            except json.JSONDecodeError:
                continue
    return False


def _log_daily(entry: dict) -> None:
    """Append a JSON-lines entry to the daily log."""
    p = Path(DAILY_LOG_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _should_retrain(force: bool = False) -> bool:
    """Check if we need to retrain based on the model registry."""
    if force:
        return True
    reg_path = Path(MODEL_REGISTRY)
    if not reg_path.exists():
        return True  # No model yet → must train
    with open(reg_path) as f:
        registry = json.load(f)
    last_train = registry.get("last_train_date", "")
    if not last_train:
        return True
    days_since = (date.today() - date.fromisoformat(last_train)).days
    return days_since >= RETRAIN_EVERY_DAYS


def _update_model_registry(metrics: dict) -> None:
    """Record model training metadata."""
    reg_path = Path(MODEL_REGISTRY)
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    registry = {}
    if reg_path.exists():
        with open(reg_path) as f:
            registry = json.load(f)
    registry.update({
        "last_train_date": date.today().isoformat(),
        "train_count": registry.get("train_count", 0) + 1,
        "last_metrics": metrics,
    })
    with open(reg_path, "w") as f:
        json.dump(registry, f, indent=2, default=str)


# ── Step 1: Incremental data download ─────────────────────

def update_ohlcv(cfg, predict_to: str) -> pd.DataFrame:
    """Download only the missing OHLCV bars since last update."""
    from app.data.store.parquet_store import ParquetStore
    from app.data.massive import MassiveClient, AggregatesAPI

    store = ParquetStore()
    ohlcv = store.read("ohlcv_smallcap")
    uni_df = store.read("smallcap_universe")
    # Only download updates for active tickers (delisted ones won't have new bars)
    active_mask = uni_df["active"] == True if "active" in uni_df.columns else pd.Series(True, index=uni_df.index)
    tickers = uni_df[active_mask]["ticker"].tolist()

    # Also include tickers with open positions (may not be in universe anymore)
    portfolio_path = Path(PORTFOLIO_PATH)
    if portfolio_path.exists():
        with open(portfolio_path) as f:
            portfolio = json.load(f)
        held_tickers = {p["ticker"] for p in portfolio.get("positions", [])}
        pending_tickers = {p["ticker"] for p in portfolio.get("pending_signals", [])}
        extra = (held_tickers | pending_tickers) - set(tickers)
        if extra:
            tickers = tickers + sorted(extra)
            print(f"  + {len(extra)} tickers con posición abierta añadidos: {', '.join(sorted(extra))}")

    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    last_date = ohlcv["date"].max().date()
    predict_to_date = pd.Timestamp(predict_to).date()

    # Check for tickers lagging behind the global max date
    last_per_ticker = ohlcv.groupby("ticker")["date"].max()
    stale_tickers = last_per_ticker[last_per_ticker < ohlcv["date"].max() - pd.Timedelta(days=2)]
    n_stale = len(stale_tickers)

    # Skip the download only if we already hold the target day's bar. Comparing
    # against (today - 1) meant a run holding yesterday's bar short-circuited and
    # never fetched today's — the cron captured no data for the day (fixed
    # 2026-06-12). predict_to already encodes weekend/holiday handling.
    if last_date >= predict_to_date and n_stale == 0:
        print(f"  OHLCV already current ({last_date})")
        return ohlcv

    if n_stale > 0:
        print(f"  ⚠ {n_stale} tickers desfasados: {', '.join(stale_tickers.index[:10])}")

    print(f"  Updating OHLCV: {last_date} → {predict_to} ({len(tickers)} tickers)")

    # Rate from MASSIVE_CALLS_PER_MINUTE (default 50, paid plan). When the
    # Polygon plan is downgraded to free, set this env to 5 — no code change
    # needed; the daily download just paces slower (~12s/ticker, job still
    # completes within the Actions 6h limit).
    cpm = int(os.environ.get("MASSIVE_CALLS_PER_MINUTE", "50"))
    client = MassiveClient(calls_per_minute=cpm)
    aggs = AggregatesAPI(client)

    # Import download_ohlcv from main pipeline
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_smallcap_pipeline import download_ohlcv

    ohlcv = download_ohlcv(
        aggs, tickers,
        train_start=(last_date - timedelta(days=5)).isoformat(),
        predict_to=predict_to,
        existing_ohlcv=ohlcv,
    )

    # Also update SPY
    existing_spy = store.read("smallcap_spy") if store.exists("smallcap_spy") else None
    spy_from = (last_date + timedelta(days=1)).isoformat()
    try:
        spy_bars = aggs.get_custom_bars("SPY", from_date=spy_from,
                                         to_date=predict_to, adjusted=True)
    except Exception as e:
        log.warning("spy_download_failed", error=str(e))
        spy_bars = []
    if spy_bars:
        spy_rows = [{"date": pd.Timestamp(b.trading_date), "ticker": "SPY",
                     "open": b.open, "high": b.high, "low": b.low,
                     "close": b.close, "volume": b.volume} for b in spy_bars]
        spy_new = pd.DataFrame(spy_rows)
        if existing_spy is not None and not existing_spy.empty:
            existing_spy["date"] = pd.to_datetime(existing_spy["date"])
            spy_df = pd.concat([existing_spy, spy_new], ignore_index=True)
            spy_df = spy_df.drop_duplicates(subset=["date"], keep="last")
        else:
            spy_df = spy_new
        store.write("smallcap_spy", spy_df)

    client.close()
    store.write("ohlcv_smallcap", ohlcv)
    print(f"  ✓ OHLCV: {len(ohlcv):,} rows, {ohlcv['date'].max().date()}")
    return ohlcv


# ── Step 2: Rebuild features ──────────────────────────────

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
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_smallcap_pipeline import _add_lag_features
    features = _add_lag_features(features)

    # Add sentiment features
    try:
        from app.features.sentiment import build_sentiment_features, build_market_sentiment_features
        news_df = store.read("smallcap_news") if store.exists("smallcap_news") else None
        if news_df is not None and not news_df.empty:
            sent = build_sentiment_features(news_df, ohlcv, lookback_days=7)
            if not sent.empty:
                features["date"] = pd.to_datetime(features["date"])
                sent["date"] = pd.to_datetime(sent["date"])
                features = features.merge(sent, on=["ticker", "date"], how="left")
                for c in sent.columns:
                    if c not in ("ticker", "date"):
                        features[c] = features[c].fillna(0)

            mkt = build_market_sentiment_features(news_df, ohlcv, lookback_days=7)
            if not mkt.empty:
                mkt["date"] = pd.to_datetime(mkt["date"])
                features = features.merge(mkt, on="date", how="left")
                for c in mkt.columns:
                    if c != "date":
                        features[c] = features[c].fillna(0)
            print(f"  ✓ Sentiment features added")
    except Exception as e:
        log.warning("sentiment_features_failed", error=str(e))

    # Add EDGAR fundamental features (dilution_pct, current_ratio)
    features = _merge_edgar_features(features)

    # Add meta-learning features (Error-Aware Features)
    features = _merge_meta_features(features, uni_df)

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

        n_filled = merged[V2_EDGAR_FEATURES[0]].notna().sum() if V2_EDGAR_FEATURES[0] in merged.columns else 0
        print(f"  ✓ EDGAR features added ({n_filled:,} rows with data)")
        return merged

    except Exception as e:
        log.warning("edgar_features_failed", error=str(e))
        return features


def _merge_meta_features(features: pd.DataFrame, universe_df: pd.DataFrame) -> pd.DataFrame:
    """Merge meta-learning features from signal history into the feature matrix."""
    try:
        from app.features.meta_features import build_meta_feature_panel
        from app.features.pipeline import assign_sectors

        # Load signal history (backtest + live)
        signal_hist = _load_combined_signal_history()
        if signal_hist.empty or signal_hist["outcome_filled"].sum() < 30:
            return features

        # Build sector map
        tmp = assign_sectors(
            universe_df[["ticker"]].drop_duplicates().assign(date=pd.Timestamp("2024-01-01")),
            universe_df.to_dict("records"),
        )
        sector_map = dict(zip(tmp["ticker"], tmp["sector"]))

        features["date"] = pd.to_datetime(features["date"])
        # Only compute for dates where signal history exists
        sig_min = pd.to_datetime(signal_hist["signal_date"]).min() + pd.Timedelta(days=30)
        meta_dates = sorted(features[features["date"] >= sig_min]["date"].unique())

        if not meta_dates:
            return features

        meta_panel = build_meta_feature_panel(signal_hist, meta_dates, sector_map)
        if meta_panel.empty:
            return features

        meta_panel["date"] = pd.to_datetime(meta_panel["date"])
        features = features.merge(
            meta_panel[["date", "ticker"] + V2_META_FEATURES],
            on=["date", "ticker"],
            how="left",
        )
        n_filled = features[V2_META_FEATURES[0]].notna().sum()
        print(f"  ✓ Meta features added ({n_filled:,} rows with data)")

    except Exception as e:
        log.warning("meta_features_failed", error=str(e))

    return features


def _load_combined_signal_history() -> pd.DataFrame:
    """Load signal history: backtest (historical) + live (recent)."""
    from app.data.store.parquet_store import ParquetStore
    store = ParquetStore()
    parts = []

    # Backtest-generated history
    bt_path = Path(SIGNAL_HISTORY_BACKTEST)
    if bt_path.exists():
        bt = pd.read_parquet(bt_path)
        parts.append(bt)

    # Live signal history
    sh_path = Path("data/paper_trading/signal_history.parquet")
    if sh_path.exists():
        live = pd.read_parquet(sh_path)
        if not live.empty:
            parts.append(live)

    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    combined["signal_date"] = pd.to_datetime(combined["signal_date"])
    # Deduplicate: live takes priority over backtest for overlapping dates
    combined = combined.sort_values("signal_date")
    if "source" in combined.columns:
        combined = combined.drop_duplicates(
            subset=["signal_date", "ticker"], keep="last"
        )
    return combined


# ── Step 3: Train or load models ──────────────────────────

def train_or_load_models(features: pd.DataFrame, predict_from: str, cfg,
                         force_retrain: bool = False):
    """Train V3 LambdaRank model on sector-relative 20d returns, or load cached.

    Returns (model, predict_data, train_metrics).
    """
    import pickle
    import lightgbm as lgb

    model_path = Path(MODEL_PATH)
    need_retrain = _should_retrain(force_retrain)

    if not need_retrain and model_path.exists():
        print("  Loading cached V3 model (retrain not due)...")
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)  # noqa: S301

            features["date"] = pd.to_datetime(features["date"])
            predict_data = features[
                features["date"] >= pd.Timestamp(predict_from)
            ].copy()

            print(f"  ✓ V3 model loaded (trained {_days_since_train()} days ago)")
            return model, predict_data, {}
        except Exception as e:
            print(f"  ⚠ Model load failed ({e}), retraining...")

    # Retrain V3 model
    print("  ═══ V3 LAMBDARANK MODEL RETRAINING ═══")
    features["date"] = pd.to_datetime(features["date"])

    train_data = features[features["date"] < pd.Timestamp(predict_from)].copy()
    predict_data = features[features["date"] >= pd.Timestamp(predict_from)].copy()

    # Check required columns exist — base features + EDGAR + meta features
    available_features = [f for f in V2_FEATURES if f in train_data.columns]
    available_edgar = [f for f in V2_EDGAR_FEATURES if f in train_data.columns]
    available_meta = [f for f in V2_META_FEATURES if f in train_data.columns]
    all_train_features = available_features + available_edgar + available_meta
    if len(available_features) < 15:
        print(f"  ⚠ Only {len(available_features)}/{len(V2_FEATURES)} features available")
    if available_edgar:
        n_edgar = train_data[available_edgar[0]].notna().sum()
        print(f"  EDGAR features: {len(available_edgar)} ({n_edgar:,} rows with data)")
    if available_meta:
        n_meta = train_data[available_meta[0]].notna().sum()
        print(f"  Meta features: {len(available_meta)} ({n_meta:,} rows with data)")

    # Clean training data — V3 LambdaRank: relevance bins per date + group structure
    train_clean = train_data.dropna(subset=[V2_TARGET]).sort_values("date").copy()
    train_clean["_rel"] = train_clean.groupby("date")[V2_TARGET].transform(
        lambda s: pd.qcut(s.rank(method="first"), V3_N_BINS,
                          labels=False, duplicates="drop")
    )
    train_clean["_rel"] = train_clean["_rel"].fillna(0).astype(int).clip(0, V3_N_BINS - 1)
    X_train = train_clean[all_train_features].fillna(0).values
    y_train = train_clean["_rel"].values
    group = train_clean.groupby("date").size().values

    print(f"  Training (V3 LambdaRank): {len(train_clean):,} rows, "
          f"{len(all_train_features)} features, {len(group)} dates (groups)")
    print(f"  Target: {V2_TARGET} (binned 0..{V3_N_BINS - 1})")

    ds = lgb.Dataset(X_train, y_train, group=group,
                     feature_name=all_train_features, free_raw_data=True)
    model = lgb.train(V2_LGB_PARAMS, ds, num_boost_round=V3_NUM_BOOST_ROUND,
                      callbacks=[lgb.log_evaluation(0)])

    # Save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    metrics = {
        "n_train": len(train_clean),
        "n_features": len(all_train_features),
        "n_edgar_features": len(available_edgar),
        "n_meta_features": len(available_meta),
        "target": V2_TARGET,
        "model_type": "lgb_lambdarank_v3",
    }
    _update_model_registry(metrics)
    print(f"  ✓ V3 LambdaRank model trained ({model.num_trees()} trees)")

    return model, predict_data, metrics


def _days_since_train() -> int:
    """Days since last model training."""
    reg_path = Path(MODEL_REGISTRY)
    if not reg_path.exists():
        return 999
    with open(reg_path) as f:
        registry = json.load(f)
    last = registry.get("last_train_date", "")
    if not last:
        return 999
    return (date.today() - date.fromisoformat(last)).days


# ── Step 4: Generate today's signals ──────────────────────

def generate_today_signals(model, predict_data, ohlcv, today: str) -> pd.DataFrame:
    """Generate V2 signals: rank stocks by predicted sector-relative return."""
    today_ts = pd.Timestamp(today)
    predict_data["date"] = pd.to_datetime(predict_data["date"])

    available_dates = sorted(predict_data["date"].unique())
    if not available_dates:
        print("  ⚠ No prediction data available")
        return pd.DataFrame()

    latest = available_dates[-1]
    today_data = predict_data[predict_data["date"] == latest].copy()

    if today_data.empty:
        print(f"  ⚠ No data for {latest.date()}")
        return pd.DataFrame()

    # Freshness guard: stale features mean the download failed silently —
    # never generate signals from old data.
    if is_stale(latest, today_ts):
        print(f"  ✗ Features are stale (latest={latest.date()}, today={today_ts.date()}) "
              f"— skipping signal generation")
        return pd.DataFrame()

    # Tradability gate: delisted/illiquid tickers stay in OHLCV for training
    # (anti-survivorship) but must never be SELECTED. Live trading bought
    # sub-penny zombies (SRNE @ $0.0006) before this filter existed.
    today_data["_tradable"] = tradable_mask(today_data)
    n_excluded = int((~today_data["_tradable"]).sum())
    print(f"  ✓ Tradability gate: {today_data['_tradable'].sum()} tradable, "
          f"{n_excluded} excluded (price/liquidity)")

    # Score all stocks with V2 model (base + EDGAR + meta features)
    available_features = [f for f in V2_FEATURES if f in today_data.columns]
    available_edgar = [f for f in V2_EDGAR_FEATURES if f in today_data.columns]
    available_meta = [f for f in V2_META_FEATURES if f in today_data.columns]
    all_features = available_features + available_edgar + available_meta
    X = today_data[all_features].fillna(0).values
    today_data["v2_score"] = model.predict(X)

    # Rank by predicted sector-relative return (higher = better)
    today_data = today_data.sort_values("v2_score", ascending=False)

    records = []
    n_buys = 0
    for _, row in today_data.iterrows():
        ticker = str(row.get("ticker", ""))
        dt_str = str(latest.date())

        rejection = ""
        if not bool(row["_tradable"]):
            recommendation = "HOLD"
            position_size = 0.0
            rejection = "untradable: below min price/liquidity"
        elif n_buys < V2_TOP_K:
            recommendation = "BUY"
            position_size = 1.0 / V2_TOP_K  # Equal weight
            n_buys += 1
        else:
            recommendation = "HOLD"
            position_size = 0.0

        # ATR-adaptive trailing stop
        vol = float(row.get("realized_vol_20d", 0.3))
        atr_pct = float(row.get("atr_pct_20d", vol / np.sqrt(252) * 2))
        median_atr = today_data.head(V2_TOP_K)["atr_pct_20d"].median() if "atr_pct_20d" in today_data.columns else 0.03
        if median_atr > 0 and recommendation == "BUY":
            adaptive_trail = np.clip(0.16 * (atr_pct / median_atr), 0.10, 0.16)
        else:
            adaptive_trail = 0.16

        records.append({
            "ticker": ticker,
            "date": dt_str,
            "recommendation": recommendation,
            "ensemble_score": float(row["v2_score"]),
            "calibrated_prob": float(row["v2_score"]),  # compatibility
            "expected_return": float(row["v2_score"]),
            "position_size_pct": position_size,
            "trailing_stop_pct": adaptive_trail,
            "stop_loss_pct": adaptive_trail,
            "rejection_reasons": rejection,
        })

    signals = pd.DataFrame(records)
    buys = signals[signals["recommendation"] == "BUY"]
    print(f"  ✓ Signals: {len(signals)} total, {len(buys)} BUY")
    if not buys.empty:
        for _, b in buys.iterrows():
            print(f"    BUY {b['ticker']:6s} — score: {b['ensemble_score']:.4f}, "
                  f"trail: {b['trailing_stop_pct']:.0%}")

    return signals


# ── Step 5: Paper trading execution ──────────────────────

def _generate_signals_for_date(model, features: pd.DataFrame, ohlcv: pd.DataFrame,
                               target_date: str, verbose: bool = False) -> pd.DataFrame:
    """Generate signals for a specific date using pre-built features.

    Like generate_today_signals but targets a specific historical date
    instead of always using the latest available date.
    """
    target_ts = pd.Timestamp(target_date)
    features["date"] = pd.to_datetime(features["date"])
    day_data = features[features["date"] == target_ts].copy()

    if day_data.empty:
        return pd.DataFrame()

    available_features = [f for f in V2_FEATURES if f in day_data.columns]
    available_edgar = [f for f in V2_EDGAR_FEATURES if f in day_data.columns]
    available_meta = [f for f in V2_META_FEATURES if f in day_data.columns]
    all_features = available_features + available_edgar + available_meta
    X = day_data[all_features].fillna(0).values
    day_data["v2_score"] = model.predict(X)
    day_data = day_data.sort_values("v2_score", ascending=False)

    # Same tradability gate as generate_today_signals (selection only)
    day_data["_tradable"] = tradable_mask(day_data)

    records = []
    n_buys = 0
    for _, row in day_data.iterrows():
        ticker = str(row.get("ticker", ""))
        rejection = ""
        if not bool(row["_tradable"]):
            recommendation = "HOLD"
            position_size = 0.0
            rejection = "untradable: below min price/liquidity"
        elif n_buys < V2_TOP_K:
            recommendation = "BUY"
            position_size = 1.0 / V2_TOP_K
            n_buys += 1
        else:
            recommendation = "HOLD"
            position_size = 0.0

        vol = float(row.get("realized_vol_20d", 0.3))
        atr_pct = float(row.get("atr_pct_20d", vol / np.sqrt(252) * 2))
        median_atr = (
            day_data.head(V2_TOP_K)["atr_pct_20d"].median()
            if "atr_pct_20d" in day_data.columns else 0.03
        )
        if median_atr > 0 and recommendation == "BUY":
            adaptive_trail = np.clip(0.16 * (atr_pct / median_atr), 0.10, 0.16)
        else:
            adaptive_trail = 0.16

        records.append({
            "ticker": ticker, "date": target_date,
            "recommendation": recommendation,
            "ensemble_score": float(row["v2_score"]),
            "calibrated_prob": float(row["v2_score"]),
            "expected_return": float(row["v2_score"]),
            "position_size_pct": position_size,
            "trailing_stop_pct": adaptive_trail,
            "stop_loss_pct": adaptive_trail,
            "rejection_reasons": rejection,
        })

    signals = pd.DataFrame(records)
    if verbose:
        buys = signals[signals["recommendation"] == "BUY"]
        if not buys.empty:
            tickers = ", ".join(f"{r['ticker']}({r['ensemble_score']:.3f})" for _, r in buys.iterrows())
            print(f"    {target_date}: {tickers}")
    return signals


def _get_missed_trading_days(ohlcv: pd.DataFrame, last_update: str, today: str) -> list[str]:
    """Get trading days between last_update and today (exclusive of last_update, inclusive of today).

    If the pipeline wasn't run for several days, we need to replay each
    missed day to correctly execute pending signals, check trailing stops,
    and count holding periods.
    """
    today_dt = date.fromisoformat(today)
    # A freshly created/reset portfolio has no last_update. It must start
    # trading TODAY — never replay the full OHLCV history (which begins in
    # 2021). Replaying history into a "live" portfolio fabricated thousands
    # of backtest trades dated 2021+ after the V4 reset (bug fixed 2026-06-12).
    if not last_update:
        return [today]

    ohlcv_dates = sorted(ohlcv["date"].dt.date.unique())
    last_dt = date.fromisoformat(last_update)

    # Return all trading days after last_update up to and including today
    missed = [str(d) for d in ohlcv_dates if last_dt < d <= today_dt]
    return missed


def run_paper_trading(signals: pd.DataFrame, ohlcv: pd.DataFrame,
                      today: str, capital: float, dry_run: bool = False,
                      model=None, features: pd.DataFrame | None = None,
                      portfolio_path: str | None = None,
                      adaptive_stop: bool = False,
                      strategy_label: str = "") -> dict:
    """Execute paper trading cycle with full signal tracking.

    Replays ALL missed trading days since last execution to ensure:
    - Pending signals execute at the correct day's open (not days later)
    - Trailing stops are checked every day (not just today)
    - Holding period counter advances correctly
    - NEW: Generates signals for each missed day (not just today)
    """
    from app.paper_trading import PaperTrader
    from app.signal_tracker import SignalTracker

    p_path = portfolio_path or PORTFOLIO_PATH

    # State lives in Supabase (source of truth). Hydrate the local JSON from it
    # so the run continues the real portfolio even on a fresh checkout (no state
    # committed to git). If Supabase is unreachable the read raises and the run
    # fails loudly — preferable to silently resetting the portfolio.
    strat = strategy_label or ("adaptive" if adaptive_stop else "baseline")
    if supabase_store.is_configured():
        remote = supabase_store.read_state(strat)
        if remote is not None:
            Path(p_path).parent.mkdir(parents=True, exist_ok=True)
            Path(p_path).write_text(json.dumps(remote))
            print(f"  ↻ Hydrated {strat} portfolio from Supabase")

    pt = PaperTrader.load_or_create(
        p_path,
        initial_capital=capital,
        max_positions=8,
        holding_period_days=20,
        adaptive_stop=adaptive_stop,
        # V4 exit sweep (2026-06-11): +40% profit target locks runaway
        # winners; baseline Sharpe 2.52->2.79, adaptive WR 59.7% maxDD -15.1%
        profit_target=0.40,
    )

    # Derive signal tracker and log paths from portfolio path
    p_dir = Path(p_path).parent
    tracker_path = str(p_dir / "signal_history.parquet")
    trade_log = str(p_dir / "trades.parquet")
    daily_log = str(p_dir / "daily_log.jsonl")

    tracker = SignalTracker(path=tracker_path)

    # Determine missed days since last execution
    last_update = pt.state.last_update or ""
    missed_days = _get_missed_trading_days(ohlcv, last_update, today)

    # Separate intermediate days (need signal generation) from today
    intermediate_days = [d for d in missed_days if d != today]
    can_generate_intermediate = model is not None and features is not None

    if intermediate_days:
        if can_generate_intermediate:
            print(f"  ⚠ Catching up {len(intermediate_days)} missed days with full signal replay ({intermediate_days[0]} → {intermediate_days[-1]})")
        else:
            print(f"  ⚠ Catching up {len(intermediate_days)} missed days (positions only, no model for signals)")

    # Replay each missed day: execute pending → update positions → generate new signals
    all_closed = []
    all_entered = []
    intermediate_signals_count = 0
    for day in intermediate_days:
        # 1. Execute pending signals at this day's open
        entered = pt.execute_pending(ohlcv, day)
        if entered:
            all_entered.extend(entered)
        # 2. Check trailing stops and holding period expiry
        closed = pt.update_positions(ohlcv, day)
        if closed:
            all_closed.extend(closed)
        # 3. Generate and queue new signals for this day (if model available)
        if can_generate_intermediate:
            day_signals = _generate_signals_for_date(model, features, ohlcv, day, verbose=True)
            if not day_signals.empty:
                traded, skipped = pt.process_signals(day_signals, day)
                tracker.record_signals(day_signals, traded, skipped, day)
                intermediate_signals_count += len(day_signals[day_signals["recommendation"] == "BUY"])

    # Process today: execute pending + update positions
    if today in missed_days:
        entered = pt.execute_pending(ohlcv, today)
        if entered:
            all_entered.extend(entered)
        closed = pt.update_positions(ohlcv, today)
        if closed:
            all_closed.extend(closed)

    # Report catch-up activity
    if all_entered:
        print(f"  ✓ Entered (catch-up): {', '.join(all_entered)}")
    if all_closed:
        for t in all_closed:
            print(f"  ✗ Closed {t.ticker}: {t.pnl_pct:+.2%} ({t.exit_reason}) on {t.exit_date}")
        from dataclasses import asdict
        tracker.update_trade_outcomes([asdict(t) for t in all_closed])
    if intermediate_signals_count:
        print(f"  ✓ Intermediate signals generated: {intermediate_signals_count} BUY across {len(intermediate_days)} days")

    # Queue today's new signals for tomorrow's execution
    traded_tickers: set[str] = set()
    skip_reasons: dict[str, str] = {}
    if not dry_run and not signals.empty:
        traded_tickers, skip_reasons = pt.process_signals(signals, today)

    # Record ALL BUY signals in tracker (traded + skipped)
    if not signals.empty:
        tracker.record_signals(signals, traded_tickers, skip_reasons, today)

    # 5. Backfill outcomes for signals old enough (20+ trading days)
    n_filled = tracker.backfill_outcomes(ohlcv, horizon_days=V2_HOLD_DAYS)
    if n_filled:
        print(f"  ✓ Outcomes backfilled: {n_filled} signals")

    # 6. Show tracker stats
    stats = tracker.summary_stats()
    if stats.get("outcomes_available", 0) > 0:
        print(f"  ✓ Signal history: {stats['total_signals']} signals, "
              f"{stats['outcomes_available']} with outcomes")
        if "score_return_correlation" in stats:
            print(f"    Score↔Return correlation: {stats['score_return_correlation']:.3f}")
        if "traded_avg_ret_20d" in stats:
            print(f"    Traded avg 20d ret: {stats['traded_avg_ret_20d']:+.2%} "
                  f"(hit rate: {stats['traded_hit_rate']:.0%})")
        if "missed_avg_ret_20d" in stats:
            print(f"    Missed avg 20d ret: {stats['missed_avg_ret_20d']:+.2%} "
                  f"(hit rate: {stats['missed_hit_rate']:.0%})")

    # 7. Get summary
    summary = pt.summary(ohlcv)

    # 8. Save state
    if not dry_run:
        pt.save()
        tracker.save()

        # Also save trades to parquet (append)
        trades_df = pt.trades_to_dataframe()
        if not trades_df.empty:
            tp = Path(trade_log)
            tp.parent.mkdir(parents=True, exist_ok=True)
            trades_df.to_parquet(tp, index=False)

        # Write daily log entry for this strategy
        dl = Path(daily_log)
        dl.parent.mkdir(parents=True, exist_ok=True)
        log_entry = {
            "date": today,
            "portfolio_value": summary["total_value"],
            "cash": summary["cash"],
            "n_positions": summary["n_open_positions"],
            "n_closed": summary["n_closed_trades"],
        }
        with open(dl, "a") as f:
            f.write(json.dumps(log_entry, default=str) + "\n")

        # Dual-write to Supabase (single source of truth, migrating off git).
        # Best-effort: a Supabase failure must never break the pipeline.
        if supabase_store.is_configured():
            from dataclasses import asdict
            try:
                supabase_store.write_state(strat, asdict(pt.state))
                if all_closed:
                    supabase_store.append_trades(strat, [asdict(t) for t in all_closed])
                supabase_store.upsert_nav(strat, today, float(summary["total_value"]))
                if not signals.empty:
                    sig_rows = [{
                        "signal_date": today,
                        "ticker": r["ticker"],
                        "score": float(r["ensemble_score"]),
                        "recommendation": r["recommendation"],
                        "was_traded": r["ticker"] in traded_tickers,
                        "skip_reason": (skip_reasons.get(r["ticker"])
                                        or r.get("rejection_reasons") or None),
                        "actual_ret_20d": None,
                    } for _, r in signals.iterrows()]
                    supabase_store.upsert_signals(strat, sig_rows)
                print(f"  ✓ Supabase: state + nav + {len(all_closed)} trades "
                      f"+ {len(signals)} signals ({strat})")
            except Exception as e:
                log.warning("supabase_write_failed", strategy=strat, error=str(e))
                print(f"  ⚠ Supabase write failed ({strat}): {e}")
    else:
        tracker.save()  # Always save signal history, even in dry-run

    return summary


# ── Main ──────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SCAI Daily Pipeline")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Force model retraining regardless of schedule")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate signals but don't execute trades")
    parser.add_argument("--capital", type=float, default=1000.0,
                        help="Initial paper trading capital (default: 1000)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip OHLCV download (use cached data)")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip feature rebuild (use cached features)")
    parser.add_argument("--train-start", default="2020-01-01",
                        help="Training data start date")
    parser.add_argument("--predict-from", default=None,
                        help="Prediction start date (default: 30 days ago)")
    args = parser.parse_args()

    cfg = get_settings()
    setup_logging("INFO")
    set_global_seed(cfg.seed)

    today = date.today().isoformat()

    # Never act on an in-progress (partial) daily bar. If the US market hasn't
    # closed yet, the current day's bar is incomplete and would trigger trailing
    # stops against intraday prices. In that case operate as of the last
    # completed session (this mirrors running `scai` after close, as on macOS).
    now_et = datetime.now(ZoneInfo("America/New_York"))
    market_closed = now_et.weekday() < 5 and now_et.time() >= time(16, 15)
    predict_to = today if market_closed else (date.today() - timedelta(days=1)).isoformat()

    # Idempotency: skip if already ran today (safe to re-trigger on wake/login)
    if not args.dry_run and _already_ran_today():
        print(f"  ✓ Already ran today ({today}). Skipping. Use --dry-run to re-check.\n")
        return

    # Skip weekends (no market data)
    weekday = date.today().weekday()
    if weekday >= 5:  # Saturday=5, Sunday=6
        print(f"  ℹ Weekend ({date.today().strftime('%A')}). Skipping.\n")
        return

    # Default predict_from: 30 days back (enough for signal generation context)
    if args.predict_from is None:
        predict_from = (date.today() - timedelta(days=30)).isoformat()
    else:
        predict_from = args.predict_from

    print()
    print("=" * 60)
    print("  SCAI Daily Pipeline")
    print(f"  Date:          {today}")
    print(f"  Mode:          {'DRY RUN' if args.dry_run else 'LIVE PAPER'}")
    print(f"  Retrain:       {'FORCED' if args.force_retrain else f'every {RETRAIN_EVERY_DAYS}d (last: {_days_since_train()}d ago)'}")
    print(f"  Capital:       €{args.capital:,.0f}")
    print("=" * 60)
    print()

    from app.data.store.parquet_store import ParquetStore
    store = ParquetStore()

    # ── Step 1: Update data ──────────────────────────
    print("STEP 1/5 ▸ Updating market data...")
    if args.skip_download:
        ohlcv = store.read("ohlcv_smallcap")
        ohlcv["date"] = pd.to_datetime(ohlcv["date"])
        print(f"  Using cached OHLCV ({ohlcv['date'].max().date()})")
    else:
        ohlcv = update_ohlcv(cfg, predict_to)

    # Drop any partial bar for the in-progress day (e.g. left by a mid-session
    # download) so trailing stops are only ever evaluated on completed sessions.
    if not market_closed:
        cal_today = date.today()
        mask_today = ohlcv["date"].dt.date == cal_today
        if mask_today.any():
            n_partial = int(mask_today.sum())
            ohlcv = ohlcv[~mask_today].copy()
            if not args.dry_run:
                store.write("ohlcv_smallcap", ohlcv)  # purge so it's re-fetched after close
            print(f"  ⚠ Market open (ET {now_et:%H:%M}): dropped {n_partial} partial bars "
                  f"for {cal_today}; using last completed session")
    # Re-anchor 'today' to the latest completed session actually present in data
    today = ohlcv["date"].max().date().isoformat()
    print()

    # ── Step 2: Rebuild features ─────────────────────
    print("STEP 2/5 ▸ Building features...")
    if args.skip_features:
        features = store.read("features_smallcap")
        print(f"  Using cached features ({len(features):,} rows)")
    else:
        features = rebuild_features(ohlcv, cfg)
    print()

    # ── Step 3: Train or load models ──────────────────
    print("STEP 3/5 ▸ Models...")
    model, predict_data, train_metrics = (
        train_or_load_models(features, predict_from, cfg, args.force_retrain)
    )
    print()

    # ── Step 4: Generate signals ─────────────────────
    print("STEP 4/5 ▸ Generating signals...")
    signals = generate_today_signals(model, predict_data, ohlcv, today)

    # Save signals
    if not signals.empty:
        signals_path = f"data/paper_trading/signals_{today}.parquet"
        Path(signals_path).parent.mkdir(parents=True, exist_ok=True)
        signals.to_parquet(signals_path, index=False)
    print()

    # ── Step 5: Paper trading (dual strategies) ────
    print("STEP 5/5 ▸ Paper trading execution...")

    # Strategy A: Baseline (standard trailing stop)
    print("  ── Strategy A: Baseline ──")
    summary_a = run_paper_trading(
        signals, ohlcv, today, args.capital, args.dry_run,
        model=model, features=features,
        portfolio_path=PORTFOLIO_PATH,
        adaptive_stop=False,
        strategy_label="baseline",
    )

    # Strategy B: Adaptive Stop (tighten to 6% after day 5 if profitable)
    print("  ── Strategy B: Adaptive Stop ──")
    summary_b = run_paper_trading(
        signals, ohlcv, today, args.capital, args.dry_run,
        model=model, features=features,
        portfolio_path=PORTFOLIO_PATH_ADAPTIVE,
        adaptive_stop=True,
        strategy_label="adaptive",
    )
    print()

    # ── Summary ──────────────────────────────────────
    for label, summary in [("BASELINE", summary_a), ("ADAPTIVE STOP", summary_b)]:
        print("=" * 60)
        print(f"  {label}")
        print("=" * 60)
        print(f"  Portfolio:     €{summary['total_value']:,.2f} ({summary['total_return']})")
        print(f"  Cash:          €{summary['cash']:,.2f}")
        print(f"  Open:          {summary['n_open_positions']} positions")
        print(f"  Closed today:  check log")
        print(f"  Total trades:  {summary['n_closed_trades']} (win rate: {summary['win_rate']})")
        print(f"  Pending:       {summary['pending_signals']} signals")
        if summary["open_positions"]:
            print()
            print(f"  {'Ticker':8s} {'Entry':>8s} {'Current':>8s} {'P&L':>8s} {'Trail@':>8s} {'Trail%':>6s}")
            print(f"  {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")
            for pos in summary["open_positions"]:
                trail_str = f"{pos.get('trail_pct', 0):.0f}%"
                print(f"  {pos['ticker']:8s} {pos['entry_price']:8.2f} {pos['current_price']:8.2f} "
                      f"{pos['pnl_pct']:>8s} {pos['trail_trigger']:8.2f} {trail_str:>6s}")
        print()

    # Log daily entry (baseline as primary)
    _log_daily({
        "date": today,
        "portfolio_value": summary_a["total_value"],
        "cash": summary_a["cash"],
        "n_positions": summary_a["n_open_positions"],
        "n_closed": summary_a["n_closed_trades"],
        "portfolio_value_adaptive": summary_b["total_value"],
        "retrained": bool(train_metrics),
        "dry_run": args.dry_run,
    })

    print("  ✓ Done\n")


def _check_meta_learning_readiness() -> None:
    """Report meta-learning signal history status."""
    try:
        combined = _load_combined_signal_history()
        if combined.empty:
            return
        n_total = len(combined)
        n_filled = combined.get("outcome_filled", pd.Series(dtype=bool)).sum()
        print(f"  ℹ Meta-learning: {n_filled:,} signals with outcomes (total: {n_total:,})")
    except Exception:
        pass


if __name__ == "__main__":
    main()
