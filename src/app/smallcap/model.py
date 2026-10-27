"""Small-cap LambdaRank model: frozen spec, weekly retrain, registry."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from app.utils import get_logger

log = get_logger(__name__)


# ── Constants ───────────────────────────────────────────────
RETRAIN_EVERY_DAYS = 7          # Retrain model every N calendar days


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

    # Check required columns exist — base features + EDGAR
    available_features = [f for f in V2_FEATURES if f in train_data.columns]
    available_edgar = [f for f in V2_EDGAR_FEATURES if f in train_data.columns]
    all_train_features = available_features + available_edgar
    if len(available_features) < 15:
        print(f"  ⚠ Only {len(available_features)}/{len(V2_FEATURES)} features available")
    if available_edgar:
        n_edgar = train_data[available_edgar[0]].notna().sum()
        print(f"  EDGAR features: {len(available_edgar)} ({n_edgar:,} rows with data)")

    # Clean training data — V3 LambdaRank: relevance bins per date + group structure
    train_clean = train_data.dropna(subset=[V2_TARGET]).sort_values("date").copy()
    train_clean["_rel"] = train_clean.groupby("date")[V2_TARGET].transform(
        lambda s: pd.qcut(s.rank(method="first"), V3_N_BINS,
                          labels=False, duplicates="drop")
    )
    train_clean["_rel"] = train_clean["_rel"].fillna(0).astype(int).clip(0, V3_N_BINS - 1)
    from app.features.leak_checks import assert_no_leak
    assert_no_leak(train_clean, all_train_features, V2_TARGET)   # raises -> model not saved
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

