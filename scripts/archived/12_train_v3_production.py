"""V3 production training — train v3_hp_combo on ALL data through yesterday.

Saves to data/models/smallcap_v3_lambdarank.pkl  +  registry update.
This is the model artifact the daily_pipeline.py loads in production.
"""
from __future__ import annotations

import json
import pickle
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from _v3_harness import (
    V2_FEATURES_BASE, V2_EDGAR_FEATURES, V2_META_FEATURES,
    V2_LGB_PARAMS, V2_TARGET,
)

LEAK_CHECK_SCRIPT = Path(__file__).parent / "18_verify_no_leak.py"

N_BINS = 16
MODEL_PATH = Path("data/models/smallcap_v3_lambdarank.pkl")
REGISTRY_PATH = Path("data/paper_trading/model_registry.json")


def make_v3_params() -> dict:
    p = dict(V2_LGB_PARAMS)
    p.update({
        "objective": "lambdarank", "metric": "ndcg",
        "num_leaves": 31, "max_depth": 6,
        "min_child_samples": 30, "learning_rate": 0.05,
        "lambdarank_truncation_level": 8,
        "label_gain": list(range(N_BINS)),
        "reg_lambda": 5.0,
    })
    p.pop("reg_alpha", None)
    return p


def main() -> None:
    print("Loading features...")
    f = pd.read_parquet("data/processed/features_smallcap.parquet")
    f["date"] = pd.to_datetime(f["date"])
    print(f"  shape={f.shape}  date range {f.date.min().date()} → {f.date.max().date()}")

    feat_cols = V2_FEATURES_BASE + V2_EDGAR_FEATURES + V2_META_FEATURES
    print(f"  features={len(feat_cols)}")

    # ── MANDATORY anti-leak gate (see PROJECT.md § Anti-Leak Protocol) ──
    # Refuses to train if any feature shows correlation > thresholds with target.
    print("\nRunning anti-leak verification (gate)...")
    # Temporarily write feat_cols to registry so the gate inspects the right list
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if REGISTRY_PATH.exists():
        _reg = json.loads(REGISTRY_PATH.read_text())
    else:
        _reg = {}
    _reg["feat_cols"] = feat_cols
    REGISTRY_PATH.write_text(json.dumps(_reg, indent=2, default=str))
    rc = subprocess.call([sys.executable, str(LEAK_CHECK_SCRIPT)])
    if rc != 0:
        raise SystemExit(
            "\n❌ Anti-leak verification FAILED. Refusing to train.\n"
            "   Inspect features listed above and remove them before retrying."
        )

    train = f.dropna(subset=[V2_TARGET]).sort_values("date").copy()
    print(f"  train_rows={len(train):,}")

    # relevance bins per date
    train["_rel"] = train.groupby("date")[V2_TARGET].transform(
        lambda s: pd.qcut(s.rank(method="first"), N_BINS, labels=False, duplicates="drop")
    )
    train["_rel"] = train["_rel"].fillna(0).astype(int).clip(0, N_BINS - 1)
    X = train[feat_cols].fillna(0).values
    y = train["_rel"].values
    group = train.groupby("date").size().values

    params = make_v3_params()
    print(f"  params: {params}")
    print("Training (600 rounds)...")
    ds = lgb.Dataset(X, y, group=group, feature_name=feat_cols, free_raw_data=True)
    model = lgb.train(params, ds, num_boost_round=600,
                      callbacks=[lgb.log_evaluation(0)])
    print(f"  ✓ {model.num_trees()} trees")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(model, fh)
    print(f"  saved → {MODEL_PATH}")

    # Update registry
    if REGISTRY_PATH.exists():
        reg = json.loads(REGISTRY_PATH.read_text())
    else:
        reg = {}
    reg.update({
        "version": "v3",
        "objective": "lambdarank",
        "model_path": str(MODEL_PATH),
        "n_train": int(len(train)),
        "n_features": len(feat_cols),
        "feat_cols": feat_cols,
        "params": params,
        "last_train_date": date.today().isoformat(),
        "train_count": int(reg.get("train_count", 0)) + 1,
    })
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2, default=str))
    print(f"  registry updated → {REGISTRY_PATH}")


if __name__ == "__main__":
    main()
