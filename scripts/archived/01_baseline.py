"""V3 Step 0 — Reproduce V2 baseline with frozen harness.

This is the ground truth that all V3 changes must beat. Saves to
data/v3_benchmarks/v2_baseline.json.

Expected ballpark from prior validation: IC≈0.053, Sharpe≈1.89, WR≈42%, 10/16 folds positive.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _v3_harness import (
    V2_FEATURES_BASE, V2_EDGAR_FEATURES, V2_META_FEATURES,
    V2_LGB_PARAMS, V2_TARGET,
    run_walkforward, save_result,
)


def main() -> None:
    print("Loading data...")
    features = pd.read_parquet("data/processed/features_smallcap.parquet")
    ohlcv = pd.read_parquet("data/processed/ohlcv_smallcap.parquet")
    features["date"] = pd.to_datetime(features["date"])
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    feat_cols = V2_FEATURES_BASE + V2_EDGAR_FEATURES + V2_META_FEATURES
    print(f"Features used: {len(feat_cols)} | training rows: {features[V2_TARGET].notna().sum():,}")

    t0 = time.time()
    result = run_walkforward(
        features=features, ohlcv=ohlcv,
        feat_cols=feat_cols,
        config_name="v2_baseline",
        lgb_params=V2_LGB_PARAMS,
    )
    fp = save_result(result)

    agg = result.aggregate()
    print(f"\n=== V2 BASELINE  (elapsed {time.time() - t0:.0f}s) ===")
    for k, v in agg.items():
        if isinstance(v, float):
            if any(t in k for t in ("rate", "return", "alpha", "selectivity")):
                print(f"  {k:<28} {v:.2%}")
            else:
                print(f"  {k:<28} {v:.4f}")
        else:
            print(f"  {k:<28} {v}")
    print(f"Saved to {fp}")


if __name__ == "__main__":
    main()
