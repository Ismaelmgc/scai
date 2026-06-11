"""V4 Phase 1: tradability-filter threshold sweep + honest re-baseline.

One full walk-forward training pass (16 LGB models, production 28-feature
no-meta config) with prediction caching, then replays the cache across a
price × ADV threshold grid. Establishes:

- ``v4_nofilt_nometa``  — clean unfiltered re-baseline (no leaky meta
  features, zero cost): the honest "old fiction" reference.
- ``v4_filt_p{P}_a{A}`` — each filter combo.
- ``v4_filt_baseline``  — the winning combo (with 15 bps/side cost), the
  anchor every later V4 experiment is compared against.

Winner rule: maximize mean tradable-IC subject to skipping < 5% of
rebalances for lack of TOP_K tradable names; ties go to the stricter combo.

Usage:
    PYTHONPATH=src python scripts/v3/20_filter_sweep.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "v3"))

from _v3_harness import (  # noqa: E402
    V2_FEATURES_BASE, V2_EDGAR_FEATURES, V2_TARGET,
    run_walkforward, replay_walkforward, save_result, print_comparison,
)

CACHE_DIR = ROOT / "data" / "v3_benchmarks" / "cache" / "v4_nometa"

# Production V3 hyperparameters (v3_hp_combo winner)
PROD_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg",
    "num_leaves": 31, "max_depth": 6,
    "learning_rate": 0.05, "min_child_samples": 30,
    "subsample": 0.75, "colsample_bytree": 0.7,
    "reg_lambda": 5.0,
    "n_jobs": 1, "seed": 42, "verbose": -1,
}

PRICE_GRID = [1.50, 2.00, 3.00]
ADV_GRID = [300_000, 500_000, 1_000_000]
COST_BPS = 15.0  # per side, for the final anchored baseline
MAX_SKIP_FRAC = 0.05


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    feat_cols = V2_FEATURES_BASE + V2_EDGAR_FEATURES
    need = ["date", "ticker", V2_TARGET, "atr_pct_20d", "close",
            "adv_usd_20d", "cs_spread_20d"] + feat_cols
    print(f"Loading features ({len(need)} of 517 cols)...")
    features = pd.read_parquet(ROOT / "data/processed/features_smallcap.parquet",
                               columns=need)
    features["date"] = pd.to_datetime(features["date"])
    print(f"  {len(features):,} rows, {features.ticker.nunique()} tickers")

    ohlcv = pd.read_parquet(ROOT / "data/processed/ohlcv_smallcap.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    return features, ohlcv, feat_cols


def main() -> None:
    t0 = time.time()
    features, ohlcv, feat_cols = load_data()

    # ── 1. Full training pass + cache + unfiltered re-baseline ──
    print("\n[1/3] Training 16-fold walk-forward (28 features, no meta, no filter)...")
    nofilt = run_walkforward(
        features, ohlcv, feat_cols,
        config_name="v4_nofilt_nometa",
        lgb_params=PROD_PARAMS, objective_lambdarank=True,
        cache_dir=CACHE_DIR, verbose=True,
    )
    save_result(nofilt)
    del features  # free ~28-col x 830K frame; replays only need the cache

    # ── 2. Replay the filter grid ──
    print(f"\n[2/3] Replaying {len(PRICE_GRID) * len(ADV_GRID)} filter combos from cache...")
    results = {}
    for price in PRICE_GRID:
        for adv in ADV_GRID:
            name = f"v4_filt_p{price:g}_a{adv // 1000}k"
            res = replay_walkforward(
                CACHE_DIR, ohlcv, config_name=name,
                min_price=price, min_adv_usd=adv,
            )
            res.feat_cols = feat_cols
            res.n_features = len(feat_cols)
            save_result(res)
            results[(price, adv)] = res
            agg = res.aggregate()
            print(f"  {name:24s} ICtr={agg['mean_ic_tradable']:+.4f} "
                  f"ret={agg['mean_return']:+7.1%} WR={agg['mean_win_rate']:.1%} "
                  f"sharpe={agg['mean_sharpe']:+.2f} +folds={agg['folds_positive_ret']} "
                  f"skipped={agg['skipped_rebalances']}")

    # ── 3. Pick winner: max tradable-IC s.t. <5% skipped; stricter wins ties ──
    print("\n[3/3] Selecting winner...")
    candidates = []
    for (price, adv), res in results.items():
        agg = res.aggregate()
        n_skipped = agg["skipped_rebalances"]
        # Attempted rebalances ≈ filled (n_trades/TOP_K, partial fills rare) + skipped
        n_attempted = sum(round(f.n_trades / 8) + f.n_skipped_rebalances for f in res.folds)
        skip_frac = n_skipped / n_attempted if n_attempted else 1.0
        candidates.append((price, adv, agg["mean_ic_tradable"], skip_frac, res))
    eligible = [c for c in candidates if c[3] < MAX_SKIP_FRAC]
    if not eligible:
        print("  !! No combo passes the availability constraint — using least-skipping")
        eligible = sorted(candidates, key=lambda c: c[3])[:1]
    # max IC; ties (within 0.0005) -> stricter (higher price, then higher adv)
    eligible.sort(key=lambda c: (round(c[2], 4), c[0], c[1]), reverse=True)
    w_price, w_adv, w_ic, w_skip, w_res = eligible[0]
    print(f"  WINNER: price>=${w_price:g}, ADV>=${w_adv:,} "
          f"(ICtr={w_ic:+.4f}, skipped={w_skip:.1%} of rebalances)")

    # Anchor: winning filter + 15 bps/side cost = v4_filt_baseline
    anchored = replay_walkforward(
        CACHE_DIR, ohlcv, config_name="v4_filt_baseline",
        min_price=w_price, min_adv_usd=w_adv, cost_bps=COST_BPS,
    )
    anchored.feat_cols = feat_cols
    anchored.n_features = len(feat_cols)
    save_result(anchored)

    # Spread-aware costing variant, reporting only
    spread_cost = replay_walkforward(
        CACHE_DIR, ohlcv, config_name="v4_filt_baseline_spreadcost",
        min_price=w_price, min_adv_usd=w_adv, cost_bps=COST_BPS,
        use_spread_cost=True,
    )
    spread_cost.feat_cols = feat_cols
    spread_cost.n_features = len(feat_cols)
    save_result(spread_cost)

    print_comparison(nofilt, w_res, anchored, spread_cost)

    # Persist the chosen thresholds for production + later phases
    decision = {
        "min_price": w_price, "min_adv_usd": w_adv, "cost_bps": COST_BPS,
        "rule": "max tradable-IC s.t. <5% skipped rebalances; stricter wins ties",
        "anchor_config": "v4_filt_baseline",
        "decided": pd.Timestamp.now().isoformat(),
    }
    fp = ROOT / "data" / "v3_benchmarks" / "v4_filter_decision.json"
    fp.write_text(json.dumps(decision, indent=2))
    print(f"\nDecision saved to {fp}")
    print(f"Total runtime: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
