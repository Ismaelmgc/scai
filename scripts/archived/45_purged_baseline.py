"""V4 validation: is the deployed baseline inflated by the 20-day label leak?

The harness's walk-forward trains right up to test_start with NO purge. The target
is a 20-trading-day forward return, so the last ~20 train days carry labels that
reach INTO the test window — classic look-ahead overlap (López de Prado purging).
This re-measures the DEPLOYED baseline (28 features, LambdaRank 16-bin, tradability
filter, 15bps, prod exits) twice — unpurged vs purged (drop the last HOLD_DAYS of
each fold's train) — and asks, via the MC paired bootstrap (34), whether the
difference is real or noise.

Reading the verdict:
  - purged ≈ unpurged (CI spans 0) → the +4%/WR is NOT a leak artefact; trust it.
  - purged << unpurged (CI < 0)    → the leak was inflating; the honest baseline is
                                     lower and every prior experiment must be
                                     re-judged against the purged number.

Trains 2× 16 folds (heavy) but caches predictions, then replays BOTH prod exit
policies for free. No production touched.

Usage:
    PYTHONPATH=src python scripts/v3/45_purged_baseline.py [--trees N]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "v3"))

from _v3_harness import (  # noqa: E402
    HOLD_DAYS,
    TOP_K,
    V2_EDGAR_FEATURES,
    V2_FEATURES_BASE,
    V2_TARGET,
    run_walkforward,
)

_spec = importlib.util.spec_from_file_location("mc", ROOT / "scripts/v3/34_mc_validate.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

DECISION_FP = ROOT / "data" / "v3_benchmarks" / "v4_filter_decision.json"
FEATURES_FP = ROOT / "data" / "processed" / "features_smallcap.parquet"
OHLCV_FP = ROOT / "data" / "processed" / "ohlcv_smallcap.parquet"
CACHE_ROOT = ROOT / "data" / "v3_benchmarks" / "cache"
FEAT_BASE = V2_FEATURES_BASE + V2_EDGAR_FEATURES  # deployed 28 features

PROD_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg",
    "num_leaves": 31, "max_depth": 6,
    "learning_rate": 0.05, "min_child_samples": 30,
    "subsample": 0.75, "colsample_bytree": 0.7,
    "reg_lambda": 5.0,
    "n_jobs": 1, "seed": 42, "verbose": -1,
}


def train_cache(features, ohlcv, decision, cache_dir, purge_days, trees) -> None:
    run_walkforward(
        features, ohlcv, FEAT_BASE,
        config_name=cache_dir.name,
        lgb_params=PROD_PARAMS, objective_lambdarank=True, n_bins=16,
        min_price=decision["min_price"], min_adv_usd=decision["min_adv_usd"],
        cost_bps=decision["cost_bps"],
        exit_policy=mc.policy_from(decision["exit_policy"]),
        cache_dir=cache_dir, num_boost_round=trees, purge_days=purge_days,
        verbose=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=600, help="num_boost_round (prod=600)")
    args = ap.parse_args()
    t0 = time.time()

    decision = json.loads(DECISION_FP.read_text())
    mp, ma, cb = decision["min_price"], decision["min_adv_usd"], decision["cost_bps"]

    need = (["date", "ticker", V2_TARGET, "atr_pct_20d", "close",
             "adv_usd_20d", "cs_spread_20d"] + FEAT_BASE)
    features = pd.read_parquet(FEATURES_FP, columns=need)
    features["date"] = pd.to_datetime(features["date"])
    ohlcv = pd.read_parquet(OHLCV_FP)
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    cache_up = CACHE_ROOT / "v4_purge_unpurged"
    cache_pg = CACHE_ROOT / "v4_purge_purged"
    print(f"=== training UNPURGED ({args.trees} trees) ===")
    train_cache(features, ohlcv, decision, cache_up, 0, args.trees)
    print(f"\n=== training PURGED (purge_days={HOLD_DAYS}) ===")
    train_cache(features, ohlcv, decision, cache_pg, HOLD_DAYS, args.trees)

    print("\n" + "=" * 74)
    print("  PURGED vs UNPURGED — deployed baseline, 10k paired bootstrap")
    print("=" * 74)
    for pname, pkey in [("pt40", "exit_policy"), ("adaptive6_pt40", "exit_policy_adaptive")]:
        pol = mc.policy_from(decision[pkey])
        up = mc.replay_per_fold(cache_up, ohlcv, mp, ma, cb, pol, TOP_K)
        pg = mc.replay_per_fold(cache_pg, ohlcv, mp, ma, cb, pol, TOP_K)
        print(f"\n  --- exit {pname} ---")
        mc.summarize(up, "unpurged")
        mc.summarize(pg, "purged")
        pb = mc.paired_bootstrap(pg["total_return"].values, up["total_return"].values)
        wr_up = (up["win_rate"] * up["n_trades"]).sum() / up["n_trades"].sum()
        wr_pg = (pg["win_rate"] * pg["n_trades"]).sum() / pg["n_trades"].sum()
        verdict = ("LEAK INFLATED (purged significantly lower)" if pb["ci_hi"] < 0
                   else "ROBUST (CI spans 0 — not a leak artefact)" if pb["ci_lo"] <= 0
                   else "purged HIGHER (unexpected)")
        print(f"  Δret purged-unpurged {pb['mean_diff']:+.1%} "
              f"[95% {pb['ci_lo']:+.1%},{pb['ci_hi']:+.1%}] p(≤0)={pb['p_le0']:.2f}  "
              f"ΔWR {wr_pg - wr_up:+.1%}")
        print(f"  -> {verdict}")

    print(f"\n  done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
