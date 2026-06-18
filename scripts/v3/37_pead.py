"""V4 experiment: PEAD (price-reaction earnings drift) through the harness + MC.

Coverage gate (download_earnings.py) + IC screen (build_pead_features.py): the
signal is real and correctly signed (pead_react median IC +0.021) but SPARSE
(24% of tradable rows; delisted names lack a current CIK) and weaker than features
that already failed the product gate. One clean harness test before concluding.

Adds pead_react + pead_react_decay to the 28 production features, retrains the
16-fold walk-forward (cached, 250 trees to match the search control v4_fs_base28),
replays top-8 under adaptive6_pt40, and MC paired-bootstraps vs the matched
250-tree baseline.

Usage:
    PYTHONPATH=src python scripts/v3/37_pead.py
"""
from __future__ import annotations

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
    V2_EDGAR_FEATURES,
    V2_FEATURES_BASE,
    V2_TARGET,
    run_walkforward,
)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "v3" / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_leak = _load("18_verify_no_leak.py", "verify_no_leak")
mc = _load("34_mc_validate.py", "mc_validate")

DECISION_FP = ROOT / "data" / "v3_benchmarks" / "v4_filter_decision.json"
CACHE_ROOT = ROOT / "data" / "v3_benchmarks" / "cache"
PEAD_FP = ROOT / "data" / "pead_features.parquet"
BASE = V2_FEATURES_BASE + V2_EDGAR_FEATURES
PEAD = ["pead_react", "pead_react_decay"]
PROD_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg",
    "num_leaves": 31, "max_depth": 6, "learning_rate": 0.05,
    "min_child_samples": 30, "subsample": 0.75, "colsample_bytree": 0.7,
    "reg_lambda": 5.0, "n_jobs": 1, "seed": 42, "verbose": -1,
}


def wr(df):
    return (df["win_rate"] * df["n_trades"]).sum() / df["n_trades"].sum()


def main() -> None:
    t0 = time.time()
    decision = json.loads(DECISION_FP.read_text())
    mp, ma, cb = decision["min_price"], decision["min_adv_usd"], decision["cost_bps"]
    pol = mc.policy_from(decision["exit_policy_adaptive"])

    need = (["date", "ticker", V2_TARGET, "atr_pct_20d", "close", "adv_usd_20d",
             "cs_spread_20d"] + BASE)
    need = list(dict.fromkeys(need))
    feats = pd.read_parquet(ROOT / "data/processed/features_smallcap.parquet", columns=need)
    feats["date"] = pd.to_datetime(feats["date"])
    pead = pd.read_parquet(PEAD_FP)
    pead["date"] = pd.to_datetime(pead["date"])
    feats = feats.merge(pead[["date", "ticker"] + PEAD], on=["date", "ticker"], how="left")
    ohlcv = pd.read_parquet(ROOT / "data/processed/ohlcv_smallcap.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    samp = feats.dropna(subset=[V2_TARGET]).sample(min(200_000, len(feats)), random_state=42)
    errs = (_leak.check_pearson(samp, PEAD) + _leak.check_per_date_spearman(samp, PEAD)
            + _leak.check_degenerate(samp, PEAD))
    print("\n  leak gate on PEAD features:", errs if errs else "CLEAN")

    cache_dir = CACHE_ROOT / "v4_pead"
    run_walkforward(
        feats, ohlcv, BASE + PEAD, config_name="v4_pead", cache_dir=cache_dir,
        lgb_params=PROD_PARAMS, objective_lambdarank=True, num_boost_round=250,
        min_price=mp, min_adv_usd=ma, cost_bps=cb, verbose=False,
    )

    base = mc.replay_per_fold(CACHE_ROOT / "v4_fs_base28", ohlcv, mp, ma, cb, pol, 8)
    cand = mc.replay_per_fold(cache_dir, ohlcv, mp, ma, cb, pol, 8)
    pb = mc.paired_bootstrap(cand["total_return"].values, base["total_return"].values)

    print(f"\n  PEAD vs 28f baseline (matched 250 trees, top-8, adaptive6_pt40, {cb:g}bps)\n")
    print(f"  {'config':16s} {'ret':>6} {'WR':>4} {'Sharpe':>6} {'worstDD':>7}")
    print(f"  {'base28':16s} {base['total_return'].mean():+6.1%} {wr(base):4.0%} "
          f"{base['sharpe'].mean():+6.2f} {base['max_dd'].min():+7.1%}")
    print(f"  {'base28+PEAD':16s} {cand['total_return'].mean():+6.1%} {wr(cand):4.0%} "
          f"{cand['sharpe'].mean():+6.2f} {cand['max_dd'].min():+7.1%}")
    print(f"\n  diff {pb['mean_diff']:+.1%} [95% {pb['ci_lo']:+.1%}, {pb['ci_hi']:+.1%}]  "
          f"p(diff<=0)={pb['p_le0']:.2f}")
    verdict = "REAL (CI>0)" if pb["ci_lo"] > 0 else "noise (CI spans 0)"
    print(f"  -> PEAD effect on the product is {verdict}")
    print(f"\n  Runtime: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
