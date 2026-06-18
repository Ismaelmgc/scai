"""V4 experiment: train the ranker on the triple-barrier (realized) target.

Hypothesis (the most promising untested lever): the product is the ceiling because
we OPTIMIZE the wrong objective — the model ranks by raw fwd_ret_20d_sector_rel but
the book earns the return under our trailing-stop/40%-PT/adaptive exit. Relabel
with the realized return under that exit (build_triple_barrier.py) and retrain, so
the LambdaRank ranks by what actually gets booked.

Two target variants vs the raw-target control (v4_fs_base28, 250 trees, same
features/universe): tb_abs (realized return) and tb_secrel (sector-relative
realized return). Top-8 under adaptive6_pt40, MC paired bootstrap on return.

Usage:
    PYTHONPATH=src python scripts/v3/40_triple_barrier.py
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

_spec = importlib.util.spec_from_file_location("mc", ROOT / "scripts/v3/34_mc_validate.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

DECISION_FP = ROOT / "data" / "v3_benchmarks" / "v4_filter_decision.json"
TB_FP = ROOT / "data" / "research" / "tb_labels.parquet"
CACHE_ROOT = ROOT / "data" / "v3_benchmarks" / "cache"
CTRL_CACHE = CACHE_ROOT / "v4_fs_base28"
BASE = V2_FEATURES_BASE + V2_EDGAR_FEATURES
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

    need = (["date", "ticker", V2_TARGET, "sector", "atr_pct_20d", "close",
             "adv_usd_20d", "cs_spread_20d"] + BASE)
    need = list(dict.fromkeys(need))
    feats = pd.read_parquet(ROOT / "data/processed/features_smallcap.parquet", columns=need)
    feats["date"] = pd.to_datetime(feats["date"])
    tb = pd.read_parquet(TB_FP)
    tb["date"] = pd.to_datetime(tb["date"])
    feats = feats.merge(tb, on=["date", "ticker"], how="left")
    sec_mean = feats.groupby(["date", "sector"])["tb_ret"].transform("mean")
    feats["tb_secrel"] = feats["tb_ret"] - sec_mean

    ohlcv = pd.read_parquet(ROOT / "data/processed/ohlcv_smallcap.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    ctrl = mc.replay_per_fold(CTRL_CACHE, ohlcv, mp, ma, cb, pol, 8)
    print(f"\n  Triple-barrier target vs raw-target control (top-8 adaptive6_pt40, "
          f"{cb:g}bps, 250 trees)\n")
    print(f"  {'target':16s} {'ret':>7} {'WR':>4} {'Sharpe':>6} {'+folds':>6} "
          f"{'diff vs ctrl [95% CI]':>26}")
    print(f"  {'raw (control)':16s} {ctrl['total_return'].mean():+7.1%} {wr(ctrl):4.0%} "
          f"{ctrl['sharpe'].mean():+6.2f} {int((ctrl['total_return']>0).sum()):>4}/16")

    for name, col in [("tb_abs", "tb_ret"), ("tb_secrel", "tb_secrel")]:
        f2 = feats.copy()
        f2[V2_TARGET] = f2[col]                      # train/bin on the TB target
        cache_dir = CACHE_ROOT / f"v4_{name}"
        run_walkforward(
            f2, ohlcv, BASE, config_name=f"v4_{name}", cache_dir=cache_dir,
            lgb_params=PROD_PARAMS, objective_lambdarank=True, num_boost_round=250,
            min_price=mp, min_adv_usd=ma, cost_bps=cb, verbose=False,
        )
        df = mc.replay_per_fold(cache_dir, ohlcv, mp, ma, cb, pol, 8)
        pb = mc.paired_bootstrap(df["total_return"].values, ctrl["total_return"].values)
        flag = "  <-CI>0" if pb["ci_lo"] > 0 else ""
        print(f"  {name:16s} {df['total_return'].mean():+7.1%} {wr(df):4.0%} "
              f"{df['sharpe'].mean():+6.2f} {int((df['total_return']>0).sum()):>4}/16  "
              f"{pb['mean_diff']:+.1%} [{pb['ci_lo']:+.1%},{pb['ci_hi']:+.1%}]{flag}")

    print(f"\n  (CI>0 vs control = real improvement.)  Runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
