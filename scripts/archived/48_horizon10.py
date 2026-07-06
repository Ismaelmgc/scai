"""V4.1 experiment — 10-day horizon (breadth-through-time) vs the purged 20d baseline.

Hypothesis: IR ≈ IC·√breadth. Halving the holding period doubles the number of
independent bets per year; if the 10d sector-relative signal keeps even ~75% of
the 20d IC, risk-adjusted return rises. The cost side is the killer to test:
turnover doubles, so the round-trip cost is paid twice as often.

Setup: identical pipeline to 45's purged leg, except
  target fwd_ret_10d_sector_rel, HOLD_DAYS=10 (N_COHORTS=2), time_stop=10,
  purge_days=10 (= the label horizon; that's all the purge that's needed).
Same 28 features, LambdaRank 16 bins, 600 trees, tradability filter, 15bps.

GATE: paired bootstrap vs purged 20d baseline (same folds) with the session-wide
Šidák haircut (M=7), SELECT/CONFIRM split (folds 1-12 / 13-16), DSR > 0.95.
Also reported spread-aware, where doubled turnover must show up honestly.

Usage:
    PYTHONPATH=src python scripts/v3/48_horizon10.py [--trees N]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "v3"))

import _v3_harness as h  # noqa: E402

_spec = importlib.util.spec_from_file_location("mc", ROOT / "scripts/v3/34_mc_validate.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

DECISION_FP = ROOT / "data" / "v3_benchmarks" / "v4_filter_decision.json"
FEATURES_FP = ROOT / "data" / "processed" / "features_smallcap.parquet"
OHLCV_FP = ROOT / "data" / "processed" / "ohlcv_smallcap.parquet"
CACHE_ROOT = ROOT / "data" / "v3_benchmarks" / "cache"
FEAT = h.V2_FEATURES_BASE + h.V2_EDGAR_FEATURES

TARGET_10D = "fwd_ret_10d_sector_rel"
HOLD_10D = 10
SESSION_TRIALS = 7
SELECT_FOLDS = 12

PROD_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg",
    "num_leaves": 31, "max_depth": 6,
    "learning_rate": 0.05, "min_child_samples": 30,
    "subsample": 0.75, "colsample_bytree": 0.7,
    "reg_lambda": 5.0,
    "n_jobs": 1, "seed": 42, "verbose": -1,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=600)
    args = ap.parse_args()
    t0 = time.time()

    decision = json.loads(DECISION_FP.read_text())
    mp, ma, cb = decision["min_price"], decision["min_adv_usd"], decision["cost_bps"]

    need = (["date", "ticker", TARGET_10D, "atr_pct_20d", "close",
             "adv_usd_20d", "cs_spread_20d"] + FEAT)
    features = pd.read_parquet(FEATURES_FP, columns=need)
    features["date"] = pd.to_datetime(features["date"])
    ohlcv = pd.read_parquet(OHLCV_FP)
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    # Re-point the harness at the 10d problem. Module globals are read at call
    # time, so this changes target/hold/cohorts for run_walkforward and
    # _evaluate_fold without touching the file (research-only pattern).
    h.V2_TARGET = TARGET_10D
    h.HOLD_DAYS = HOLD_10D
    h.N_COHORTS = HOLD_10D // h.REBALANCE_EVERY  # 2 overlapping cohorts

    pol_dict = dict(decision["exit_policy"])
    pol_dict["time_stop"] = HOLD_10D
    policy = mc.policy_from(pol_dict)

    cache = CACHE_ROOT / "v4_h10_purged"
    if not (cache / "meta.json").exists():
        print(f"=== training 10d PURGED ({args.trees} trees, purge={HOLD_10D}d) ===")
        h.run_walkforward(
            features, ohlcv, FEAT, config_name="v4_h10_purged",
            lgb_params=PROD_PARAMS, objective_lambdarank=True, n_bins=16,
            min_price=mp, min_adv_usd=ma, cost_bps=cb, exit_policy=policy,
            cache_dir=cache, num_boost_round=args.trees, purge_days=HOLD_10D,
            verbose=True,
        )
    else:
        print("(cache exists, skipping training)")

    # Baseline = purged 20d, prod exit, same folds. Replay both at flat 15bps
    # and spread-aware. NOTE: h globals are patched to 10d — replay the 20d
    # baseline FIRST with restored globals to avoid corrupting it.
    meta10 = json.loads((cache / "meta.json").read_text())
    cache20 = CACHE_ROOT / "v4_purge_purged"
    meta20 = json.loads((cache20 / "meta.json").read_text())
    pol20 = mc.policy_from(decision["exit_policy"])

    def replay(cache_dir, meta, pol, target, hold, spread):
        h.V2_TARGET, h.HOLD_DAYS = target, hold
        h.N_COHORTS = hold // h.REBALANCE_EVERY
        rows = []
        for fm in meta:
            td = pd.read_parquet(Path(cache_dir) / f"fold_{fm['fold']:02d}.parquet")
            td["date"] = pd.to_datetime(td["date"])
            ev = h._evaluate_fold(
                td, ohlcv,
                (pd.Timestamp(fm["test_start"]), pd.Timestamp(fm["test_end"])),
                mp, ma, pol, cb, spread)
            rows.append({"fold": fm["fold"], **{k: ev[k] for k in
                         ("total_return", "sharpe", "win_rate", "n_trades", "max_dd")}})
        return pd.DataFrame(rows)

    print("\n" + "=" * 74)
    print("  10d purged vs 20d purged baseline — paired bootstrap, both cost models")
    print("=" * 74)
    verdicts = {}
    for label, spread in [("flat 15bps", False), ("spread-aware", True)]:
        b20 = replay(cache20, meta20, pol20, "fwd_ret_20d_sector_rel", 20, spread)
        b10 = replay(cache, meta10, policy, TARGET_10D, HOLD_10D, spread)
        n = min(len(b10), len(b20))
        b10, b20 = b10.iloc[:n], b20.iloc[:n]
        print(f"\n  --- cost {label} ---")
        mc.summarize(b20, "20d purged (base)")
        mc.summarize(b10, "10d purged")
        pb = mc.paired_bootstrap(b10["total_return"].values, b20["total_return"].values)
        p_sidak = mc.sidak(pb["p_le0"], SESSION_TRIALS)
        sel = (b10["total_return"][:SELECT_FOLDS] - b20["total_return"][:SELECT_FOLDS]).mean()
        conf = (b10["total_return"][SELECT_FOLDS:] - b20["total_return"][SELECT_FOLDS:]).mean()
        trial_sr_std = float(np.std([
            df["total_return"].mean() / df["total_return"].std() for df in (b10, b20)]))
        dsr = mc.def_sharpe(b10["total_return"].values, trial_sr_std, SESSION_TRIALS)
        print(f"  d10-20 {pb['mean_diff']:+.1%} [{pb['ci_lo']:+.1%},{pb['ci_hi']:+.1%}] "
              f"pSidak={p_sidak:.3f} sel{sel:+.1%} conf{conf:+.1%} DSR={dsr:.3f}")
        verdicts[label] = p_sidak < 0.05 and conf > 0 and dsr > 0.95

    ok = verdicts.get("flat 15bps") and verdicts.get("spread-aware")
    print(f"\n  GATE (both cost models must pass): "
          f"{'PROMOTE' if ok else 'REJECT — 10d horizon does not beat the purged 20d baseline'}")
    print(f"  Runtime {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
