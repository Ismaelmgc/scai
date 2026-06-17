"""V4 experiment: residual momentum + 52w-high + low-IVOL via the harness.

Round-1 lesson: raw 12-1 momentum DOUBLED ranking IC but CRASHED the top-8 book.
This round tests the literature's de-crashed alternatives, judged primarily on the
PRODUCT (top-8 / top-6 return, WR, Sharpe AND worst-fold drawdown), not just IC:

  residmom_12_1 / residmom_6_1  (Blitz residual momentum; data/residmom_features.parquet)
  pct_from_52w_high             (George-Hwang; already in the parquet, unused)
  idio_vol_60d                  (low-IVOL anomaly; already in the parquet, unused)

Standalone per-date IC triage (tradable cross-section) found pct_from_52w_high
+0.087, idio_vol_60d -0.090, residmom_12_1 +0.039 (reversal family was noise →
dropped). High IC != better top-8, so each set runs:
  (A) gate vs v4_filt_baseline, top-8, default policy  [22_feature_batches criteria]
  (B) crash test: replay cache under production exit policies at top_k {8,6} with
      worst-fold maxDD, vs the baseline cache v4_nometa  [mirrors 30].

Usage:
    PYTHONPATH=src python scripts/v3/33_residual_momentum.py
    PYTHONPATH=src python scripts/v3/33_residual_momentum.py residmom hi52w
"""
from __future__ import annotations

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

from _v3_harness import (  # noqa: E402
    V2_EDGAR_FEATURES,
    V2_FEATURES_BASE,
    V2_TARGET,
    ExitPolicy,
    FoldMetrics,
    RunResult,
    _evaluate_fold,
    run_walkforward,
    save_result,
)

_spec = importlib.util.spec_from_file_location(
    "verify_no_leak", ROOT / "scripts" / "v3" / "18_verify_no_leak.py")
_leak = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_leak)

DECISION_FP = ROOT / "data" / "v3_benchmarks" / "v4_filter_decision.json"
BASELINE_FP = ROOT / "data" / "v3_benchmarks" / "v4_filt_baseline.json"
RESIDMOM_FP = ROOT / "data" / "residmom_features.parquet"
CACHE_ROOT = ROOT / "data" / "v3_benchmarks" / "cache"
BASE_CACHE = CACHE_ROOT / "v4_nometa"

# Features pulled from the parquet that are NOT in production (52w-high, idio-vol).
EXTRA_FROM_PARQUET = ["pct_from_52w_high", "idio_vol_60d"]

SETS = {
    "residmom": ["residmom_12_1", "residmom_6_1"],
    "hi52w": ["pct_from_52w_high"],
    "lowvol": ["idio_vol_60d"],
    "hi52w_lowvol": ["pct_from_52w_high", "idio_vol_60d"],
    "combo": ["residmom_12_1", "pct_from_52w_high", "idio_vol_60d"],
}

PROD_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg",
    "num_leaves": 31, "max_depth": 6,
    "learning_rate": 0.05, "min_child_samples": 30,
    "subsample": 0.75, "colsample_bytree": 0.7,
    "reg_lambda": 5.0,
    "n_jobs": 1, "seed": 42, "verbose": -1,
}
KS = [8, 6]


def _policy(d: dict) -> ExitPolicy:
    fields = set(ExitPolicy.__dataclass_fields__)
    return ExitPolicy(**{k: v for k, v in d.items() if k in fields})


def evaluate_cache(cache_dir: Path, ohlcv, mp, ma, cb, policy: ExitPolicy, k: int) -> dict:
    """Replay cached predictions under a given exit policy / book size (mirror 30)."""
    meta = json.loads((cache_dir / "meta.json").read_text())
    rr = RunResult(config_name="x", feat_cols=[], n_features=0)
    for fm in meta:
        td = pd.read_parquet(cache_dir / f"fold_{fm['fold']:02d}.parquet")
        td["date"] = pd.to_datetime(td["date"])
        ev = _evaluate_fold(
            td, ohlcv, (pd.Timestamp(fm["test_start"]), pd.Timestamp(fm["test_end"])),
            mp, ma, policy, cb, False, top_k=k,
        )
        rr.folds.append(FoldMetrics(
            fold=fm["fold"], period=fm["period"], train_rows=fm["train_rows"],
            test_rows=fm["test_rows"], mean_ic=fm["mean_ic"], ic_ir=fm["ic_ir"],
            hit_rate_ic=fm["hit_rate_ic"], total_return=ev["total_return"],
            sharpe=ev["sharpe"], max_dd=ev["max_dd"], n_trades=ev["n_trades"],
            win_rate=ev["win_rate"], median_return=ev["median_return"],
            market_return=ev["market_return"], mean_ic_tradable=ev["mean_ic_tradable"],
            n_skipped_rebalances=ev["n_skipped_rebalances"], avg_candidates=ev["avg_candidates"],
        ))
    a = rr.aggregate()
    a["worst_dd"] = float(np.min([f.max_dd for f in rr.folds]))
    return a


def run_set(name, cols, features, ohlcv, decision, base_agg) -> Path | None:
    feat_base = V2_FEATURES_BASE + V2_EDGAR_FEATURES
    train = features.dropna(subset=[V2_TARGET])
    sample = train.sample(min(200_000, len(train)), random_state=42)
    errors = (_leak.check_feature_names(cols)
              + _leak.check_degenerate(sample, cols)
              + _leak.check_pearson(sample, cols)
              + _leak.check_per_date_spearman(sample, cols))
    if errors:
        print(f"  [{name}] LEAK GATE FAILED: {errors}")
        return None
    cache_dir = CACHE_ROOT / f"v4_rm_{name}"
    res = run_walkforward(
        features, ohlcv, feat_base + cols,
        config_name=f"v4_rm_{name}", cache_dir=cache_dir,
        lgb_params=PROD_PARAMS, objective_lambdarank=True,
        min_price=decision["min_price"], min_adv_usd=decision["min_adv_usd"],
        cost_bps=decision["cost_bps"], verbose=False,
    )
    save_result(res)
    a = res.aggregate()
    base_pos = int(base_agg["folds_positive_ret"].split("/")[0])
    cand_pos = int(a["folds_positive_ret"].split("/")[0])
    c1 = a["mean_ic_tradable"] >= base_agg["mean_ic_tradable"] + 0.002
    c2 = cand_pos >= base_pos
    c3 = (a["mean_return"] >= 0.95 * base_agg["mean_return"]
          and a["mean_win_rate"] >= base_agg["mean_win_rate"] - 0.01)
    strict = sum([a["mean_ic_tradable"] > base_agg["mean_ic_tradable"],
                  a["mean_return"] > base_agg["mean_return"],
                  a["mean_win_rate"] > base_agg["mean_win_rate"]])
    verdict = "PROMOTE" if (c1 and c2 and c3 and strict >= 2) else "REJECT"
    print(f"  [{name}] {verdict}  "
          f"ICtr {base_agg['mean_ic_tradable']:+.4f}->{a['mean_ic_tradable']:+.4f}  "
          f"ret {base_agg['mean_return']:+.1%}->{a['mean_return']:+.1%}  "
          f"WR {base_agg['mean_win_rate']:.1%}->{a['mean_win_rate']:.1%}  "
          f"Sharpe {base_agg['mean_sharpe']:+.2f}->{a['mean_sharpe']:+.2f}  "
          f"+folds {base_agg['folds_positive_ret']}->{a['folds_positive_ret']}  "
          f"[c1={c1} c2={c2} c3={c3} strict={strict}]")
    return cache_dir


def main() -> None:
    t0 = time.time()
    decision = json.loads(DECISION_FP.read_text())
    base_agg = json.loads(BASELINE_FP.read_text())["aggregate"]
    mp, ma, cb = decision["min_price"], decision["min_adv_usd"], decision["cost_bps"]
    policies = {
        "pt40": _policy(decision["exit_policy"]),
        "adaptive6_pt40": _policy(decision["exit_policy_adaptive"]),
    }
    requested = [s for s in sys.argv[1:] if s in SETS] or list(SETS)

    feat_base = V2_FEATURES_BASE + V2_EDGAR_FEATURES
    need = (["date", "ticker", V2_TARGET, "atr_pct_20d", "close", "adv_usd_20d",
             "cs_spread_20d"] + feat_base + EXTRA_FROM_PARQUET)
    need = list(dict.fromkeys(need))
    features = pd.read_parquet(ROOT / "data/processed/features_smallcap.parquet", columns=need)
    features["date"] = pd.to_datetime(features["date"])
    rm = pd.read_parquet(RESIDMOM_FP)
    rm["date"] = pd.to_datetime(rm["date"])
    features = features.merge(rm, on=["date", "ticker"], how="left")

    ohlcv = pd.read_parquet(ROOT / "data/processed/ohlcv_smallcap.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    print(f"\n  (A) GATE — top-8, default policy, vs baseline "
          f"(ICtr {base_agg['mean_ic_tradable']:+.4f}, ret {base_agg['mean_return']:+.1%}, "
          f"WR {base_agg['mean_win_rate']:.1%})\n")
    caches = {}
    for name in requested:
        c = run_set(name, SETS[name], features, ohlcv, decision, base_agg)
        if c is not None:
            caches[name] = c

    # ── (B) crash test: production policies, top_k {8,6}, worst-fold DD ──
    print("\n  (B) CRASH TEST — production exit policies, top_k {8,6}, worst-fold DD\n")
    for pname, pol in policies.items():
        print(f"  === {pname} ===")
        print(f"  {'config':18s} {'K':>2}  {'netRet':>7} {'Sharpe':>6} {'WR':>5} "
              f"{'+folds':>6} {'worstDD':>7}")
        for k in KS:
            a = evaluate_cache(BASE_CACHE, ohlcv, mp, ma, cb, pol, k)
            print(f"  {'baseline (28f)':18s} {k:>2}  {a['mean_return']:+7.1%} "
                  f"{a['mean_sharpe']:+6.2f} {a['mean_win_rate']:4.0%} "
                  f"{a['folds_positive_ret']:>6} {a['worst_dd']:+7.1%}")
        for name, cdir in caches.items():
            for k in KS:
                a = evaluate_cache(cdir, ohlcv, mp, ma, cb, pol, k)
                print(f"  {name:18s} {k:>2}  {a['mean_return']:+7.1%} "
                      f"{a['mean_sharpe']:+6.2f} {a['mean_win_rate']:4.0%} "
                      f"{a['folds_positive_ret']:>6} {a['worst_dd']:+7.1%}")
        print()

    print(f"  Total runtime: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
