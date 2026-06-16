"""V4 experiment: portfolio breadth sweep (TOP_K) — fundamental law of active mgmt.

IR ~= IC * sqrt(breadth). The strategy holds the top-8; if the IC is positive,
holding MORE names should diversify idiosyncratic noise and raise the Sharpe
(at the cost of average conviction). This is the cheap, no-new-data precursor to
the universe-expansion question: if more breadth helps here, widening the universe
(more candidates) is worth the data lift.

Replays the production-model cache (model unchanged) under the Phase-1 filter +
15bps for several TOP_K. Equal-weight throughout (matches production). Control
(top_k=8) reproduces v4_filt_baseline.

Usage:
    PYTHONPATH=src python scripts/v3/28_breadth_sweep.py
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
    ExitPolicy,
    FoldMetrics,
    RunResult,
    _evaluate_fold,
)

CACHE_DIR = ROOT / "data" / "v3_benchmarks" / "cache" / "v4_nometa"
DECISION_FP = ROOT / "data" / "v3_benchmarks" / "v4_filter_decision.json"
BASELINE_FP = ROOT / "data" / "v3_benchmarks" / "v4_filt_baseline.json"
TOP_KS = [8, 10, 12, 15, 20]


def main() -> None:
    t0 = time.time()
    decision = json.loads(DECISION_FP.read_text())
    base_agg = json.loads(BASELINE_FP.read_text())["aggregate"]
    mp, ma, cb = decision["min_price"], decision["min_adv_usd"], decision["cost_bps"]

    ohlcv = pd.read_parquet(ROOT / "data/processed/ohlcv_smallcap.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    meta = json.loads((CACHE_DIR / "meta.json").read_text())
    folds = []
    for fm in meta:
        td = pd.read_parquet(CACHE_DIR / f"fold_{fm['fold']:02d}.parquet")
        td["date"] = pd.to_datetime(td["date"])
        folds.append((fm, td))
    policy = ExitPolicy()

    def eval_k(k: int) -> dict:
        rr = RunResult(config_name=f"top{k}", feat_cols=[], n_features=0)
        for fm, td in folds:
            ev = _evaluate_fold(
                td, ohlcv,
                (pd.Timestamp(fm["test_start"]), pd.Timestamp(fm["test_end"])),
                mp, ma, policy, cb, False, top_k=k,
            )
            rr.folds.append(FoldMetrics(
                fold=fm["fold"], period=fm["period"],
                train_rows=fm["train_rows"], test_rows=fm["test_rows"],
                mean_ic=fm["mean_ic"], ic_ir=fm["ic_ir"], hit_rate_ic=fm["hit_rate_ic"],
                total_return=ev["total_return"], sharpe=ev["sharpe"], max_dd=ev["max_dd"],
                n_trades=ev["n_trades"], win_rate=ev["win_rate"],
                median_return=ev["median_return"], market_return=ev["market_return"],
                mean_ic_tradable=ev["mean_ic_tradable"],
                n_skipped_rebalances=ev["n_skipped_rebalances"],
                avg_candidates=ev["avg_candidates"],
            ))
        return rr.aggregate()

    print(f"\n  Breadth sweep (TOP_K) — 16 folds, {cb:g}bps\n")
    hdr = "  TOP_K    netRet   Sharpe    WR    +folds   trades"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    print(f"  BASE(8) {base_agg['mean_return']:+7.1%}  {base_agg['mean_sharpe']:+5.2f}  "
          f"{base_agg['mean_win_rate']:4.0%}  {base_agg['folds_positive_ret']:>5}   "
          f"{base_agg['total_trades']:5d}")
    print("  " + "-" * (len(hdr) - 2))
    for k in TOP_KS:
        a = eval_k(k)
        tag = "  (=control)" if k == 8 else ""
        print(f"  {k:5d}   {a['mean_return']:+7.1%}  {a['mean_sharpe']:+5.2f}  "
              f"{a['mean_win_rate']:4.0%}  {a['folds_positive_ret']:>5}   "
              f"{a['total_trades']:5d}{tag}")

    print("\n  (Sharpe sube con breadth si IR~=IC*sqrt(N) aplica. Control(8) ~ baseline.)")
    print(f"  Runtime: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
