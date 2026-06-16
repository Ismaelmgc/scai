"""V4 validation: tighter book (TOP_K) under the PRODUCTION exit policies.

The concentration sweep (29) used the default exit policy. The deployed portfolios
use pt40 (baseline) and adaptive6_pt40 (adaptive). Before recommending any TOP_K
change, re-check the tighter books under the real exit policies AND report
drawdown — concentration's cost is higher idiosyncratic risk. Cache replay
(model unchanged), 15bps, K in {8 control, 6, 5, 4}.

Usage:
    PYTHONPATH=src python scripts/v3/30_concentration_validate.py
"""
from __future__ import annotations

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
    ExitPolicy,
    FoldMetrics,
    RunResult,
    _evaluate_fold,
)

CACHE_DIR = ROOT / "data" / "v3_benchmarks" / "cache" / "v4_nometa"
DECISION_FP = ROOT / "data" / "v3_benchmarks" / "v4_filter_decision.json"
KS = [8, 6, 5, 4]
_POLICY_FIELDS = {f for f in ExitPolicy.__dataclass_fields__}


def _policy(d: dict) -> ExitPolicy:
    return ExitPolicy(**{k: v for k, v in d.items() if k in _POLICY_FIELDS})


def main() -> None:
    t0 = time.time()
    decision = json.loads(DECISION_FP.read_text())
    mp, ma, cb = decision["min_price"], decision["min_adv_usd"], decision["cost_bps"]
    policies = {
        "pt40": _policy(decision["exit_policy"]),
        "adaptive6_pt40": _policy(decision["exit_policy_adaptive"]),
    }

    ohlcv = pd.read_parquet(ROOT / "data/processed/ohlcv_smallcap.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    meta = json.loads((CACHE_DIR / "meta.json").read_text())
    folds = []
    for fm in meta:
        td = pd.read_parquet(CACHE_DIR / f"fold_{fm['fold']:02d}.parquet")
        td["date"] = pd.to_datetime(td["date"])
        folds.append((fm, td))

    def evaluate(policy: ExitPolicy, k: int):
        rr = RunResult(config_name="x", feat_cols=[], n_features=0)
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
        a = rr.aggregate()
        dds = [f.max_dd for f in rr.folds]
        a["worst_dd"] = float(np.min(dds))
        a["mean_dd"] = float(np.mean(dds))
        return a

    print(f"\n  Concentration under production exit policies — 16 folds, {cb:g}bps\n")
    for pname, pol in policies.items():
        print(f"  === {pname} ===")
        print("  TOP_K   netRet   Sharpe    WR    +folds   worstDD  meanDD")
        for k in KS:
            a = evaluate(pol, k)
            tag = "  <- prod" if k == 8 else ""
            print(f"  {k:5d}  {a['mean_return']:+7.1%}  {a['mean_sharpe']:+5.2f}  "
                  f"{a['mean_win_rate']:4.0%}  {a['folds_positive_ret']:>5}   "
                  f"{a['worst_dd']:+6.1%}  {a['mean_dd']:+6.1%}{tag}")
        print()

    print(f"  Runtime: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
