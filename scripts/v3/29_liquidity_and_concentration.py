"""V4 analysis (cache-only, fast): two de-risk readouts before bigger bets.

A) Liquidity-tercile IC — does signal quality hold for the most-liquid (mid-cap-
   like) names? If the high-ADV tercile has comparable/better per-date IC than the
   low-ADV tercile, expanding the universe toward mid-caps is worth the data lift;
   if it's much worse, expansion likely won't help (mid-caps too efficient).

B) Concentration sweep — tighten TOP_K below 8 (3..8). If a tighter book lifts WR
   and Sharpe, the edge is precision-concentrated and a meta-labeling/precision
   filter is promising (cheap proxy for it).

Both replay the production-model cache (model unchanged). Read-only analysis.

Usage:
    PYTHONPATH=src python scripts/v3/29_liquidity_and_concentration.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "v3"))

from _v3_harness import (  # noqa: E402
    V2_TARGET,
    ExitPolicy,
    FoldMetrics,
    RunResult,
    _evaluate_fold,
)

from app.features.tradability import tradable_mask  # noqa: E402

CACHE_DIR = ROOT / "data" / "v3_benchmarks" / "cache" / "v4_nometa"
DECISION_FP = ROOT / "data" / "v3_benchmarks" / "v4_filter_decision.json"
BASELINE_FP = ROOT / "data" / "v3_benchmarks" / "v4_filt_baseline.json"
TIGHT_KS = [3, 4, 5, 6, 8]


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

    # ── A) Liquidity-tercile per-date IC ──
    ic_buckets = {0: [], 1: [], 2: []}  # 0=low ADV, 2=high ADV
    for _, td in folds:
        t = td[tradable_mask(td, mp, ma)]
        for _, day in t.groupby("date"):
            if len(day) < 30:
                continue
            q = pd.qcut(day["adv_usd_20d"], 3, labels=False, duplicates="drop")
            for b in (0, 1, 2):
                g = day[q == b]
                if len(g) >= 10 and g["pred"].nunique() > 1:
                    ic, _ = spearmanr(g["pred"], g[V2_TARGET])
                    if not np.isnan(ic):
                        ic_buckets[b].append(ic)
    print("\n  A) Per-date IC by ADV tercile (tradable cross-section):")
    print("     tercile        mean IC    n_dates   median ADV$")
    for b, name in [(0, "low ADV"), (1, "mid ADV"), (2, "high ADV")]:
        ics = ic_buckets[b]
        print(f"     {name:9s}    {np.mean(ics):+.4f}    {len(ics):5d}")
    print("     (high-ADV ~ mid-cap proxy; comparable IC => expansion worth testing)")

    # ── B) Concentration sweep ──
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

    print("\n  B) Concentration sweep (tighten TOP_K):")
    print("     TOP_K   netRet   Sharpe    WR    +folds")
    print(f"     base8  {base_agg['mean_return']:+7.1%}  {base_agg['mean_sharpe']:+5.2f}  "
          f"{base_agg['mean_win_rate']:4.0%}  {base_agg['folds_positive_ret']:>5}")
    for k in TIGHT_KS:
        a = eval_k(k)
        print(f"     {k:5d}  {a['mean_return']:+7.1%}  {a['mean_sharpe']:+5.2f}  "
              f"{a['mean_win_rate']:4.0%}  {a['folds_positive_ret']:>5}")
    print("     (si WR/Sharpe suben al concentrar => meta-labeling/precisión prometedor)")

    print(f"\n  Runtime: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
