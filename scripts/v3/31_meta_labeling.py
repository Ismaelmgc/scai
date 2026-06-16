"""V4 experiment: meta-labeling to capture the concentration edge intelligently.

Concentration (30) showed the top ~6 beat the top-8 on return. Instead of a flat
cut to 6, train a SECONDARY model (López de Prado meta-labeling) to pick the best
6 of the primary's top candidates by P(win), aiming for the concentration gain
with less drawdown.

Leak-free design: the meta-model is trained on an EXPANDING window of PAST folds
only — each cached fold's TEST predictions are out-of-sample primary outputs, and
the realized fwd_ret_20d gives the win/lose label. For fold i the meta trains on
folds 1..i-1 and predicts fold i (fold 1 falls back to the primary ranking).

Meta features = the 28 production features + the primary score. Selection: among
the primary's tradable top-N (by score), take the top-6 by meta P(win). Compared
to primary top-8 (baseline) and flat primary top-6, under adaptive6_pt40.

Usage:
    PYTHONPATH=src python scripts/v3/31_meta_labeling.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "v3"))

from _v3_harness import (  # noqa: E402
    V2_EDGAR_FEATURES,
    V2_FEATURES_BASE,
    V2_RAW_COL,
    ExitPolicy,
    FoldMetrics,
    RunResult,
    _evaluate_fold,
)

from app.features.tradability import tradable_mask  # noqa: E402

CACHE_DIR = ROOT / "data" / "v3_benchmarks" / "cache" / "v4_nometa"
DECISION_FP = ROOT / "data" / "v3_benchmarks" / "v4_filter_decision.json"
PRIMARY_N = 12   # primary prefilter: meta only re-ranks the primary's top-N
FINAL_K = 6      # final book size
META_FEATS = V2_FEATURES_BASE + V2_EDGAR_FEATURES + ["pred"]
META_PARAMS = {
    "objective": "binary", "metric": "auc",
    "num_leaves": 15, "max_depth": 4, "learning_rate": 0.05,
    "min_child_samples": 100, "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_lambda": 5.0, "n_jobs": 1, "seed": 42, "verbose": -1,
}


def _policy(d: dict) -> ExitPolicy:
    fields = set(ExitPolicy.__dataclass_fields__)
    return ExitPolicy(**{k: v for k, v in d.items() if k in fields})


def _aggregate_with_dd(folds_list) -> dict:
    rr = RunResult(config_name="x", feat_cols=[], n_features=0)
    rr.folds = folds_list
    a = rr.aggregate()
    a["worst_dd"] = float(np.min([f.max_dd for f in folds_list]))
    return a


def main() -> None:
    t0 = time.time()
    decision = json.loads(DECISION_FP.read_text())
    mp, ma, cb = decision["min_price"], decision["min_adv_usd"], decision["cost_bps"]
    policy = _policy(decision["exit_policy_adaptive"])  # best from validation

    ohlcv = pd.read_parquet(ROOT / "data/processed/ohlcv_smallcap.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    # Production features + raw 20d return (win label), joined to each cache fold.
    fcols = ["date", "ticker", V2_RAW_COL] + V2_FEATURES_BASE + V2_EDGAR_FEATURES
    fcols = list(dict.fromkeys(fcols))
    feats = pd.read_parquet(ROOT / "data/processed/features_smallcap.parquet", columns=fcols)
    feats["date"] = pd.to_datetime(feats["date"])

    meta = json.loads((CACHE_DIR / "meta.json").read_text())
    folds = []
    for fm in meta:
        td = pd.read_parquet(CACHE_DIR / f"fold_{fm['fold']:02d}.parquet")
        td["date"] = pd.to_datetime(td["date"])
        td = td.merge(feats, on=["date", "ticker"], how="left", suffixes=("", "_f"))
        folds.append((fm, td))

    def fold_metrics(fm, td, sel_col, k):
        td2 = td.copy()
        td2["pred"] = td2[sel_col]
        ev = _evaluate_fold(
            td2, ohlcv,
            (pd.Timestamp(fm["test_start"]), pd.Timestamp(fm["test_end"])),
            mp, ma, policy, cb, False, top_k=k,
        )
        return FoldMetrics(
            fold=fm["fold"], period=fm["period"], train_rows=fm["train_rows"],
            test_rows=fm["test_rows"], mean_ic=fm["mean_ic"], ic_ir=fm["ic_ir"],
            hit_rate_ic=fm["hit_rate_ic"], total_return=ev["total_return"],
            sharpe=ev["sharpe"], max_dd=ev["max_dd"], n_trades=ev["n_trades"],
            win_rate=ev["win_rate"], median_return=ev["median_return"],
            market_return=ev["market_return"], mean_ic_tradable=ev["mean_ic_tradable"],
            n_skipped_rebalances=ev["n_skipped_rebalances"], avg_candidates=ev["avg_candidates"],
        )

    base8, flat6, meta6 = [], [], []
    past: list[pd.DataFrame] = []
    for fm, td in folds:
        # ── meta P(win) via expanding window of PAST folds only (no leak) ──
        if past:
            tr = pd.concat(past, ignore_index=True).dropna(subset=[V2_RAW_COL])
            Xtr = tr[META_FEATS].fillna(0).values
            ytr = (tr[V2_RAW_COL] > 0).astype(int).values
            ds = lgb.Dataset(Xtr, ytr, feature_name=META_FEATS, free_raw_data=True)
            clf = lgb.train(META_PARAMS, ds, num_boost_round=200)
            td["meta_p"] = clf.predict(td[META_FEATS].fillna(0).values)
        else:
            td["meta_p"] = td["pred"]  # fold 1: no meta history → primary order

        # selection score: among tradable primary top-N, rank by meta_p; else -inf
        trad = tradable_mask(td, mp, ma)
        rk = td["pred"].where(trad).groupby(td["date"]).rank(ascending=False, method="first")
        td["sel_meta"] = td["meta_p"].where(trad & (rk <= PRIMARY_N), other=-1e9)

        base8.append(fold_metrics(fm, td, "pred", 8))
        flat6.append(fold_metrics(fm, td, "pred", FINAL_K))
        meta6.append(fold_metrics(fm, td, "sel_meta", FINAL_K))
        past.append(td[META_FEATS + [V2_RAW_COL]].copy())

    rows = [("primary top-8 (prod)", _aggregate_with_dd(base8)),
            ("primary top-6 (flat)", _aggregate_with_dd(flat6)),
            ("meta top-6 (of top-12)", _aggregate_with_dd(meta6))]
    print(f"\n  Meta-labeling vs flat concentration — adaptive6_pt40, 16 folds, {cb:g}bps\n")
    print("  variant                   netRet   Sharpe    WR    +folds   worstDD")
    print("  " + "-" * 66)
    for name, a in rows:
        print(f"  {name:24s}  {a['mean_return']:+7.1%}  {a['mean_sharpe']:+5.2f}  "
              f"{a['mean_win_rate']:4.0%}  {a['folds_positive_ret']:>5}   {a['worst_dd']:+6.1%}")
    print("\n  (meta gana si bate al corte plano top-6 en WR/Sharpe o reduce el DD.)")
    print(f"  Runtime: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
