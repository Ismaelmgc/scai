"""V4 experiment: UPWARD breadth test — does a bigger universe improve the top-8?

The downward ablation (38) showed candidate-pool size is a first-order driver of
return AND WR. This is the upward confirmation: rebuild the feature matrix on the
EXPANDED universe (~2969 names: 1048 existing + 1921 newly downloaded tradeable
$50M-$2B small-caps; see bootstrap_breadth_universe.py) and run the same 16-fold
walk-forward. Treatment (expanded) vs the matched 250-tree, original-universe
control (cache v4_fs_base28), MC paired-bootstrap on the top-8.

Production untouched: reads research_breadth/* + the local processed parquets,
writes only research_breadth/features_expanded.parquet and a research cache.

Usage:
    PYTHONPATH=src python scripts/v3/39_breadth_upward.py
    PYTHONPATH=src python scripts/v3/39_breadth_upward.py rebuild   # force feature rebuild
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

from app.data.store.parquet_store import ParquetStore  # noqa: E402
from app.features.pipeline import build_feature_matrix  # noqa: E402

_spec = importlib.util.spec_from_file_location("mc", ROOT / "scripts/v3/34_mc_validate.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

RB = ROOT / "data" / "research_breadth"
FEAT_FP = RB / "features_expanded.parquet"
DECISION_FP = ROOT / "data" / "v3_benchmarks" / "v4_filter_decision.json"
CACHE_ROOT = ROOT / "data" / "v3_benchmarks" / "cache"
CTRL_CACHE = CACHE_ROOT / "v4_fs_base28"          # matched 250-tree original-universe control
BASE = V2_FEATURES_BASE + V2_EDGAR_FEATURES
PROD_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg",
    "num_leaves": 31, "max_depth": 6, "learning_rate": 0.05,
    "min_child_samples": 30, "subsample": 0.75, "colsample_bytree": 0.7,
    "reg_lambda": 5.0, "n_jobs": 1, "seed": 42, "verbose": -1,
}


def _merge_edgar(features: pd.DataFrame) -> pd.DataFrame:
    edgar_path = ROOT / "data" / "edgar_facts.parquet"
    if not edgar_path.exists():
        return features
    from app.data.free_sources.sec_edgar import compute_edgar_features
    ef = compute_edgar_features(pd.read_parquet(edgar_path))
    if ef.empty:
        return features
    keep = ["ticker", "filing_date"] + [c for c in V2_EDGAR_FEATURES if c in ef.columns]
    ef = ef[keep].dropna(subset=["filing_date"]).copy()
    ef["filing_date"] = pd.to_datetime(ef["filing_date"])
    features["date"] = pd.to_datetime(features["date"])
    merged = pd.merge_asof(
        features.sort_values("date").reset_index(drop=True),
        ef.sort_values("filing_date").reset_index(drop=True),
        left_on="date", right_on="filing_date", by="ticker", direction="backward",
    )
    return merged.drop(columns=["filing_date"], errors="ignore")


def build_expanded(original_only: bool = False) -> pd.DataFrame:
    store = ParquetStore()
    oh = pd.read_parquet(ROOT / "data/processed/ohlcv_smallcap.parquet")
    new = pd.read_parquet(RB / "ohlcv_new.parquet")
    for d in (oh, new):
        d["date"] = pd.to_datetime(d["date"])
    cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
    cols += [c for c in ("vwap", "transactions") if c in oh.columns and c in new.columns]
    parts = [oh[cols]] if original_only else [oh[cols], new[cols]]
    combined = (pd.concat(parts, ignore_index=True)
                .drop_duplicates(["date", "ticker"]).sort_values(["ticker", "date"]))
    uni = pd.read_parquet(RB / "universe_expanded.parquet").to_dict("records")
    spy = store.read("smallcap_spy")
    print(f"  building features on {combined['ticker'].nunique()} tickers, "
          f"{len(combined):,} bars...")
    feats = build_feature_matrix(combined, fundamentals=None, market_df=spy,
                                 universe=uni, horizons=[1, 5, 10, 20])
    # Trim the 481-col matrix to what the harness needs BEFORE the EDGAR merge
    # (the full matrix x 2.8M rows is ~10GB; merge_asof's copy would OOM).
    aux = ["atr_pct_20d", "close", "adv_usd_20d", "cs_spread_20d"]
    keep = (["date", "ticker", V2_TARGET] + aux
            + [c for c in V2_FEATURES_BASE if c in feats.columns])
    feats = feats[list(dict.fromkeys(keep))].copy()
    feats = _merge_edgar(feats)
    feats.to_parquet(FEAT_FP, index=False)
    print(f"  saved expanded features -> {FEAT_FP} ({len(feats):,} rows)")
    return feats


def per_fold(cache_dir, ohlcv, mp, ma, cb, pol, k):
    return mc.replay_per_fold(cache_dir, ohlcv, mp, ma, cb, pol, k)


def wr(df):
    return (df["win_rate"] * df["n_trades"]).sum() / df["n_trades"].sum()


def run_control_sameprovenance(mp, ma, cb, pol) -> None:
    """Same-provenance control: MY feature build on the ORIGINAL universe only.

    Rules out a feature-build confound — if this reproduces ~+8.4% (the production
    control v4_fs_base28), then my build is sound and the expansion genuinely hurts.
    """
    feats = build_expanded(original_only=True)
    oh0 = pd.read_parquet(ROOT / "data/processed/ohlcv_smallcap.parquet")
    oh0["date"] = pd.to_datetime(oh0["date"])
    cache_dir = CACHE_ROOT / "v4_orig_mybuild"
    run_walkforward(
        feats, oh0, BASE, config_name="v4_orig_mybuild", cache_dir=cache_dir,
        lgb_params=PROD_PARAMS, objective_lambdarank=True, num_boost_round=250,
        min_price=mp, min_adv_usd=ma, cost_bps=cb, verbose=False,
    )
    df = mc.replay_per_fold(cache_dir, oh0, mp, ma, cb, pol, 8)
    prod = mc.replay_per_fold(CTRL_CACHE, oh0, mp, ma, cb, pol, 8)
    print("\n  SAME-PROVENANCE CONTROL — my build, original 1048 universe\n")
    print(f"  prod control (v4_fs_base28): ret {prod['total_return'].mean():+.1%}  "
          f"WR {wr(prod):.0%}  Sharpe {prod['sharpe'].mean():+.2f}")
    print(f"  my build, original universe : ret {df['total_return'].mean():+.1%}  "
          f"WR {wr(df):.0%}  Sharpe {df['sharpe'].mean():+.2f}")
    print("  (match => my build is sound => the expansion genuinely hurts)")


def main() -> None:
    t0 = time.time()
    decision = json.loads(DECISION_FP.read_text())
    mp, ma, cb = decision["min_price"], decision["min_adv_usd"], decision["cost_bps"]
    pol = mc.policy_from(decision["exit_policy_adaptive"])

    if "control" in sys.argv:
        run_control_sameprovenance(mp, ma, cb, pol)
        return

    if FEAT_FP.exists() and "rebuild" not in sys.argv:
        feats = pd.read_parquet(FEAT_FP)
        feats["date"] = pd.to_datetime(feats["date"])
        print(f"  reuse expanded features ({len(feats):,} rows)")
    else:
        feats = build_expanded()

    ohlcv = pd.read_parquet(RB / "ohlcv_new.parquet")[
        ["date", "ticker", "open", "high", "low", "close", "volume"]]
    oh0 = pd.read_parquet(ROOT / "data/processed/ohlcv_smallcap.parquet")
    ohlcv = pd.concat([oh0, ohlcv], ignore_index=True).drop_duplicates(["date", "ticker"])
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    # Treatment: train on the expanded universe (matched 250 trees).
    cache_dir = CACHE_ROOT / "v4_breadth"
    run_walkforward(
        feats, ohlcv, BASE, config_name="v4_breadth", cache_dir=cache_dir,
        lgb_params=PROD_PARAMS, objective_lambdarank=True, num_boost_round=250,
        min_price=mp, min_adv_usd=ma, cost_bps=cb, verbose=False,
    )

    ctrl = per_fold(CTRL_CACHE, ohlcv, mp, ma, cb, pol, 8)
    treat = per_fold(cache_dir, ohlcv, mp, ma, cb, pol, 8)
    # candidate pool sizes (confirm breadth actually grew)
    import numpy as np
    from _v3_harness import _evaluate_fold
    meta = json.loads((cache_dir / "meta.json").read_text())
    cand_t = np.mean([_evaluate_fold(
        pd.read_parquet(cache_dir / f"fold_{m['fold']:02d}.parquet").assign(
            date=lambda d: pd.to_datetime(d.date)),
        ohlcv, (pd.Timestamp(m["test_start"]), pd.Timestamp(m["test_end"])),
        mp, ma, pol, cb, False, top_k=8)["avg_candidates"] for m in meta])

    pb8 = mc.paired_bootstrap(treat["total_return"].values, ctrl["total_return"].values)
    print(f"\n  UPWARD breadth test — top-8, adaptive6_pt40, {cb:g}bps, matched 250 trees\n")
    print(f"  {'universe':22s} {'cands':>6} {'ret':>7} {'WR':>4} {'Sharpe':>6} {'+folds':>6}")
    print(f"  {'original (1048)':22s} {'~526':>6} {ctrl['total_return'].mean():+7.1%} "
          f"{wr(ctrl):4.0%} {ctrl['sharpe'].mean():+6.2f} "
          f"{int((ctrl['total_return']>0).sum()):>4}/16")
    print(f"  {'expanded (2969)':22s} {cand_t:>6.0f} {treat['total_return'].mean():+7.1%} "
          f"{wr(treat):4.0%} {treat['sharpe'].mean():+6.2f} "
          f"{int((treat['total_return']>0).sum()):>4}/16")
    print(f"\n  diff {pb8['mean_diff']:+.1%} [95% {pb8['ci_lo']:+.1%}, {pb8['ci_hi']:+.1%}]  "
          f"p(diff<=0)={pb8['p_le0']:.3f}")
    verdict = "REAL improvement (CI>0)" if pb8["ci_lo"] > 0 else "NOT distinguishable (CI spans 0)"
    print(f"  -> breadth expansion is {verdict}")
    print(f"\n  Runtime: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
