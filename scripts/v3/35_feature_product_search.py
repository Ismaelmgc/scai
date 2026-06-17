"""V4 experiment: combinatorial feature-subset x product (TOP_K) search.

Rounds 1-2 only ever ADDED features (28 + X) and rejected. The unexplored space
is COMBINATIONS: dropping/swapping features and jointly sweeping the book size.
Maybe a SUBSET of the 28 (+ a new one) at a different TOP_K beats baseline on
return/WR even though each addition alone did not.

This is the #1 overfitting trap, so the search is disciplined:
  * candidates are IMPORTANCE/CORRELATION-guided (~16 principled subsets), not blind.
  * each subset is retrained (16-fold, cached) at a reduced tree count (screen);
    the TOP_K 4..10 sweep is then FREE via cache replay under prod exit policies.
  * SELECT on folds 1-11, CONFIRM on the held-out recent block folds 12-16.
  * Monte Carlo (34_mc_validate): paired bootstrap vs baseline + Sidak haircut for
    the number of trials. A subset PROMOTES only if CONFIRM beats baseline AND the
    bootstrap CI of the difference excludes 0 after the haircut.

Usage:
    PYTHONPATH=src python scripts/v3/35_feature_product_search.py          # full
    PYTHONPATH=src python scripts/v3/35_feature_product_search.py smoke    # 3 sets, few trees
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
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


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "v3" / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_leak = _load("18_verify_no_leak.py", "verify_no_leak")
mc = _load("34_mc_validate.py", "mc_validate")

DECISION_FP = ROOT / "data" / "v3_benchmarks" / "v4_filter_decision.json"
ACCUM_FP = ROOT / "data" / "accum_features.parquet"
RESIDMOM_FP = ROOT / "data" / "residmom_features.parquet"
CACHE_ROOT = ROOT / "data" / "v3_benchmarks" / "cache"

BASE = V2_FEATURES_BASE + V2_EDGAR_FEATURES        # the 28 production features
NEW = ["residmom_12_1", "residmom_6_1", "pct_from_52w_high", "idio_vol_60d"]
POOL = BASE + NEW
KS = list(range(4, 11))                            # product sweep 4..10
SELECT_MAX_FOLD = 11                               # folds 1..11 select, 12.. confirm

PROD_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg",
    "num_leaves": 31, "max_depth": 6, "learning_rate": 0.05,
    "min_child_samples": 30, "subsample": 0.75, "colsample_bytree": 0.7,
    "reg_lambda": 5.0, "n_jobs": 1, "seed": 42, "verbose": -1,
}


def importance_and_corr(features: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Train one model on the full pool for gain importances + correlation matrix."""
    tr = features.dropna(subset=[V2_TARGET])
    tr = tr.sample(min(250_000, len(tr)), random_state=42).sort_values("date")
    tr["_rel"] = tr.groupby("date")[V2_TARGET].transform(
        lambda s: pd.qcut(s.rank(method="first"), 16, labels=False, duplicates="drop"))
    tr["_rel"] = tr["_rel"].fillna(0).astype(int).clip(0, 15)
    params = dict(PROD_PARAMS, lambdarank_truncation_level=8, label_gain=list(range(16)))
    ds = lgb.Dataset(tr[POOL].fillna(0).values, tr["_rel"].values,
                     group=tr.groupby("date").size().values, feature_name=POOL)
    model = lgb.train(params, ds, num_boost_round=300)
    imp = pd.Series(model.feature_importance("gain"), index=POOL).sort_values(ascending=False)
    corr = tr[POOL].sample(min(60_000, len(tr)), random_state=1).corr().abs()
    return imp, corr


def corr_prune(cols: list[str], corr: pd.DataFrame, imp: pd.Series, thr: float = 0.9) -> list[str]:
    """Keep features by descending importance, dropping any |corr|>thr with a kept one."""
    kept: list[str] = []
    for f in [c for c in imp.index if c in cols]:
        if all(corr.loc[f, k] <= thr for k in kept):
            kept.append(f)
    return kept


def build_subsets(imp: pd.Series, corr: pd.DataFrame) -> dict[str, list[str]]:
    base_by_imp = [f for f in imp.index if f in BASE]      # 28, importance desc
    pruned = corr_prune(POOL, corr, imp, thr=0.9)
    sets = {
        "base28": BASE,                                    # control
        "drop2": base_by_imp[:-2],
        "drop4": base_by_imp[:-4],
        "drop6": base_by_imp[:-6],
        "drop8": base_by_imp[:-8],
        "swap_ret252_residmom": [f for f in BASE if f != "ret_252d"] + ["residmom_12_1"],
        "base+residmom": BASE + ["residmom_12_1"],
        "base+52w": BASE + ["pct_from_52w_high"],
        "corr_prune": pruned,
        "corr_prune+52w": list(dict.fromkeys(pruned + ["pct_from_52w_high"])),
        "corr_prune+lowvol": list(dict.fromkeys(pruned + ["idio_vol_60d"])),
        "corr_prune+residmom": list(dict.fromkeys(pruned + ["residmom_12_1"])),
        "drop6+residmom+52w": base_by_imp[:-6] + ["residmom_12_1", "pct_from_52w_high"],
        "drop4+lowvol": base_by_imp[:-4] + ["idio_vol_60d"],
    }
    return sets


def block(df: pd.DataFrame, select: bool) -> pd.DataFrame:
    return df[df.fold <= SELECT_MAX_FOLD] if select else df[df.fold > SELECT_MAX_FOLD]


def main() -> None:
    t0 = time.time()
    smoke = "smoke" in sys.argv
    trees = 120 if smoke else 250
    decision = json.loads(DECISION_FP.read_text())
    mp, ma, cb = decision["min_price"], decision["min_adv_usd"], decision["cost_bps"]
    pol = mc.policy_from(decision["exit_policy_adaptive"])

    need = (["date", "ticker", V2_TARGET, "atr_pct_20d", "close", "adv_usd_20d",
             "cs_spread_20d"] + BASE + ["pct_from_52w_high", "idio_vol_60d"])
    need = list(dict.fromkeys(need))
    features = pd.read_parquet(ROOT / "data/processed/features_smallcap.parquet", columns=need)
    features["date"] = pd.to_datetime(features["date"])
    rm = pd.read_parquet(RESIDMOM_FP)[["date", "ticker", "residmom_12_1", "residmom_6_1"]]
    rm["date"] = pd.to_datetime(rm["date"])
    features = features.merge(rm, on=["date", "ticker"], how="left")
    ohlcv = pd.read_parquet(ROOT / "data/processed/ohlcv_smallcap.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    # leak gate on the new candidates (base already vetted)
    samp = features.dropna(subset=[V2_TARGET]).sample(min(200_000, len(features)), random_state=42)
    errs = (_leak.check_pearson(samp, NEW) + _leak.check_per_date_spearman(samp, NEW)
            + _leak.check_degenerate(samp, NEW))
    print("\n  leak gate on new features:", errs if errs else "CLEAN")

    imp, corr = importance_and_corr(features)
    print("\n  pool importance (gain, top): "
          + ", ".join(f"{k}={v:.0f}" for k, v in imp.head(8).items()))
    sets = build_subsets(imp, corr)
    if smoke:
        sets = {k: sets[k] for k in ["base28", "drop4", "corr_prune+52w"]}

    # ── train + cache each subset, then free TOP_K sweep via replay ──
    perfold: dict[tuple[str, int], pd.DataFrame] = {}
    for name, cols in sets.items():
        cache_dir = CACHE_ROOT / f"v4_fs_{name}"
        run_walkforward(
            features, ohlcv, cols, config_name=f"v4_fs_{name}", cache_dir=cache_dir,
            lgb_params=PROD_PARAMS, objective_lambdarank=True, num_boost_round=trees,
            min_price=mp, min_adv_usd=ma, cost_bps=cb, verbose=False,
        )
        for k in KS:
            perfold[(name, k)] = mc.replay_per_fold(cache_dir, ohlcv, mp, ma, cb, pol, k)
        print(f"  trained+swept: {name} ({len(cols)} feats)")

    # ── pick best K per subset on the SELECT block; report CONFIRM + MC ──
    base_best = perfold[("base28", 8)]
    base_sel, base_conf = block(base_best, True), block(base_best, False)
    n_trials = len(sets) * len(KS)
    rows = []
    for name in sets:
        best_k, best_sel = None, -9
        for k in KS:
            s = block(perfold[(name, k)], True)["total_return"].mean()
            if s > best_sel:
                best_sel, best_k = s, k
        df = perfold[(name, best_k)]
        sel, conf = block(df, True), block(df, False)
        pb_all = mc.paired_bootstrap(df["total_return"].values, base_best["total_return"].values)
        wr_conf = (conf["win_rate"] * conf["n_trades"]).sum() / max(conf["n_trades"].sum(), 1)
        rows.append({
            "subset": name, "K": best_k, "nfeat": len(sets[name]),
            "sel_ret": sel["total_return"].mean(), "conf_ret": conf["total_return"].mean(),
            "conf_wr": wr_conf, "conf_dd": conf["max_dd"].min(),
            "diff_all": pb_all["mean_diff"], "ci_lo": pb_all["ci_lo"], "ci_hi": pb_all["ci_hi"],
            "p_le0": pb_all["p_le0"], "p_adj": mc.sidak(pb_all["p_le0"], n_trials),
        })

    base_conf_wr = ((base_conf["win_rate"] * base_conf["n_trades"]).sum()
                    / base_conf["n_trades"].sum())
    print(f"\n  Baseline (28f, top-8): SELECT ret {base_sel['total_return'].mean():+.1%}  "
          f"CONFIRM ret {base_conf['total_return'].mean():+.1%}  CONFIRM WR {base_conf_wr:.0%}")
    print(f"  Trials M = {n_trials} (Sidak haircut applied to p)\n")
    print(f"  {'subset':24s} {'K':>2} {'sel_ret':>7} {'conf_ret':>8} {'conf_WR':>7} "
          f"{'conf_DD':>7} {'diff_all':>8} {'95% CI':>16} {'p_adj':>6} verdict")
    print("  " + "-" * 104)
    res = pd.DataFrame(rows).sort_values("conf_ret", ascending=False)
    for _, r in res.iterrows():
        beats = (r["conf_ret"] > base_conf["total_return"].mean()
                 and r["conf_wr"] >= base_conf_wr and r["ci_lo"] > 0 and r["p_adj"] < 0.05)
        verdict = "PROMOTE" if beats else ""
        print(f"  {r['subset']:24s} {int(r['K']):>2} {r['sel_ret']:+7.1%} {r['conf_ret']:+8.1%} "
              f"{r['conf_wr']:6.0%} {r['conf_dd']:+7.1%} {r['diff_all']:+8.1%} "
              f"[{r['ci_lo']:+.1%},{r['ci_hi']:+.1%}] {r['p_adj']:6.2f} {verdict}")
    print(f"\n  Total runtime: {(time.time() - t0)/60:.1f} min  (trees={trees})")


if __name__ == "__main__":
    main()
