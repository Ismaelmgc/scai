"""V4 Phase 4: model variants under the filtered harness.

Variants (each compared against v4_filt_baseline with the same filter,
cost and default exit policy):
  bins8 / bins32 — LambdaRank relevance binning alternatives (16 is prod).
  blend          — rank-average of LambdaRank score and a binary LGB
                   classifier P(top-quartile sector-rel return). The
                   classifier sees the same features; only the objective
                   differs. Blending decorrelates objective-specific noise.

Same promotion criteria as Phase 3 (see 22_feature_batches.py).

Usage:
    PYTHONPATH=src python scripts/v3/23_model_variants.py            # all
    PYTHONPATH=src python scripts/v3/23_model_variants.py blend     # one
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "v3"))

from scipy.stats import spearmanr  # noqa: E402

from _v3_harness import (  # noqa: E402
    V2_FEATURES_BASE, V2_EDGAR_FEATURES, V2_TARGET,
    FoldMetrics, RunResult, ExitPolicy,
    define_folds, run_walkforward, save_result, _evaluate_fold,
)

DECISION_FP = ROOT / "data" / "v3_benchmarks" / "v4_filter_decision.json"
BASELINE_FP = ROOT / "data" / "v3_benchmarks" / "v4_filt_baseline.json"

PROD_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg",
    "num_leaves": 31, "max_depth": 6,
    "learning_rate": 0.05, "min_child_samples": 30,
    "subsample": 0.75, "colsample_bytree": 0.7,
    "reg_lambda": 5.0,
    "n_jobs": 1, "seed": 42, "verbose": -1,
}

CLF_PARAMS = {
    "objective": "binary", "metric": "auc",
    "num_leaves": 31, "max_depth": 6,
    "learning_rate": 0.05, "min_child_samples": 30,
    "subsample": 0.75, "colsample_bytree": 0.7,
    "reg_lambda": 5.0,
    "n_jobs": 1, "seed": 42, "verbose": -1,
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    feat_cols = V2_FEATURES_BASE + V2_EDGAR_FEATURES
    need = ["date", "ticker", V2_TARGET, "atr_pct_20d", "close",
            "adv_usd_20d", "cs_spread_20d"] + feat_cols
    features = pd.read_parquet(ROOT / "data/processed/features_smallcap.parquet",
                               columns=need)
    features["date"] = pd.to_datetime(features["date"])
    ohlcv = pd.read_parquet(ROOT / "data/processed/ohlcv_smallcap.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    return features, ohlcv, feat_cols


def run_blend(features: pd.DataFrame, ohlcv: pd.DataFrame, feat_cols: list[str],
              decision: dict) -> RunResult:
    """LambdaRank + binary classifier P(top-quartile), rank-averaged.

    Custom fold loop (two models per fold) but identical folds, filter,
    cost and evaluation as the harness — comparable to v4_filt_baseline.
    """
    folds = define_folds(features)
    lr_params = dict(PROD_PARAMS)
    lr_params["lambdarank_truncation_level"] = 8
    lr_params["label_gain"] = list(range(16))
    result = RunResult(config_name="v4_model_blend", feat_cols=feat_cols,
                       n_features=len(feat_cols))

    for i, fold in enumerate(folds):
        train_mask = (features.date >= fold["train_start"]) & (features.date < fold["train_end"])
        train_data = features[train_mask].dropna(subset=[V2_TARGET]).copy()

        # LambdaRank head (same as production)
        train_data["_rel"] = train_data.groupby("date")[V2_TARGET].transform(
            lambda s: pd.qcut(s.rank(method="first"), 16, labels=False, duplicates="drop")
        ).fillna(0).astype(int).clip(0, 15)
        train_data = train_data.sort_values("date")
        X_tr = train_data[feat_cols].fillna(0).values
        group = train_data.groupby("date").size().values
        ds_lr = lgb.Dataset(X_tr, train_data["_rel"].values, group=group,
                            feature_name=feat_cols, free_raw_data=True)
        m_lr = lgb.train(lr_params, ds_lr, num_boost_round=400,
                         callbacks=[lgb.log_evaluation(0)])

        # Binary head: P(top-quartile sector-rel return same date)
        train_data["_top_q"] = (train_data.groupby("date")[V2_TARGET]
                                .transform(lambda s: s.rank(pct=True)) >= 0.75).astype(int)
        ds_clf = lgb.Dataset(X_tr, train_data["_top_q"].values,
                             feature_name=feat_cols, free_raw_data=True)
        m_clf = lgb.train(CLF_PARAMS, ds_clf, num_boost_round=400,
                          callbacks=[lgb.log_evaluation(0)])

        test_mask = (features.date >= fold["test_start"]) & (features.date < fold["test_end"])
        test_data = features[test_mask].dropna(subset=[V2_TARGET]).copy()
        if test_data.empty:
            continue
        X_te = test_data[feat_cols].fillna(0).values
        s_lr = pd.Series(m_lr.predict(X_te), index=test_data.index)
        s_clf = pd.Series(m_clf.predict(X_te), index=test_data.index)
        # Rank-average per date (scale-free blend)
        test_data["_r1"] = s_lr
        test_data["_r2"] = s_clf
        test_data["pred"] = (test_data.groupby("date")["_r1"].rank(pct=True)
                             + test_data.groupby("date")["_r2"].rank(pct=True)) / 2

        daily_ics = []
        for d in sorted(test_data.date.unique()):
            day = test_data[test_data.date == d]
            if len(day) < 10:
                continue
            ic, _ = spearmanr(day["pred"], day[V2_TARGET])
            daily_ics.append(ic)
        mean_ic = float(np.mean(daily_ics)) if daily_ics else 0.0
        ic_std = float(np.std(daily_ics)) if daily_ics else 0.0

        ev = _evaluate_fold(
            test_data, ohlcv, (fold["test_start"], fold["test_end"]),
            decision["min_price"], decision["min_adv_usd"],
            ExitPolicy(), decision["cost_bps"], False,
        )
        fm = FoldMetrics(
            fold=i + 1,
            period=f"{fold['test_start'].strftime('%Y-%m')}->{fold['test_end'].strftime('%Y-%m')}",
            train_rows=len(train_data), test_rows=len(test_data),
            mean_ic=mean_ic, ic_ir=mean_ic / ic_std if ic_std > 0 else 0.0,
            hit_rate_ic=float(np.mean([ic > 0 for ic in daily_ics])) if daily_ics else 0.0,
            total_return=ev["total_return"], sharpe=ev["sharpe"], max_dd=ev["max_dd"],
            n_trades=ev["n_trades"], win_rate=ev["win_rate"],
            median_return=ev["median_return"], market_return=ev["market_return"],
            mean_ic_tradable=ev["mean_ic_tradable"],
            n_skipped_rebalances=ev["n_skipped_rebalances"],
            avg_candidates=ev["avg_candidates"],
        )
        result.folds.append(fm)
        print(f"  Fold {i+1:2d} {fm.period}: ICtr={fm.mean_ic_tradable:+.4f} "
              f"Ret={fm.total_return:+7.1%} Sharpe={fm.sharpe:+5.2f} WR={fm.win_rate:.0%}")
    return result


def verdict(agg: dict, base_agg: dict, name: str) -> None:
    base_pos = int(base_agg["folds_positive_ret"].split("/")[0])
    cand_pos = int(agg["folds_positive_ret"].split("/")[0])
    c1 = agg["mean_ic_tradable"] >= base_agg["mean_ic_tradable"] + 0.002
    c2 = cand_pos >= base_pos
    c3 = (agg["mean_return"] >= 0.95 * base_agg["mean_return"]
          and agg["mean_win_rate"] >= base_agg["mean_win_rate"] - 0.01)
    strict = sum([
        agg["mean_ic_tradable"] > base_agg["mean_ic_tradable"],
        agg["mean_return"] > base_agg["mean_return"],
        agg["mean_win_rate"] > base_agg["mean_win_rate"],
    ])
    c5 = strict >= 2
    v = "PROMOTE" if (c1 and c2 and c3 and c5) else "REJECT"
    print(f"  [{name}] {v}: ICtr {base_agg['mean_ic_tradable']:+.4f}->{agg['mean_ic_tradable']:+.4f} "
          f"ret {base_agg['mean_return']:+.1%}->{agg['mean_return']:+.1%} "
          f"WR {base_agg['mean_win_rate']:.1%}->{agg['mean_win_rate']:.1%} "
          f"+folds {base_agg['folds_positive_ret']}->{agg['folds_positive_ret']} "
          f"[c1={c1} c2={c2} c3={c3} c5={c5}]")


def main() -> None:
    t0 = time.time()
    decision = json.loads(DECISION_FP.read_text())
    base_agg = json.loads(BASELINE_FP.read_text())["aggregate"]
    requested = sys.argv[1:] or ["bins8", "bins32", "blend"]

    features, ohlcv, feat_cols = load_data()

    for name in requested:
        print(f"\n=== Variant {name} ===")
        if name in ("bins8", "bins32"):
            nb = int(name.replace("bins", ""))
            params = dict(PROD_PARAMS)
            params["label_gain"] = list(range(nb))
            res = run_walkforward(
                features, ohlcv, feat_cols,
                config_name=f"v4_model_{name}",
                lgb_params=params, objective_lambdarank=True,
                min_price=decision["min_price"], min_adv_usd=decision["min_adv_usd"],
                cost_bps=decision["cost_bps"], n_bins=nb, verbose=True,
            )
        elif name == "blend":
            res = run_blend(features, ohlcv, feat_cols, decision)
        else:
            print(f"Unknown variant {name!r}")
            continue
        save_result(res)
        verdict(res.aggregate(), base_agg, name)

    print(f"\nTotal runtime: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
