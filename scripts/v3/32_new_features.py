"""V4 experiment: new accumulation/distribution + momentum features via harness.

Screening (standalone per-date IC on the tradable cross-section) found strong,
genuinely-new candidates the model lacks: mom_12_1 (+0.045, the canonical 12-1
momentum), mom_mktadj_120 (+0.040), mom_mktadj_60 (+0.019), cmf_60 (+0.016,
Chaikin Money Flow / accumulation), eom_14 (+0.013). Test small motivated groups
(avoid the overfit that sank the wholesale B_micro batch) through the filtered
16-fold walk-forward + anti-leak gate, comparable to v4_filt_baseline.

Promotion criteria identical to 22_feature_batches.py.

Usage:
    PYTHONPATH=src python scripts/v3/32_new_features.py mom
    PYTHONPATH=src python scripts/v3/32_new_features.py mom mom_acc best4
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
    save_result,
)

_spec = importlib.util.spec_from_file_location(
    "verify_no_leak", ROOT / "scripts" / "v3" / "18_verify_no_leak.py")
_leak = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_leak)

DECISION_FP = ROOT / "data" / "v3_benchmarks" / "v4_filter_decision.json"
BASELINE_FP = ROOT / "data" / "v3_benchmarks" / "v4_filt_baseline.json"
ACCUM_FP = ROOT / "data" / "accum_features.parquet"

SETS = {
    "mom": ["mom_12_1"],
    "mom_acc": ["mom_12_1", "cmf_60"],
    "best4": ["mom_12_1", "mom_mktadj_120", "cmf_60", "eom_14"],
    "momfam": ["mom_12_1", "mom_mktadj_120", "mom_mktadj_60"],
}

PROD_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg",
    "num_leaves": 31, "max_depth": 6,
    "learning_rate": 0.05, "min_child_samples": 30,
    "subsample": 0.75, "colsample_bytree": 0.7,
    "reg_lambda": 5.0,
    "n_jobs": 1, "seed": 42, "verbose": -1,
}


def run_set(name: str, cols: list[str], features: pd.DataFrame, ohlcv: pd.DataFrame,
            decision: dict, base_agg: dict) -> None:
    feat_base = V2_FEATURES_BASE + V2_EDGAR_FEATURES
    train = features.dropna(subset=[V2_TARGET])
    sample = train.sample(min(200_000, len(train)), random_state=42)
    errors = (_leak.check_feature_names(cols)
              + _leak.check_degenerate(sample, cols)
              + _leak.check_pearson(sample, cols)
              + _leak.check_per_date_spearman(sample, cols))
    if errors:
        print(f"  [{name}] LEAK GATE FAILED: {errors}")
        return
    res = run_walkforward(
        features, ohlcv, feat_base + cols,
        config_name=f"v4_feat_{name}",
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


def main() -> None:
    t0 = time.time()
    decision = json.loads(DECISION_FP.read_text())
    base_agg = json.loads(BASELINE_FP.read_text())["aggregate"]
    requested = [s for s in sys.argv[1:] if s in SETS] or ["mom", "mom_acc", "best4"]

    feat_base = V2_FEATURES_BASE + V2_EDGAR_FEATURES
    need = ["date", "ticker", V2_TARGET, "atr_pct_20d", "close",
            "adv_usd_20d", "cs_spread_20d"] + feat_base
    need = list(dict.fromkeys(need))
    features = pd.read_parquet(ROOT / "data/processed/features_smallcap.parquet", columns=need)
    features["date"] = pd.to_datetime(features["date"])
    accum = pd.read_parquet(ACCUM_FP)
    accum["date"] = pd.to_datetime(accum["date"])
    features = features.merge(accum, on=["date", "ticker"], how="left")

    ohlcv = pd.read_parquet(ROOT / "data/processed/ohlcv_smallcap.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    print(f"\n  New-feature sets vs baseline (ICtr {base_agg['mean_ic_tradable']:+.4f}, "
          f"ret {base_agg['mean_return']:+.1%}, WR {base_agg['mean_win_rate']:.1%})\n")
    for name in requested:
        run_set(name, SETS[name], features, ohlcv, decision, base_agg)
    print(f"\n  Total runtime: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
