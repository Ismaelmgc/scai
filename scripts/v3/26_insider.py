"""V4 experiment: insider-buying (Form 4) features through harness + leak gate.

Second orthogonal-data candidate from the quant-review roadmap (short interest
was rejected). Open-market insider purchases are a documented small-cap signal
with POSITIVE sign and no entanglement with the squeeze effect. Adds two PIT-safe
trailing-90d features (insider_buy_count_90d, insider_buy_value_log90d; see
app.features.insider) on top of the production 28 and runs the same filtered
16-fold walk-forward, comparable to v4_filt_baseline.

Promotion criteria vs v4_filt_baseline (identical to 22_feature_batches.py).

Usage:
    PYTHONPATH=src python scripts/v3/26_insider.py
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

from app.features.insider import add_insider_features  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "verify_no_leak", ROOT / "scripts" / "v3" / "18_verify_no_leak.py")
_leak = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_leak)

DECISION_FP = ROOT / "data" / "v3_benchmarks" / "v4_filter_decision.json"
BASELINE_FP = ROOT / "data" / "v3_benchmarks" / "v4_filt_baseline.json"
INSIDER_FP = ROOT / "data" / "insider_form4.parquet"

PROD_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg",
    "num_leaves": 31, "max_depth": 6,
    "learning_rate": 0.05, "min_child_samples": 30,
    "subsample": 0.75, "colsample_bytree": 0.7,
    "reg_lambda": 5.0,
    "n_jobs": 1, "seed": 42, "verbose": -1,
}


def main() -> None:
    t0 = time.time()
    if not INSIDER_FP.exists():
        print(f"Missing {INSIDER_FP} — run scripts/download_insider.py first.")
        sys.exit(1)

    decision = json.loads(DECISION_FP.read_text())
    base_agg = json.loads(BASELINE_FP.read_text())["aggregate"]

    feat_base = V2_FEATURES_BASE + V2_EDGAR_FEATURES
    need = ["date", "ticker", V2_TARGET, "atr_pct_20d", "close",
            "adv_usd_20d", "cs_spread_20d"] + feat_base
    need = list(dict.fromkeys(need))
    features = pd.read_parquet(ROOT / "data/processed/features_smallcap.parquet",
                              columns=need)
    features["date"] = pd.to_datetime(features["date"])

    insider = pd.read_parquet(INSIDER_FP)
    features, ins_cols = add_insider_features(features, insider)
    nz = (features["insider_buy_count_90d"] > 0).mean()
    print(f"Insider features {ins_cols} merged. Rows with a buy in 90d: {nz:.1%}")

    train = features.dropna(subset=[V2_TARGET])
    sample = train.sample(min(200_000, len(train)), random_state=42)
    errors = (_leak.check_feature_names(ins_cols)
              + _leak.check_degenerate(sample, ins_cols)
              + _leak.check_pearson(sample, ins_cols)
              + _leak.check_per_date_spearman(sample, ins_cols))
    if errors:
        print("  LEAK GATE FAILED — experiment rejected:")
        for e in errors:
            print(f"    • {e}")
        sys.exit(2)
    print(f"  leak gate OK ({len(ins_cols)} new cols)")

    ohlcv = pd.read_parquet(ROOT / "data/processed/ohlcv_smallcap.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    feat_cols = feat_base + ins_cols
    res = run_walkforward(
        features, ohlcv, feat_cols,
        config_name="v4_feat_insider",
        lgb_params=PROD_PARAMS, objective_lambdarank=True,
        min_price=decision["min_price"], min_adv_usd=decision["min_adv_usd"],
        cost_bps=decision["cost_bps"], verbose=True,
    )
    save_result(res)
    agg = res.aggregate()

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
    verdict = "PROMOTE" if (c1 and c2 and c3 and c5) else "REJECT"
    print("\n" + "=" * 70)
    print(f"  INSIDER BUYING: {verdict}")
    print(f"  ICtr {base_agg['mean_ic_tradable']:+.4f} -> {agg['mean_ic_tradable']:+.4f}")
    print(f"  ret  {base_agg['mean_return']:+.2%} -> {agg['mean_return']:+.2%}")
    print(f"  WR   {base_agg['mean_win_rate']:.1%} -> {agg['mean_win_rate']:.1%}")
    print(f"  Sharpe {base_agg['mean_sharpe']:+.2f} -> {agg['mean_sharpe']:+.2f}")
    print(f"  +folds {base_agg['folds_positive_ret']} -> {agg['folds_positive_ret']}")
    print(f"  [c1={c1} c2={c2} c3={c3} c5={c5}]")
    print("=" * 70)
    print(f"\nTotal runtime: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
