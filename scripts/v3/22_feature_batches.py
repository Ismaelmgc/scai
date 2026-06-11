"""V4 Phase 3: candidate feature batches through the filtered harness + leak gate.

Each batch = production 28 features + candidate columns, full 16-fold
walk-forward (training included — features change the model), evaluated
under the Phase-1 tradability filter and cost so results are comparable
to ``v4_filt_baseline``. New columns must pass the anti-leak checks
(imported from 18_verify_no_leak.py) BEFORE the harness runs.

Promotion criteria vs v4_filt_baseline (tradable-IC basis):
  1. mean_ic_tradable >= baseline + 0.002
  2. folds_positive_ret >= baseline
  3. mean_return >= 0.95 x baseline AND mean_win_rate >= baseline - 1pp
  4. leak gate passes
  5. at least two of {IC, return, WR} strictly better

Usage:
    PYTHONPATH=src python scripts/v3/22_feature_batches.py            # all cheap batches (E, B)
    PYTHONPATH=src python scripts/v3/22_feature_batches.py E B A     # specific batches
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "v3"))

from _v3_harness import (  # noqa: E402
    V2_FEATURES_BASE, V2_EDGAR_FEATURES, V2_TARGET,
    run_walkforward, save_result,
)

# Import leak checks from the numeric-prefixed module
_spec = importlib.util.spec_from_file_location(
    "verify_no_leak", ROOT / "scripts" / "v3" / "18_verify_no_leak.py")
_leak = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_leak)

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


def _top_importance_features(n: int = 10) -> list[str]:
    """Top-n production features by gain, from the deployed Booster."""
    with open(ROOT / "data/models/smallcap_v3_lambdarank.pkl", "rb") as fh:
        booster = pickle.load(fh)
    imp = pd.Series(booster.feature_importance(importance_type="gain"),
                    index=booster.feature_name())
    return imp.sort_values(ascending=False).head(n).index.tolist()


def build_batches() -> dict[str, list[str]]:
    """Candidate columns per batch. All names must exist in the parquet
    (availability is re-checked at load time)."""
    top10 = _top_importance_features(10)
    return {
        # E: cross-sectional rank transforms of the strongest base features
        "E_ranks": [f"{f}_rank" for f in top10],
        # B: unused microstructure / alpha features already computed
        "B_micro": [
            "up_volume_pct_5d", "up_volume_pct_20d", "trade_size_ratio",
            "cs_spread_20d", "vwap_dev_zscore", "ret_autocorr_20d",
            "ret_autocorr_60d", "info_ratio_20d", "info_ratio_60d",
            "vol_price_corr_20d", "max_dd_20d", "ret_skew_20d",
        ],
        # A: EDGAR fundamentals beyond the 2 in production (computed by
        # fundamentals.compute_edgar_features and merged as-of in pipeline;
        # included here only if present in the parquet)
        "A_fundamentals": [
            "roe", "roa", "leverage", "revenue_growth_yoy",
            "cash_ratio", "gross_margin", "net_margin", "asset_growth_yoy",
        ],
    }


def run_batch(name: str, candidates: list[str], ohlcv: pd.DataFrame,
              decision: dict, base_agg: dict) -> None:
    feat_base = V2_FEATURES_BASE + V2_EDGAR_FEATURES
    import pyarrow.parquet as pq
    schema = set(pq.read_schema(
        str(ROOT / "data/processed/features_smallcap.parquet")).names)
    available = [c for c in candidates if c in schema]
    missing = sorted(set(candidates) - set(available))
    if missing:
        print(f"  [{name}] not in parquet (skipped cols): {missing}")
    if not available:
        print(f"  [{name}] NO candidate columns available — batch skipped")
        return

    need = ["date", "ticker", V2_TARGET, "atr_pct_20d", "close",
            "adv_usd_20d", "cs_spread_20d"] + feat_base + available
    need = list(dict.fromkeys(need))  # dedupe (cs_spread_20d may be a candidate)
    features = pd.read_parquet(ROOT / "data/processed/features_smallcap.parquet",
                               columns=need)
    features["date"] = pd.to_datetime(features["date"])

    # ── Leak gate on the NEW columns ──
    train = features.dropna(subset=[V2_TARGET])
    sample = train.sample(min(200_000, len(train)), random_state=42)
    errors = (_leak.check_feature_names(available)
              + _leak.check_degenerate(sample, available)
              + _leak.check_pearson(sample, available)
              + _leak.check_per_date_spearman(sample, available))
    if errors:
        print(f"  [{name}] LEAK GATE FAILED — batch rejected:")
        for e in errors:
            print(f"    • {e}")
        return
    print(f"  [{name}] leak gate OK ({len(available)} new cols)")

    feat_cols = feat_base + available
    res = run_walkforward(
        features, ohlcv, feat_cols,
        config_name=f"v4_feat_{name}",
        lgb_params=PROD_PARAMS, objective_lambdarank=True,
        min_price=decision["min_price"], min_adv_usd=decision["min_adv_usd"],
        cost_bps=decision["cost_bps"], verbose=True,
    )
    save_result(res)
    agg = res.aggregate()

    # ── Promotion verdict ──
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
    print(f"  [{name}] {verdict}: ICtr {base_agg['mean_ic_tradable']:+.4f}->{agg['mean_ic_tradable']:+.4f} "
          f"ret {base_agg['mean_return']:+.1%}->{agg['mean_return']:+.1%} "
          f"WR {base_agg['mean_win_rate']:.1%}->{agg['mean_win_rate']:.1%} "
          f"+folds {base_agg['folds_positive_ret']}->{agg['folds_positive_ret']} "
          f"[c1={c1} c2={c2} c3={c3} c5={c5}]")


def main() -> None:
    t0 = time.time()
    decision = json.loads(DECISION_FP.read_text())
    base_agg = json.loads(BASELINE_FP.read_text())["aggregate"]
    batches = build_batches()

    requested = sys.argv[1:] or ["E_ranks", "B_micro"]
    requested = [r if r in batches else next(
        (k for k in batches if k.startswith(r)), r) for r in requested]

    ohlcv = pd.read_parquet(ROOT / "data/processed/ohlcv_smallcap.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    for name in requested:
        if name not in batches:
            print(f"Unknown batch {name!r} — choices: {list(batches)}")
            continue
        print(f"\n=== Batch {name} ===")
        run_batch(name, batches[name], ohlcv, decision, base_agg)

    print(f"\nTotal runtime: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
