"""V4 experiment: short interest as a SELECTION OVERLAY (not a model feature).

Adding short interest as a LambdaRank feature was rejected (24_short_interest.py)
— it degraded the model. But the standalone signal is real and correctly signed
(high days-to-cover → weaker forward returns). So test it where that sign is
directly actionable: as an overlay that drops the most heavily-shorted names from
the candidate set BEFORE picking the top-8. The model (its cached predictions)
is UNCHANGED — this only changes selection.

Replays the production-model cache (v4_nometa) under the Phase-1 filter + 15bps,
joining PIT-safe short interest per fold, for several exclusion quantiles. Q=0 is
a control that must reproduce v4_filt_baseline. An overlay improves the strategy
if return / Sharpe / WR rise while folds stay robust (the feature-promotion IC
gate doesn't apply — the overlay leaves IC unchanged by construction).

Usage:
    PYTHONPATH=src python scripts/v3/25_si_overlay.py
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

from app.features.short_interest import add_short_interest_features  # noqa: E402

CACHE_DIR = ROOT / "data" / "v3_benchmarks" / "cache" / "v4_nometa"
DECISION_FP = ROOT / "data" / "v3_benchmarks" / "v4_filter_decision.json"
BASELINE_FP = ROOT / "data" / "v3_benchmarks" / "v4_filt_baseline.json"
SI_FP = ROOT / "data" / "short_interest.parquet"

OVERLAY_COL = "si_days_to_cover"
QS = [0.0, 0.10, 0.20, 0.30]  # fraction of most-shorted names excluded


def main() -> None:
    t0 = time.time()
    decision = json.loads(DECISION_FP.read_text())
    base_agg = json.loads(BASELINE_FP.read_text())["aggregate"]
    mp, ma, cb = decision["min_price"], decision["min_adv_usd"], decision["cost_bps"]

    ohlcv = pd.read_parquet(ROOT / "data/processed/ohlcv_smallcap.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    si = pd.read_parquet(SI_FP)

    # Pre-load each cached fold with SI joined once.
    meta = json.loads((CACHE_DIR / "meta.json").read_text())
    folds = []
    for fm in meta:
        td = pd.read_parquet(CACHE_DIR / f"fold_{fm['fold']:02d}.parquet")
        td["date"] = pd.to_datetime(td["date"])
        td, _ = add_short_interest_features(td, si)
        folds.append((fm, td))
    policy = ExitPolicy()  # default — matches the gate baseline

    def eval_q(q: float) -> dict:
        rr = RunResult(config_name=f"v4_si_overlay_q{int(q*100)}", feat_cols=[], n_features=0)
        for fm, td in folds:
            ev = _evaluate_fold(
                td, ohlcv,
                (pd.Timestamp(fm["test_start"]), pd.Timestamp(fm["test_end"])),
                mp, ma, policy, cb, False,
                overlay_col=OVERLAY_COL, overlay_exclude_q=q,
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

    print(f"\n  SI overlay — exclude top-q most-shorted, 16 folds, {cb:g}bps\n")
    hdr = "  exclude%   netRet   Sharpe    WR    +folds   trades"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    print(f"  BASELINE  {base_agg['mean_return']:+7.1%}  {base_agg['mean_sharpe']:+5.2f}  "
          f"{base_agg['mean_win_rate']:4.0%}  {base_agg['folds_positive_ret']:>5}   "
          f"{base_agg['total_trades']:5d}")
    print("  " + "-" * (len(hdr) - 2))
    for q in QS:
        a = eval_q(q)
        tag = "  (=control)" if q == 0 else ""
        print(f"  {q*100:5.0f}%    {a['mean_return']:+7.1%}  {a['mean_sharpe']:+5.2f}  "
              f"{a['mean_win_rate']:4.0%}  {a['folds_positive_ret']:>5}   "
              f"{a['total_trades']:5d}{tag}")

    print("  (Overlay gana si ret/Sharpe/WR suben con folds robustos; Q=0 ~ baseline.)")
    print(f"\n  Runtime: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
