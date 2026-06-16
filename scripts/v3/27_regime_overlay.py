"""V4 experiment: market-regime timing overlay (hold cash in risk-off regimes).

The fold-by-fold results have a few catastrophic windows (e.g. -12%, -21%) that
drag WR and return. Idea: skip rebalances when the market is in a risk-off
regime, holding cash that period, and trade only risk-on. Model UNCHANGED — this
gates WHEN we deploy the same cached predictions (replay, seconds).

Regime is computed PIT-safe from SPY daily closes (only past data):
  - sma50 / sma100: SPY close above its N-day moving average (trend filter)
  - mom20: SPY 20-day return > 0
Control (no skip) must reproduce v4_filt_baseline.

Usage:
    PYTHONPATH=src python scripts/v3/27_regime_overlay.py
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

CACHE_DIR = ROOT / "data" / "v3_benchmarks" / "cache" / "v4_nometa"
DECISION_FP = ROOT / "data" / "v3_benchmarks" / "v4_filter_decision.json"
BASELINE_FP = ROOT / "data" / "v3_benchmarks" / "v4_filt_baseline.json"
SPY_FP = ROOT / "data" / "processed" / "smallcap_spy.parquet"


def regime_series() -> dict[str, pd.Series]:
    """Boolean risk-on Series (indexed by date) for several regime definitions."""
    spy = pd.read_parquet(SPY_FP)[["date", "close"]].copy()
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date").set_index("date")
    c = spy["close"]
    return {
        "sma50": c > c.rolling(50).mean(),
        "sma100": c > c.rolling(100).mean(),
        "mom20": c.pct_change(20) > 0,
    }


def main() -> None:
    t0 = time.time()
    decision = json.loads(DECISION_FP.read_text())
    base_agg = json.loads(BASELINE_FP.read_text())["aggregate"]
    mp, ma, cb = decision["min_price"], decision["min_adv_usd"], decision["cost_bps"]

    ohlcv = pd.read_parquet(ROOT / "data/processed/ohlcv_smallcap.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    meta = json.loads((CACHE_DIR / "meta.json").read_text())
    folds = []
    all_dates: set = set()
    for fm in meta:
        td = pd.read_parquet(CACHE_DIR / f"fold_{fm['fold']:02d}.parquet")
        td["date"] = pd.to_datetime(td["date"])
        folds.append((fm, td))
        all_dates.update(td["date"].unique())
    policy = ExitPolicy()
    regimes = regime_series()

    def skip_set(risk_on: pd.Series) -> set:
        """Risk-off dates (as the cache's datetime64 values), default risk-on."""
        off = set()
        for d in all_dates:
            v = risk_on.get(pd.Timestamp(d))
            if v is not None and not bool(v):  # missing/None → treat as risk-on
                off.add(d)
        return off

    def eval_skip(skip: set | None) -> dict:
        rr = RunResult(config_name="regime", feat_cols=[], n_features=0)
        for fm, td in folds:
            ev = _evaluate_fold(
                td, ohlcv,
                (pd.Timestamp(fm["test_start"]), pd.Timestamp(fm["test_end"])),
                mp, ma, policy, cb, False, skip_dates=skip,
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

    print(f"\n  Regime timing overlay — skip risk-off rebalances, 16 folds, {cb:g}bps\n")
    hdr = "  regime      netRet   Sharpe    WR    +folds   trades"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    print(f"  BASELINE   {base_agg['mean_return']:+7.1%}  {base_agg['mean_sharpe']:+5.2f}  "
          f"{base_agg['mean_win_rate']:4.0%}  {base_agg['folds_positive_ret']:>5}   "
          f"{base_agg['total_trades']:5d}")
    ctrl = eval_skip(None)
    print(f"  control    {ctrl['mean_return']:+7.1%}  {ctrl['mean_sharpe']:+5.2f}  "
          f"{ctrl['mean_win_rate']:4.0%}  {ctrl['folds_positive_ret']:>5}   "
          f"{ctrl['total_trades']:5d}  (=baseline)")
    print("  " + "-" * (len(hdr) - 2))
    for name, ron in regimes.items():
        a = eval_skip(skip_set(ron))
        print(f"  {name:9s}  {a['mean_return']:+7.1%}  {a['mean_sharpe']:+5.2f}  "
              f"{a['mean_win_rate']:4.0%}  {a['folds_positive_ret']:>5}   {a['total_trades']:5d}")

    print("\n  (Gana si ret/Sharpe/WR suben con folds robustos vs baseline.)")
    print(f"  Runtime: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
