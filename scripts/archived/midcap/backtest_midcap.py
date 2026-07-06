"""#6 Mid-cap ($2-10B) 20d swing backtest — does the production swing edge hold on
mid-caps (less arbitraged than large, lower-cost + less manipulable than small)?

Same harness/model/exit as production: LambdaRank on fwd_ret_20d_sector_rel (here
date-relative, sectors Unknown), 26 base features, top-8, 20d hold, rebalance 5d,
pt40 exit, 15bps+spread, $1.50/$500K tradability gate (won't bind on mid-caps).
16-fold walk-forward + MC vs zero, contextualised against the small-cap baseline.

Usage:
    PYTHONPATH=src python scripts/midcap/backtest_midcap.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "v3"))

from _v3_harness import V2_FEATURES_BASE, ExitPolicy, run_walkforward  # noqa: E402

_spec = importlib.util.spec_from_file_location("mc", ROOT / "scripts/v3/34_mc_validate.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

DATA = ROOT / "data" / "midcap"
DEC = json.loads((ROOT / "data/v3_benchmarks/v4_filter_decision.json").read_text())
MP, MA, CB = DEC["min_price"], DEC["min_adv_usd"], DEC["cost_bps"]
BASE = V2_FEATURES_BASE


def main() -> None:
    t0 = time.time()
    feats = pd.read_parquet(DATA / "features_midcap.parquet")
    feats["date"] = pd.to_datetime(feats["date"])
    ohlcv = pd.read_parquet(DATA / "ohlcv_midcap.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    # yfinance lacks vwap/transactions, so avg_trade_size_20d & vwap_dev_avg_20d
    # can't be computed → use the intersection (24 of 26 base features).
    base = [f for f in BASE if f in feats.columns]
    dropped = [f for f in BASE if f not in feats.columns]
    if dropped:
        print(f"  (dropped {len(dropped)} base features missing on yfinance: {dropped})")

    pol_fields = set(ExitPolicy.__dataclass_fields__)
    policy = ExitPolicy(**{k: v for k, v in DEC["exit_policy"].items() if k in pol_fields})

    res = run_walkforward(
        feats, ohlcv, base, config_name="midcap_swing",
        cache_dir=DATA / "cache" / "midcap_swing",
        objective_lambdarank=True, num_boost_round=250,
        min_price=MP, min_adv_usd=MA, exit_policy=policy, cost_bps=CB,
        use_spread_cost=True, verbose=False,
    )
    folds = res.folds
    rets = np.array([f.total_return for f in folds])
    alphas = np.array([f.total_return - f.market_return for f in folds])
    wr = sum(f.win_rate * f.n_trades for f in folds) / sum(f.n_trades for f in folds)
    sharpe = np.mean([f.sharpe for f in folds])
    mo = (1 + rets.mean()) ** (1 / 3) - 1
    p = mc.one_sample_bootstrap(rets)
    pa = mc.one_sample_bootstrap(alphas)

    print(f"\n  Mid-cap $2-10B 20d swing — {feats['ticker'].nunique()} names, "
          f"{len(folds)} folds, top-8 pt40, {CB:g}bps+spread\n")
    print(f"  return/fold : {rets.mean():+.1%}  ({mo:+.1%}/mo)  "
          f"CI[{p['ci_lo']:+.1%},{p['ci_hi']:+.1%}]{' CI>0' if p['ci_lo'] > 0 else ''}")
    print(f"  alpha/fold  : {alphas.mean():+.1%}  CI[{pa['ci_lo']:+.1%},{pa['ci_hi']:+.1%}]"
          f"{' CI>0' if pa['ci_lo'] > 0 else ''}")
    print(f"  Sharpe      : {sharpe:.2f}    WR : {wr:.0%}    "
          f"+folds {int((rets > 0).sum())}/{len(folds)}")
    print(f"  mean IC     : {np.mean([f.mean_ic for f in folds]):.4f}  "
          f"(tradable {np.mean([f.mean_ic_tradable for f in folds]):.4f})")
    print("\n  vs small-cap production baseline ~+8.4%/fold WR58% Sharpe2.4 (v4_fs_base28).")
    print(f"  Runtime {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
