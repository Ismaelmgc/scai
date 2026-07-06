"""LIQUIDCAP — horizon screen: 10d and 5d holding vs the 20d anchor.

The transferable lesson from Jane Street / Medallion economics (not their
infrastructure): annual return ≈ edge-per-trade × trades-per-year. Jane Street
monetizes bps of edge over astronomical turnover; Medallion (capacity-capped,
short holding) does 39-76%/yr while the same firm's own long-horizon scalable
fund (RIEF) can lose 22% the same year. Retail can't compete on microseconds,
but the gradient is real and OURS to test: on S&P 500 names at ~5bps/side,
halving the holding period doubles independent bets/year — the same experiment
that FAILED on small-caps (48: 10d won at flat cost, died on spread) becomes
viable where spreads are 10-100x smaller.

Configs (trials #10 and #11 in the session Šidák):
  J  lambdarank_h10  base+new features, target fwd_ret_10d_sector_rel,
                     HOLD=10, purge=10, time_stop=10
  K  lambdarank_h5   target fwd_ret_5d_sector_rel, HOLD=5, purge=5, time_stop=5

Gates: paired bootstrap vs A (20d anchor, same fold windows) Šidák(11) < 0.05,
CONFIRM block (>=2024-07) > 0, and BOTH cost models (5bps flat AND
spread-aware) must agree — turnover doubles/quadruples, so costs must be
confronted, not assumed away.

Usage:
    PYTHONPATH=src python scripts/liquidcap/horizon_screen.py [--trees N]
"""
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(ROOT / "scripts" / "liquidcap"))

import _v3_harness as h  # noqa: E402
from build_features_liquidcap import NEW_FEATURES  # noqa: E402

_spec = importlib.util.spec_from_file_location("mc", ROOT / "scripts/v3/34_mc_validate.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

DATA = ROOT / "data" / "liquidcap"
CACHE = DATA / "cache"
COST_BPS = 5.0
CONFIRM_START = pd.Timestamp("2024-07-01")
M_TRIALS = 11

LAMBDARANK_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg",
    "num_leaves": 31, "max_depth": 6,
    "learning_rate": 0.05, "min_child_samples": 30,
    "subsample": 0.75, "colsample_bytree": 0.7,
    "reg_lambda": 5.0, "n_jobs": 1, "seed": 42, "verbose": -1,
}

HORIZONS = {
    "J_lambdarank_h10": ("fwd_ret_10d_sector_rel", 10),
    "K_lambdarank_h5": ("fwd_ret_5d_sector_rel", 5),
}


def set_horizon(target: str, hold: int) -> None:
    h.V2_TARGET = target
    h.HOLD_DAYS = hold
    h.N_COHORTS = max(hold // h.REBALANCE_EVERY, 1)


def replay(name, ohlcv, policy, target, hold, spread=False):
    set_horizon(target, hold)
    meta = json.loads((CACHE / name / "meta.json").read_text())
    rows = []
    for fm in meta:
        td = pd.read_parquet(CACHE / name / f"fold_{fm['fold']:02d}.parquet")
        td["date"] = pd.to_datetime(td["date"])
        ev = h._evaluate_fold(
            td, ohlcv, (pd.Timestamp(fm["test_start"]), pd.Timestamp(fm["test_end"])),
            1.5, 500_000, policy, COST_BPS, spread)
        rows.append({"fold": fm["fold"], "test_start": fm["test_start"],
                     "ic_tr": ev["mean_ic_tradable"],
                     **{k: ev[k] for k in ("total_return", "sharpe", "win_rate",
                                           "n_trades", "max_dd")}})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=300)
    args = ap.parse_args()
    t0 = time.time()

    need_targets = [t for t, _ in HORIZONS.values()]
    feats = pd.read_parquet(DATA / "features_liquidcap.parquet")
    feats["date"] = pd.to_datetime(feats["date"])
    missing = [t for t in need_targets if t not in feats.columns]
    if missing:
        raise SystemExit(f"targets missing from features parquet: {missing}")
    ohlcv = pd.read_parquet(DATA / "ohlcv_sp500.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    ohlcv = ohlcv[ohlcv.ticker != "SPY"]

    base = [f for f in h.V2_FEATURES_BASE if f in feats.columns]
    new = [f for f in NEW_FEATURES if f in feats.columns]
    h.MIN_TRAIN_END = pd.Timestamp("2018-01-01")

    for name, (target, hold) in HORIZONS.items():
        cache = CACHE / name
        if (cache / "meta.json").exists():
            print(f"  ({name}: cache exists, skip)")
            continue
        set_horizon(target, hold)
        policy = h.ExitPolicy(profit_target=0.40, time_stop=hold)
        print(f"\n=== training {name} (hold={hold}, purge={hold}) ===", flush=True)
        h.run_walkforward(
            feats, ohlcv, base + new, config_name=name,
            lgb_params=LAMBDARANK_PARAMS, objective_lambdarank=True, n_bins=16,
            min_price=1.5, min_adv_usd=500_000, exit_policy=policy,
            cost_bps=COST_BPS, cache_dir=cache, num_boost_round=args.trees,
            purge_days=hold, verbose=True,
        )

    print(f"\n  === HORIZON RESULTS vs A (20d anchor), Sidak M={M_TRIALS} ===")
    for label, spread in [("flat 5bps", False), ("spread-aware", True)]:
        print(f"\n  --- cost {label} ---", flush=True)
        pol20 = h.ExitPolicy(profit_target=0.40)
        a = replay("A_lambdarank_base", ohlcv, pol20,
                   "fwd_ret_20d_sector_rel", 20, spread)
        mo = (1 + a["total_return"].mean()) ** (1 / 3) - 1
        print(f"  A 20d anchor     {a['total_return'].mean():+7.1%}/f {mo:+6.2%}/mo "
              f"Sh{a['sharpe'].mean():5.2f}")
        for name, (target, hold) in HORIZONS.items():
            pol = h.ExitPolicy(profit_target=0.40, time_stop=hold)
            df = replay(name, ohlcv, pol, target, hold, spread)
            n = min(len(df), len(a))
            pb = mc.paired_bootstrap(df["total_return"].values[:n],
                                     a["total_return"].values[:n])
            ps = mc.sidak(pb["p_le0"], M_TRIALS)
            conf = df[pd.to_datetime(df["test_start"]) >= CONFIRM_START]
            mo = (1 + df["total_return"].mean()) ** (1 / 3) - 1
            print(f"  {name:16} {df['total_return'].mean():+7.1%}/f {mo:+6.2%}/mo "
                  f"Sh{df['sharpe'].mean():5.2f} IC{df['ic_tr'].mean():+.4f} "
                  f"dA{pb['mean_diff']:+.1%}[{pb['ci_lo']:+.1%},{pb['ci_hi']:+.1%}] "
                  f"pSidak={ps:.2f} conf{conf['total_return'].mean():+6.1%}/f", flush=True)

    print(f"\n  Runtime {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
