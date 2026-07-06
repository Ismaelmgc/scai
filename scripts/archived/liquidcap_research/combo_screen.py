"""LIQUIDCAP — final spec combos: the two natural marriages of the screen winners.

  GI  fs15 + 10 EDGAR fundamental ratios, 20d   (G won on features, I on IC —
      do they stack?)
  GJ  fs15 at 10d horizon (hold/purge/time_stop 10)  (J had the best monthly
      return and confirm block; does it hold on the lean feature set?)

Trials #12 and #13 of the session (Šidák M=13). Judged vs G_lambdarank_fs15
(the frozen 20d champion) with the standard battery, both cost models.

Usage:
    PYTHONPATH=src python scripts/liquidcap/combo_screen.py [--trees N]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "v3"))
sys.path.insert(0, str(ROOT / "scripts" / "liquidcap"))

import _v3_harness as h  # noqa: E402
from build_fundamentals import RATIOS  # noqa: E402

_spec = importlib.util.spec_from_file_location("mc", ROOT / "scripts/v3/34_mc_validate.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

DATA = ROOT / "data" / "liquidcap"
CACHE = DATA / "cache"
COST_BPS = 5.0
CONFIRM_START = pd.Timestamp("2024-07-01")
M_TRIALS = 13

FS15 = ["ret_252d", "sector_ret_60d", "rev_21d", "realized_vol_120d", "amihud_60d",
        "beta_60d", "downside_vol_60d", "sma_200", "ema_26", "max_dd_60d",
        "vol_of_vol_ratio", "adv_60d", "intraday_avg_20d", "price_roc_smooth_120d",
        "obv"]

LAMBDARANK_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg",
    "num_leaves": 31, "max_depth": 6,
    "learning_rate": 0.05, "min_child_samples": 30,
    "subsample": 0.75, "colsample_bytree": 0.7,
    "reg_lambda": 5.0, "n_jobs": 1, "seed": 42, "verbose": -1,
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
                     "ic_tr": ev["mean_ic_tradable"], "total_return": ev["total_return"],
                     "sharpe": ev["sharpe"]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=300)
    args = ap.parse_args()
    t0 = time.time()

    feats = pd.read_parquet(DATA / "features_liquidcap.parquet")
    feats["date"] = pd.to_datetime(feats["date"])
    fnd = pd.read_parquet(DATA / "fundamentals_daily.parquet")
    fnd["date"] = pd.to_datetime(fnd["date"])
    feats = feats.merge(fnd, on=["date", "ticker"], how="left")
    fund = [r for r in RATIOS if r in feats.columns]
    ohlcv = pd.read_parquet(DATA / "ohlcv_sp500.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    ohlcv = ohlcv[ohlcv.ticker != "SPY"]
    h.MIN_TRAIN_END = pd.Timestamp("2018-01-01")

    jobs = {
        "GI_fs15_fund_20d": (FS15 + fund, "fwd_ret_20d_sector_rel", 20),
        "GJ_fs15_10d": (FS15, "fwd_ret_10d_sector_rel", 10),
    }
    for name, (cols, target, hold) in jobs.items():
        if (CACHE / name / "meta.json").exists():
            print(f"  ({name}: cache exists, skip)")
            continue
        set_horizon(target, hold)
        policy = h.ExitPolicy(profit_target=0.40, time_stop=hold)
        print(f"\n=== training {name} ({len(cols)} feats, hold={hold}) ===", flush=True)
        h.run_walkforward(
            feats, ohlcv, cols, config_name=name,
            lgb_params=LAMBDARANK_PARAMS, objective_lambdarank=True, n_bins=16,
            min_price=1.5, min_adv_usd=500_000, exit_policy=policy,
            cost_bps=COST_BPS, cache_dir=CACHE / name, num_boost_round=args.trees,
            purge_days=hold, verbose=True,
        )

    print(f"\n  === COMBO RESULTS vs G (frozen champion), Sidak M={M_TRIALS} ===")
    for label, spread in [("flat 5bps", False), ("spread-aware", True)]:
        print(f"\n  --- cost {label} ---", flush=True)
        g = replay("G_lambdarank_fs15", ohlcv, h.ExitPolicy(profit_target=0.40),
                   "fwd_ret_20d_sector_rel", 20, spread)
        mo = (1 + g["total_return"].mean()) ** (1 / 3) - 1
        print(f"  G (champion)     {g['total_return'].mean():+7.1%}/f {mo:+6.2%}/mo "
              f"Sh{g['sharpe'].mean():5.2f}")
        for name, (cols, target, hold) in jobs.items():
            pol = h.ExitPolicy(profit_target=0.40, time_stop=hold)
            df = replay(name, ohlcv, pol, target, hold, spread)
            n = min(len(df), len(g))
            pb = mc.paired_bootstrap(df["total_return"].values[:n],
                                     g["total_return"].values[:n])
            ps = mc.sidak(pb["p_le0"], M_TRIALS)
            conf = df[pd.to_datetime(df["test_start"]) >= CONFIRM_START]
            mo = (1 + df["total_return"].mean()) ** (1 / 3) - 1
            print(f"  {name:16} {df['total_return'].mean():+7.1%}/f {mo:+6.2%}/mo "
                  f"Sh{df['sharpe'].mean():5.2f} IC{df['ic_tr'].mean():+.4f} "
                  f"dG{pb['mean_diff']:+.1%}[{pb['ci_lo']:+.1%},{pb['ci_hi']:+.1%}] "
                  f"pSidak={ps:.2f} conf{conf['total_return'].mean():+6.1%}/f", flush=True)

    print(f"\n  Runtime {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
