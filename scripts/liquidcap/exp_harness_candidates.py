"""LIQUIDCAP feature research — full purged harness on the IC-screen survivors.

Trains GI+<extra> with the SAME frozen recipe (lambdarank 16 bins, 300 trees,
purge_days=20, MIN_TRAIN_END=2018, top-8, hold 20, pt40) and compares to the
cached GI champion under both cost models with a paired bootstrap + Šidák
haircut. A candidate set is only worth adopting if its own return CI clears 0
AND it beats GI past the multiple-testing correction.

Pass the surviving extras on the command line (space-separated):
    PYTHONPATH=src python scripts/liquidcap/exp_harness_candidates.py high_52w_prox resid_mom_12_1
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
FS15 = ["ret_252d", "sector_ret_60d", "rev_21d", "realized_vol_120d", "amihud_60d",
        "beta_60d", "downside_vol_60d", "sma_200", "ema_26", "max_dd_60d",
        "vol_of_vol_ratio", "adv_60d", "intraday_avg_20d", "price_roc_smooth_120d",
        "obv"]
LAMBDARANK = {"objective": "lambdarank", "metric": "ndcg", "num_leaves": 31,
              "max_depth": 6, "learning_rate": 0.05, "min_child_samples": 30,
              "subsample": 0.75, "colsample_bytree": 0.7, "reg_lambda": 5.0,
              "n_jobs": 1, "seed": 42, "verbose": -1}


def replay(name, ohlcv, spread):
    meta = json.loads((CACHE / name / "meta.json").read_text())
    rows = []
    for fm in meta:
        td = pd.read_parquet(CACHE / name / f"fold_{fm['fold']:02d}.parquet")
        td["date"] = pd.to_datetime(td["date"])
        ev = h._evaluate_fold(td, ohlcv, (pd.Timestamp(fm["test_start"]), pd.Timestamp(fm["test_end"])),
                              1.5, 500_000, h.ExitPolicy(profit_target=0.40), COST_BPS, spread)
        rows.append({"fold": fm["fold"], "test_start": fm["test_start"],
                     "total_return": ev["total_return"], "sharpe": ev["sharpe"],
                     "ic_tr": ev["mean_ic_tradable"]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("extras", nargs="+", help="extra features to add to GI")
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    t0 = time.time()

    h.V2_TARGET = "fwd_ret_20d_sector_rel"
    h.HOLD_DAYS = 20
    h.N_COHORTS = h.HOLD_DAYS // h.REBALANCE_EVERY
    h.MIN_TRAIN_END = pd.Timestamp("2018-01-01")

    # Load only what the harness needs — features_research.parquet has 472 cols
    # (~4GB); pulling all of them would blow up RAM for no reason.
    import pyarrow.parquet as pq
    schema_cols = set(pq.ParquetFile(DATA / "features_research.parquet").schema.names)
    need = (["date", "ticker", "fwd_ret_20d_sector_rel", "atr_pct_20d", "close",
             "adv_usd_20d", "cs_spread_20d"] + FS15 + list(args.extras))
    need = [c for c in dict.fromkeys(need) if c in schema_cols]
    feats = pd.read_parquet(DATA / "features_research.parquet", columns=need)
    feats["date"] = pd.to_datetime(feats["date"])
    fnd = pd.read_parquet(DATA / "fundamentals_daily.parquet")
    fnd["date"] = pd.to_datetime(fnd["date"])
    feats = feats.merge(fnd, on=["date", "ticker"], how="left")
    fund = [r for r in RATIOS if r in feats.columns]
    ohlcv = pd.read_parquet(DATA / "ohlcv_sp500.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    ohlcv = ohlcv[ohlcv.ticker != "SPY"]

    extras = [e for e in args.extras if e in feats.columns]
    missing = [e for e in args.extras if e not in feats.columns]
    if missing:
        print(f"  WARNING: not in matrix, skipped: {missing}")
    tag = args.tag or "GI_plus_" + "_".join(extras)
    cols = FS15 + fund + extras
    print(f"  config {tag}: {len(cols)} features (+{len(extras)}: {extras})", flush=True)

    if not (CACHE / tag / "meta.json").exists():
        h.run_walkforward(
            feats, ohlcv, cols, config_name=tag,
            lgb_params=LAMBDARANK, objective_lambdarank=True, n_bins=16,
            min_price=1.5, min_adv_usd=500_000,
            exit_policy=h.ExitPolicy(profit_target=0.40, time_stop=20),
            cost_bps=COST_BPS, cache_dir=CACHE / tag, num_boost_round=args.trees,
            purge_days=20, verbose=True)

    print(f"\n  === {tag} vs GI champion ===")
    for label, spread in [("flat 5bps", False), ("spread-aware", True)]:
        g = replay("GI_fs15_fund_20d", ohlcv, spread)
        c = replay(tag, ohlcv, spread)
        n = min(len(g), len(c))
        pb = mc.paired_bootstrap(c["total_return"].values[:n], g["total_return"].values[:n])
        g_mo = (1 + g["total_return"].mean()) ** (1 / 3) - 1
        c_mo = (1 + c["total_return"].mean()) ** (1 / 3) - 1
        gc = c[pd.to_datetime(c["test_start"]) >= CONFIRM_START]
        print(f"  --- {label} ---")
        print(f"  GI    {g['total_return'].mean():+7.2%}/f {g_mo:+6.2%}/mo Sh{g['sharpe'].mean():5.2f} IC{g['ic_tr'].mean():+.4f}")
        print(f"  {tag[:18]:18} {c['total_return'].mean():+7.2%}/f {c_mo:+6.2%}/mo Sh{c['sharpe'].mean():5.2f} IC{c['ic_tr'].mean():+.4f}")
        print(f"  Δ(cand-GI) {pb['mean_diff']:+.2%}/f [{pb['ci_lo']:+.2%},{pb['ci_hi']:+.2%}] "
              f"p(cand≤GI)={pb['p_le0']:.2f}  confirm {gc['total_return'].mean():+.1%}/f")
    print(f"\n  Runtime {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
