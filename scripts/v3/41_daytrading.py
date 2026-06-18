"""V4 experiment: short-horizon / day-trading viability (daily OHLCV).

Tests whether the ranker can pick names for a 1-day strategy instead of the 20d
swing. Two horizons, both PIT-safe (decide on features as-of close D):
  overnight : buy close[D],   sell open[D+1]   -> target = open[D+1]/close[D] - 1
  intraday  : buy open[D+1],  sell close[D+1]  -> target = close[D+1]/open[D+1] - 1

For each, train a LambdaRank on that forward return, then DAILY-rebalance the
tradable top-8 (1-day hold, no cohorts/stops) and report gross, net@30bps, and
net@spread. Day-trading small-caps turns over ~252x/yr, so the honest question is
whether any gross edge survives the cost wall. MC vs zero on the per-fold return.

Usage:
    PYTHONPATH=src python scripts/v3/41_daytrading.py
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

from _v3_harness import (  # noqa: E402
    V2_EDGAR_FEATURES,
    V2_FEATURES_BASE,
    V2_TARGET,
    run_walkforward,
)

from app.features.tradability import tradable_mask  # noqa: E402

_spec = importlib.util.spec_from_file_location("mc", ROOT / "scripts/v3/34_mc_validate.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

DECISION_FP = ROOT / "data" / "v3_benchmarks" / "v4_filter_decision.json"
CACHE_ROOT = ROOT / "data" / "v3_benchmarks" / "cache"
BASE = V2_FEATURES_BASE + V2_EDGAR_FEATURES
TOP_K = 8
PROD_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg",
    "num_leaves": 31, "max_depth": 6, "learning_rate": 0.05,
    "min_child_samples": 30, "subsample": 0.75, "colsample_bytree": 0.7,
    "reg_lambda": 5.0, "n_jobs": 1, "seed": 42, "verbose": -1,
}


def short_targets(ohlcv: pd.DataFrame) -> pd.DataFrame:
    oh = ohlcv.sort_values(["ticker", "date"]).copy()
    g = oh.groupby("ticker")
    nxt_open = g["open"].shift(-1)
    nxt_close = g["close"].shift(-1)
    oh["overnight"] = nxt_open / oh["close"] - 1          # close[D]->open[D+1]
    oh["intraday"] = nxt_close / nxt_open - 1             # open[D+1]->close[D+1]
    return oh[["date", "ticker", "overnight", "intraday"]]


def daily_replay(cache_dir, mp, ma, cb) -> pd.DataFrame:
    """Daily-rebalanced top-8, 1-day hold. target column = the forward 1d return."""
    meta = json.loads((Path(cache_dir) / "meta.json").read_text())
    rows = []
    for fm in meta:
        td = pd.read_parquet(Path(cache_dir) / f"fold_{fm['fold']:02d}.parquet")
        td["date"] = pd.to_datetime(td["date"])
        td = td[tradable_mask(td, mp, ma)]
        gross, net30, netsp, trades_net = [], [], [], []
        for _, day in td.groupby("date"):
            if len(day) < TOP_K:
                continue
            picks = day.sort_values("pred", ascending=False).head(TOP_K)
            r = picks[V2_TARGET].to_numpy()
            r = r[np.isfinite(r)]
            if len(r) == 0:
                continue
            sp = picks["cs_spread_20d"].to_numpy()
            sp = np.where(np.isfinite(sp) & (sp > 0), np.minimum(sp, 0.10), 0.0)[:len(r)]
            c30 = 2 * cb / 1e4
            gross.append(r.mean())
            net30.append((r - c30).mean())
            netsp.append((r - np.maximum(c30, sp)).mean())
            trades_net.extend((r - c30).tolist())
        if not gross:
            continue

        def agg(daily):
            s = pd.Series(daily)
            cum = float((1 + s).prod()) - 1
            shp = float(s.mean() / s.std() * np.sqrt(252)) if s.std() > 0 else 0.0
            return cum, shp
        gc, gs = agg(gross)
        n30c, n30s = agg(net30)
        nspc, _ = agg(netsp)
        rows.append({"fold": fm["fold"], "gross": gc, "gross_sharpe": gs,
                     "net30": n30c, "net30_sharpe": n30s, "netsp": nspc,
                     "wr_net30": float(np.mean([t > 0 for t in trades_net])),
                     "n": len(trades_net)})
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    decision = json.loads(DECISION_FP.read_text())
    mp, ma, cb = decision["min_price"], decision["min_adv_usd"], decision["cost_bps"]

    need = (["date", "ticker", V2_TARGET, "atr_pct_20d", "close", "adv_usd_20d",
             "cs_spread_20d"] + BASE)
    need = list(dict.fromkeys(need))
    feats = pd.read_parquet(ROOT / "data/processed/features_smallcap.parquet", columns=need)
    feats["date"] = pd.to_datetime(feats["date"])
    ohlcv = pd.read_parquet(ROOT / "data/processed/ohlcv_smallcap.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    st = short_targets(ohlcv)
    feats = feats.merge(st, on=["date", "ticker"], how="left")

    print(f"\n  Day-trading viability — daily top-8, 1-day hold, {cb:g}bps + spread\n")
    print(f"  {'horizon':10s} {'gross':>7} {'grSh':>5} {'net30':>7} {'n30Sh':>5} "
          f"{'netSpread':>9} {'WRnet':>6} {'verdict':>24}")
    for name, col in [("overnight", "overnight"), ("intraday", "intraday")]:
        f2 = feats.copy()
        f2[V2_TARGET] = f2[col]
        cache_dir = CACHE_ROOT / f"v4_dt_{name}"
        run_walkforward(
            f2, ohlcv, BASE, config_name=f"v4_dt_{name}", cache_dir=cache_dir,
            lgb_params=PROD_PARAMS, objective_lambdarank=True, num_boost_round=250,
            min_price=mp, min_adv_usd=ma, cost_bps=cb, verbose=False,
        )
        df = daily_replay(cache_dir, mp, ma, cb)
        pg = mc.one_sample_bootstrap(df["gross"].values)
        pn = mc.one_sample_bootstrap(df["net30"].values)
        verdict = ("net edge (CI>0)" if pn["ci_lo"] > 0
                   else "gross only, costs kill" if pg["ci_lo"] > 0
                   else "no edge")
        wrn = (df["wr_net30"] * df["n"]).sum() / df["n"].sum()
        print(f"  {name:10s} {df['gross'].mean():+7.1%} {df['gross_sharpe'].mean():+5.2f} "
              f"{df['net30'].mean():+7.1%} {df['net30_sharpe'].mean():+5.2f} "
              f"{df['netsp'].mean():+9.1%} {wrn:6.0%} {verdict:>24}")

    print(f"\n  (per-fold returns; gross=signal exists?, net=survives costs?)  "
          f"Runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
