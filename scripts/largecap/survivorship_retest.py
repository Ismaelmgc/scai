"""Survivorship re-test: add the LOSERS back and re-evaluate the overnight edge.

Combines the survivor universe (universe_largecap) with the reconstructed losers
(universe_losers: were $2-50B, now delisted/fell), rebuilds features on the union,
retrains the executable (1d-lag) overnight model, and reports net@5/10bps overall
and for the $2-7B tercile — side by side with the survivor-only numbers.

If the edge holds with losers in, it is not a survivorship artifact. If it
collapses, the +12%/fold was the bias.

Features matrix is cached (features_largecap_all.parquet) so re-runs skip the
8-12 min build. Run after download_largecap_losers.py.

Usage:
    PYTHONPATH=src python scripts/largecap/survivorship_retest.py
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
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "v3"))

from _v3_harness import V2_FEATURES_BASE, V2_TARGET, run_walkforward  # noqa: E402
from run_smallcap_pipeline import _add_lag_features  # noqa: E402

from app.features.pipeline import build_feature_matrix  # noqa: E402

_spec = importlib.util.spec_from_file_location("mc", ROOT / "scripts/v3/34_mc_validate.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

DATA = ROOT / "data" / "largecap"
CACHE = DATA / "cache"
FEAT_ALL = DATA / "features_largecap_all.parquet"
BASE = V2_FEATURES_BASE
# The combined matrix has ~496 cols × 1.77M rows (~7GB) — loading it all OOM-kills
# the box. The backtest only needs these ~30, so read/return lean.
NEEDED = ["date", "ticker", *BASE, "atr_pct_20d", "close", "adv_usd_20d", "cs_spread_20d"]
TOP_K, PRICE_FLOOR, RET_CLIP = 8, 3.0, 0.25
PROD_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg",
    "num_leaves": 31, "max_depth": 6, "learning_rate": 0.05,
    "min_child_samples": 30, "subsample": 0.75, "colsample_bytree": 0.7,
    "reg_lambda": 5.0, "n_jobs": 1, "seed": 42, "verbose": -1,
}


def build_combined_features() -> pd.DataFrame:
    if FEAT_ALL.exists():
        print(f"  combined features cached: {FEAT_ALL.name} (lean read, {len(NEEDED)} cols)")
        return pd.read_parquet(FEAT_ALL, columns=NEEDED)
    uni = pd.read_parquet(DATA / "universe_largecap.parquet")[["ticker", "sic_code", "name"]]
    los = pd.read_parquet(DATA / "universe_losers.parquet")[["ticker", "sic_code", "name"]]
    universe = pd.concat([uni, los], ignore_index=True).drop_duplicates("ticker")

    oh = pd.read_parquet(DATA / "ohlcv_largecap.parquet")
    lo = pd.read_parquet(DATA / "ohlcv_losers.parquet")
    oh = pd.concat([oh, lo], ignore_index=True).drop_duplicates(["date", "ticker"], keep="first")
    oh["date"] = pd.to_datetime(oh["date"])
    spy = oh[oh["ticker"] == "SPY"].copy()
    panel = oh[oh["ticker"] != "SPY"].copy()
    print(f"  combined panel: {len(panel):,} rows, {panel['ticker'].nunique()} tickers "
          f"(+{los['ticker'].nunique()} losers)")
    feats = build_feature_matrix(panel, fundamentals=None, market_df=spy,
                                 universe=universe.to_dict("records"), horizons=[1, 5, 10, 20])
    feats = _add_lag_features(feats)
    feats.to_parquet(FEAT_ALL, index=False)
    print(f"  saved {FEAT_ALL.name}: {len(feats):,} rows")
    del feats                                    # free the ~7GB full matrix
    return pd.read_parquet(FEAT_ALL, columns=NEEDED)


def cap_map() -> dict:
    uni = pd.read_parquet(DATA / "universe_largecap.parquet")
    los = pd.read_parquet(DATA / "universe_losers.parquet")
    m = dict(zip(uni["ticker"], uni["market_cap"], strict=False))
    m.update(dict(zip(los["ticker"], los["pit_cap"], strict=False)))  # losers: cap when large
    return m


def replay(cache_dir: Path, caps: dict, lo: float, hi: float, bps: int):
    meta = json.loads((cache_dir / "meta.json").read_text())
    out = []
    for fm in meta:
        td = pd.read_parquet(cache_dir / f"fold_{fm['fold']:02d}.parquet")
        td["date"] = pd.to_datetime(td["date"])
        td["cap"] = td["ticker"].map(caps)
        daily = []
        for _, day in td.groupby("date"):
            day = day[(day["close"] >= PRICE_FLOOR) & (day["cap"] >= lo) & (day["cap"] < hi)]
            if len(day) < TOP_K:
                continue
            r = day.sort_values("pred", ascending=False).head(TOP_K)[V2_TARGET].to_numpy()
            r = r[np.isfinite(r)]
            if len(r):
                daily.append((r - 2 * bps / 1e4).mean())
        if daily:
            s = pd.Series(daily)
            out.append(float((1 + s).prod()) - 1)
    return np.array(out)


def line(tag, arr):
    if len(arr) == 0:
        print(f"  {tag:22s}  (no data)")
        return
    p = mc.one_sample_bootstrap(arr)
    mo = (1 + arr.mean()) ** (1 / 3) - 1
    flag = " CI>0" if p["ci_lo"] > 0 else ""
    print(f"  {tag:22s} {arr.mean():+8.1%}/fold {mo:+7.1%}/mo  "
          f"[{p['ci_lo']:+.1%},{p['ci_hi']:+.1%}]{flag}  +folds {int((arr>0).sum())}/{len(arr)}")


def main() -> None:
    t0 = time.time()
    feats = build_combined_features()
    feats["date"] = pd.to_datetime(feats["date"])
    oh = pd.concat([pd.read_parquet(DATA / "ohlcv_largecap.parquet"),
                    pd.read_parquet(DATA / "ohlcv_losers.parquet")], ignore_index=True)
    oh = oh.drop_duplicates(["date", "ticker"], keep="first")
    oh["date"] = pd.to_datetime(oh["date"])
    o = oh[oh["ticker"] != "SPY"].sort_values(["ticker", "date"]).copy()
    o["overnight"] = o.groupby("ticker")["open"].shift(-1) / o["close"] - 1
    o["overnight_exec"] = o.groupby("ticker")["overnight"].shift(-1)
    feats = feats.merge(o[["date", "ticker", "overnight_exec"]], on=["date", "ticker"], how="left")
    feats["overnight_exec"] = feats["overnight_exec"].clip(-RET_CLIP, RET_CLIP)

    feats[V2_TARGET] = feats["overnight_exec"]
    cache_dir = CACHE / "lc_overnight_exec_all"
    run_walkforward(feats, oh, BASE, config_name="lc_overnight_exec_all", cache_dir=cache_dir,
                    lgb_params=PROD_PARAMS, objective_lambdarank=True, num_boost_round=250,
                    verbose=False)

    caps = cap_map()
    q1 = 6.6e9  # same low-tercile ceiling as survivor-only run, for comparability
    print("\n  Survivorship re-test — overnight (exec) WITH losers added\n")
    for bps in (5, 10):
        print(f"  cost {bps}bps:")
        line(f"ALL $2-50B @{bps}", replay(cache_dir, caps, 0, 1e15, bps))
        line(f"low $2-7B  @{bps}", replay(cache_dir, caps, 0, q1, bps))
    print("\n  Compare survivor-only: ALL net5 +12.0%/fold, low net5 +9.9%/fold.")
    print(f"  Runtime {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
