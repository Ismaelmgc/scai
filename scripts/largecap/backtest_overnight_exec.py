"""Executability test: does the overnight edge survive a 1-day FEATURE LAG?

Operational reality: the signal pipeline lands ~1 AM (needs close[D] EOD data +
CI delay), too late to buy at close[D]. The fix is to decouple feature date from
entry: features as-of close[D] -> signal ready ~1 AM of D+1 -> buy close[D+1]
(MOC, hours of margin) -> sell open[D+2] (MOO). Realized return is then
open[D+2]/close[D+1]-1, predicted from close[D] features (1 trading-day stale).

This trains a LambdaRank on that LAGGED target and compares to the ideal (no-lag)
overnight already cached. If the edge holds, the strategy is executable with the
existing ~1 AM infra (just add an MOC submission leg). MC bootstrap as before.

Usage:
    PYTHONPATH=src python scripts/largecap/backtest_overnight_exec.py
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

from _v3_harness import V2_FEATURES_BASE, V2_TARGET, run_walkforward  # noqa: E402

_spec = importlib.util.spec_from_file_location("mc", ROOT / "scripts/v3/34_mc_validate.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

DATA = ROOT / "data" / "largecap"
CACHE_ROOT = DATA / "cache"
BASE = V2_FEATURES_BASE
TOP_K, PRICE_FLOOR, RET_CLIP = 8, 3.0, 0.25
BPS_TIERS = [5, 10, 30]
PROD_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg",
    "num_leaves": 31, "max_depth": 6, "learning_rate": 0.05,
    "min_child_samples": 30, "subsample": 0.75, "colsample_bytree": 0.7,
    "reg_lambda": 5.0, "n_jobs": 1, "seed": 42, "verbose": -1,
}


def agg(daily):
    s = pd.Series(daily)
    cum = float((1 + s).prod()) - 1
    shp = float(s.mean() / s.std() * np.sqrt(252)) if s.std() > 0 else 0.0
    return cum, shp


def replay(cache_dir: Path) -> pd.DataFrame:
    meta = json.loads((cache_dir / "meta.json").read_text())
    rows = []
    for fm in meta:
        td = pd.read_parquet(cache_dir / f"fold_{fm['fold']:02d}.parquet")
        td["date"] = pd.to_datetime(td["date"])
        ser = {"gross": []}
        ser.update({f"net{b}": [] for b in BPS_TIERS})
        tr = []
        for _, day in td.groupby("date"):
            day = day[day["close"] >= PRICE_FLOOR]
            if len(day) < TOP_K:
                continue
            r = day.sort_values("pred", ascending=False).head(TOP_K)[V2_TARGET].to_numpy()
            r = r[np.isfinite(r)]
            if len(r) == 0:
                continue
            ser["gross"].append(r.mean())
            for b in BPS_TIERS:
                ser[f"net{b}"].append((r - 2 * b / 1e4).mean())
            tr.extend(r.tolist())
        if not ser["gross"]:
            continue
        row = {"fold": fm["fold"], "wr": float(np.mean([t > 0 for t in tr]))}
        row["gross"], row["gross_sharpe"] = agg(ser["gross"])
        for b in BPS_TIERS:
            row[f"net{b}"], _ = agg(ser[f"net{b}"])
        rows.append(row)
    return pd.DataFrame(rows)


def report(tag: str, df: pd.DataFrame) -> None:
    pg = mc.one_sample_bootstrap(df["gross"].values)
    p5 = mc.one_sample_bootstrap(df["net5"].values)
    p10 = mc.one_sample_bootstrap(df["net10"].values)
    f5 = " CI>0" if p5["ci_lo"] > 0 else ""
    f10 = " CI>0" if p10["ci_lo"] > 0 else ""
    print(f"  {tag:16s} gross {df['gross'].mean():+6.1%}(Sh{df['gross_sharpe'].mean():.2f},"
          f"WR{df['wr'].mean():.0%}) | net5 {df['net5'].mean():+6.1%}[{p5['ci_lo']:+.1%},"
          f"{p5['ci_hi']:+.1%}]{f5} | net10 {df['net10'].mean():+6.1%}{f10} | "
          f"net30 {df['net30'].mean():+6.1%}  (+folds g{int((df['gross']>0).sum())}/{len(df)})")
    _ = pg


def main() -> None:
    t0 = time.time()
    feats = pd.read_parquet(DATA / "features_largecap.parquet")
    feats["date"] = pd.to_datetime(feats["date"])
    ohlcv = pd.read_parquet(DATA / "ohlcv_largecap.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    oh = ohlcv[ohlcv["ticker"] != "SPY"].sort_values(["ticker", "date"]).copy()
    g = oh.groupby("ticker")
    oh["overnight"] = g["open"].shift(-1) / oh["close"] - 1          # ideal: close[D]->open[D+1]
    # executable: features@D -> buy close[D+1] -> sell open[D+2] == overnight shifted -1
    oh["overnight_exec"] = oh.groupby("ticker")["overnight"].shift(-1)
    feats = feats.merge(oh[["date", "ticker", "overnight_exec"]], on=["date", "ticker"], how="left")
    feats["overnight_exec"] = feats["overnight_exec"].clip(-RET_CLIP, RET_CLIP)

    print("\n  Executability: overnight edge under a 1-day feature lag "
          "(buy NEXT close, sell its open)\n")

    # ideal (already cached from backtest_overnight.py) for side-by-side
    ideal_cache = CACHE_ROOT / "lc_overnight"
    if ideal_cache.exists():
        report("ideal (no-lag)", replay(ideal_cache))

    f2 = feats.copy()
    f2[V2_TARGET] = f2["overnight_exec"]
    cache_dir = CACHE_ROOT / "lc_overnight_exec"
    run_walkforward(
        f2, ohlcv, BASE, config_name="lc_overnight_exec", cache_dir=cache_dir,
        lgb_params=PROD_PARAMS, objective_lambdarank=True, num_boost_round=250,
        verbose=False,
    )
    report("exec (1d-lag)", replay(cache_dir))
    print(f"\n  If exec ~ ideal -> executable with current ~1 AM infra (add MOC leg).  "
          f"Runtime {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
