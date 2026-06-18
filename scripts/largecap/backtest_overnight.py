"""Large-cap short-horizon (overnight / intraday) viability — net of REAL spreads.

The thesis: on small-caps the overnight signal was real GROSS (Sharpe ~2.87) but
died on ~2%/day round-trip spreads. Large-caps ($2B-$50B) have spreads ~1-10bps,
so the question is whether the same daily-rebalanced edge SURVIVES net here.

Two PIT-safe horizons (decide on features as-of close D):
  overnight : buy close[D],  sell open[D+1]   -> open[D+1]/close[D] - 1
  intraday  : buy open[D+1], sell close[D+1]  -> close[D+1]/open[D+1] - 1

Train a LambdaRank on each forward return (26 base features, no EDGAR), then
DAILY-rebalance the top-8 (1-day hold) and report gross + net at 5/10/30 bps and
net@measured-spread (cs_spread_20d). MC one-sample bootstrap on per-fold returns:
gross CI>0 = signal exists; net CI>0 = survives costs.

Usage:
    PYTHONPATH=src python scripts/largecap/backtest_overnight.py
"""
from __future__ import annotations

import importlib.util
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
BASE = V2_FEATURES_BASE                      # 26 base, no EDGAR for this universe
TOP_K = 8
PRICE_FLOOR = 3.0                            # drop adjusted-price glitches (sub-$1 ratios explode)
RET_CLIP = 0.25                              # winsorize target (99.99th pctile ~+39%)
BPS_TIERS = [5, 10, 30]                      # round-trip cost scenarios
PROD_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg",
    "num_leaves": 31, "max_depth": 6, "learning_rate": 0.05,
    "min_child_samples": 30, "subsample": 0.75, "colsample_bytree": 0.7,
    "reg_lambda": 5.0, "n_jobs": 1, "seed": 42, "verbose": -1,
}


def short_targets(ohlcv: pd.DataFrame) -> pd.DataFrame:
    oh = ohlcv[ohlcv["ticker"] != "SPY"].sort_values(["ticker", "date"]).copy()
    g = oh.groupby("ticker")
    nxt_open, nxt_close = g["open"].shift(-1), g["close"].shift(-1)
    oh["overnight"] = nxt_open / oh["close"] - 1
    oh["intraday"] = nxt_close / nxt_open - 1
    return oh[["date", "ticker", "overnight", "intraday"]]


def agg(daily: list[float]) -> tuple[float, float]:
    s = pd.Series(daily)
    cum = float((1 + s).prod()) - 1
    shp = float(s.mean() / s.std() * np.sqrt(252)) if s.std() > 0 else 0.0
    return cum, shp


def daily_replay(cache_dir: Path) -> pd.DataFrame:
    """Daily top-8, 1-day hold. Per-fold cum return: gross, net@bps tiers, net@spread."""
    import json
    meta = json.loads((cache_dir / "meta.json").read_text())
    rows = []
    for fm in meta:
        td = pd.read_parquet(cache_dir / f"fold_{fm['fold']:02d}.parquet")
        td["date"] = pd.to_datetime(td["date"])
        series: dict[str, list] = {"gross": [], "spread": []}
        for b in BPS_TIERS:
            series[f"net{b}"] = []
        trades_sp = []
        for _, day in td.groupby("date"):
            day = day[day["close"] >= PRICE_FLOOR]   # drop adjusted-price glitches
            if len(day) < TOP_K:
                continue
            picks = day.sort_values("pred", ascending=False).head(TOP_K)
            r = picks[V2_TARGET].to_numpy()
            keep = np.isfinite(r)
            r = r[keep]
            if len(r) == 0:
                continue
            sp = picks["cs_spread_20d"].to_numpy()[keep]
            sp = np.where(np.isfinite(sp) & (sp > 0), np.minimum(sp, 0.10), 0.0)
            series["gross"].append(r.mean())
            for b in BPS_TIERS:
                series[f"net{b}"].append((r - 2 * b / 1e4).mean())
            series["spread"].append((r - sp).mean())
            trades_sp.extend((r - sp).tolist())
        if not series["gross"]:
            continue
        row = {"fold": fm["fold"], "n": len(trades_sp),
               "wr_spread": float(np.mean([t > 0 for t in trades_sp]))}
        row["gross"], row["gross_sharpe"] = agg(series["gross"])
        for b in BPS_TIERS:
            row[f"net{b}"], row[f"net{b}_sharpe"] = agg(series[f"net{b}"])
        row["spread"], row["spread_sharpe"] = agg(series["spread"])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    feats = pd.read_parquet(DATA / "features_largecap.parquet")
    feats["date"] = pd.to_datetime(feats["date"])
    ohlcv = pd.read_parquet(DATA / "ohlcv_largecap.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    st = short_targets(ohlcv)
    feats = feats.merge(st, on=["date", "ticker"], how="left")
    # Winsorize targets to a tradeable bound. Adjusted-price artifacts on
    # distressed/reverse-split names (close ~$1e-6 -> +485999x) explode daily
    # compounding; >25% overnight on a large-cap is a glitch or an untradeable
    # news gap (99.99th pctile is ~+39%). Clipping cleans both train + replay
    # since the cache stores this same V2_TARGET column.
    for c in ("overnight", "intraday"):
        feats[c] = feats[c].clip(-RET_CLIP, RET_CLIP)

    med_sp = feats["cs_spread_20d"].replace([np.inf, -np.inf], np.nan).median()
    print(f"\n  Large-cap day-trading — daily top-8, 1-day hold "
          f"(universe {feats['ticker'].nunique()} names, median spread {med_sp:.2%})\n")
    hdr = (f"  {'horizon':9s} {'gross':>7} {'grSh':>5} " +
           " ".join(f"{'net'+str(b):>7}" for b in BPS_TIERS) +
           f" {'netSpr':>7} {'WRspr':>6} {'verdict':>22}")
    print(hdr)
    for name in ("overnight", "intraday"):
        f2 = feats.copy()
        f2[V2_TARGET] = f2[name]
        cache_dir = CACHE_ROOT / f"lc_{name}"
        run_walkforward(
            f2, ohlcv, BASE, config_name=f"lc_{name}", cache_dir=cache_dir,
            lgb_params=PROD_PARAMS, objective_lambdarank=True, num_boost_round=250,
            verbose=False,
        )
        df = daily_replay(cache_dir)
        pg = mc.one_sample_bootstrap(df["gross"].values)
        psp = mc.one_sample_bootstrap(df["spread"].values)
        p30 = mc.one_sample_bootstrap(df["net30"].values)
        verdict = ("net edge @spread (CI>0)" if psp["ci_lo"] > 0
                   else "net@30bps (CI>0)" if p30["ci_lo"] > 0
                   else "gross only, costs kill" if pg["ci_lo"] > 0
                   else "no edge")
        wr = (df["wr_spread"] * df["n"]).sum() / df["n"].sum()
        nets = " ".join(f"{df['net'+str(b)].mean():+7.1%}" for b in BPS_TIERS)
        print(f"  {name:9s} {df['gross'].mean():+7.1%} {df['gross_sharpe'].mean():+5.2f} "
              f"{nets} {df['spread'].mean():+7.1%} {wr:6.0%} {verdict:>22}")

    print(f"\n  gross CI>0 = signal exists; netSpr CI>0 = survives real cost.  "
          f"Runtime {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
