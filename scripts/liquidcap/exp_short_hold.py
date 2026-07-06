"""LIQUIDCAP — short-hold study: buy and sell within a day or few (the "buy the
close, sell tomorrow" idea).

The frozen book holds 20 days. But the top-8 names sometimes pop right after
selection, so: does the SAME ranking (trained on 20d fwd returns) carry enough
1-3 day power to trade a fast in-and-out? The catch is turnover — a 1-day hold
rebalances DAILY, paying round-trip cost every day (~20x the 20d strategy). So
we measure GROSS (0 cost) to see raw signal power, then NET at flat 5bps and
spread-aware, under three realistic fills.

Reuses the cached GI_fs15_fund_20d purged predictions (33 folds, no retrain).
Pure N-day hold (no trailing stop / profit target — you just sell after N days).
Daily rebalance, top-8 equal weight, cohort compounding (N_COHORTS = hold).

Fills per pick selected on date T:
  close   : buy close_T,   sell close_(T+N)     (literal "buy the close"; mild
            look-ahead — the pick needs T's close)
  oc      : buy open_(T+1), sell close_(T+N)     (realistic; next-day intraday for N=1)
  oo      : buy open_(T+1), sell open_(T+1+N)    (realistic; full N-day hold)

Usage:
    PYTHONPATH=src python scripts/liquidcap/exp_short_hold.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "v3"))

import _v3_harness as h  # noqa: E402

DATA = ROOT / "data" / "liquidcap"
CACHE = DATA / "cache" / "GI_fs15_fund_20d"
MIN_PRICE, MIN_ADV, TOP_K = 1.5, 500_000, 8


def build_index(ohlcv: pd.DataFrame) -> dict:
    idx = {}
    for t, g in ohlcv.sort_values("date").groupby("ticker"):
        idx[t] = (g["date"].values.astype("datetime64[ns]"),
                  g["open"].values.astype(float), g["close"].values.astype(float))
    return idx


def trade_ret(arr, reb, hold: int, mode: str) -> float | None:
    dates, opens, closes = arr
    i = int(np.searchsorted(dates, np.datetime64(reb)))
    if i >= len(dates) or dates[i] != np.datetime64(reb):
        return None
    if mode == "close":
        if i + hold >= len(dates):
            return None
        entry, exit_ = closes[i], closes[i + hold]
    elif mode == "oc":
        if i + 1 >= len(dates) or i + hold >= len(dates):
            return None
        entry, exit_ = opens[i + 1], closes[i + hold]
    else:  # oo
        if i + 1 + hold >= len(dates):
            return None
        entry, exit_ = opens[i + 1], opens[i + 1 + hold]
    if not (np.isfinite(entry) and np.isfinite(exit_)) or entry <= 0:
        return None
    return exit_ / entry - 1.0


def eval_fold(td, oh_idx, hold, mode, cost_bps, spread) -> dict:
    dates = sorted(td.date.unique())
    n_coh = max(hold, 1)
    port, trades = [], []
    for reb in dates:
        day = td[td.date == reb]
        day = day[h.tradable_mask(day, MIN_PRICE, MIN_ADV)]
        if len(day) < TOP_K:
            continue
        picks = day.sort_values("pred", ascending=False).head(TOP_K)
        rets = []
        for _, row in picks.iterrows():
            arr = oh_idx.get(row["ticker"])
            if arr is None:
                continue
            r = trade_ret(arr, reb, hold, mode)
            if r is None:
                continue
            cost_rt = 2 * cost_bps / 1e4
            if spread:
                sp = row.get("cs_spread_20d", np.nan)
                if np.isfinite(sp) and sp > 0:
                    cost_rt = max(cost_rt, min(float(sp), 0.10))
            r -= cost_rt
            rets.append(r)
            trades.append(r)
        if rets:
            port.append(float(np.mean(rets)))
    if not port:
        return {"ret": 0.0, "sharpe": 0.0, "wr": 0.0}
    streams = [port[c::n_coh] for c in range(n_coh)]
    stream_cum = [float((1 + pd.Series(s)).prod()) for s in streams if s]
    total = float(np.mean(stream_cum)) - 1 if stream_cum else 0.0
    eff = pd.Series([r / n_coh for r in port])
    sharpe = float((eff.mean() / eff.std()) * np.sqrt(252)) if eff.std() > 0 else 0.0
    wr = float(np.mean([t > 0 for t in trades])) if trades else 0.0
    return {"ret": total, "sharpe": sharpe, "wr": wr}


def run(folds, oh_idx, hold, mode, cost_bps, spread) -> dict:
    rs = [eval_fold(td, oh_idx, hold, mode, cost_bps, spread) for _, td in folds]
    r = float(np.mean([x["ret"] for x in rs]))
    mo = (1 + r) ** (1 / 3) - 1
    return {"mo": mo, "sharpe": float(np.mean([x["sharpe"] for x in rs])),
            "wr": float(np.mean([x["wr"] for x in rs]))}


def main() -> None:
    t0 = time.time()
    meta = json.loads((CACHE / "meta.json").read_text())
    ohlcv = pd.read_parquet(DATA / "ohlcv_sp500.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    ohlcv = ohlcv[ohlcv.ticker != "SPY"]
    oh_idx = build_index(ohlcv)
    folds = []
    for fm in meta:
        td = pd.read_parquet(CACHE / f"fold_{fm['fold']:02d}.parquet")
        td["date"] = pd.to_datetime(td["date"])
        folds.append((fm, td))
    print(f"  {len(folds)} folds | {len(oh_idx)} tickers | loaded {time.time()-t0:.0f}s\n")

    modes = {"close": "buy close_T  -> sell close_T+N",
             "oc":    "buy open_T+1 -> sell close_T+N",
             "oo":    "buy open_T+1 -> sell open_T+1+N"}
    costs = [("gross", 0.0, False), ("flat 5bps", 5.0, False), ("spread", 5.0, True)]

    for hold in (1, 2, 3):
        print(f"  ==================  HOLD = {hold} day(s), daily rebalance  ==================")
        print(f"  {'fill':32} {'gross/mo':>9} {'net5/mo':>9} {'spread/mo':>10} {'Sh(net5)':>9} {'WR':>5}")
        for mkey, mdesc in modes.items():
            res = {c[0]: run(folds, oh_idx, hold, mkey, c[1], c[2]) for c in costs}
            print(f"  {mdesc:32} {res['gross']['mo']:+8.2%} {res['flat 5bps']['mo']:+8.2%} "
                  f"{res['spread']['mo']:+9.2%} {res['flat 5bps']['sharpe']:+8.2f} "
                  f"{res['flat 5bps']['wr']:4.0%}")
        print()
    print("  ref 20d strategy: +1.69%/mo flat / +0.95%/mo spread, Sh 2.43")
    print(f"  Runtime {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
