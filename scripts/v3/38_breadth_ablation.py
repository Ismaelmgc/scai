"""V4 experiment: breadth ablation (does the candidate-pool size bind?).

The baseline already picks top-8 from ~526 tradable candidates/rebalance (the top
2%). We cannot cheaply ADD breadth (no OHLCV for new names), but we CAN remove it:
at each rebalance, randomly subsample the tradable candidates to a cap N before
the top-8 cut, and trace return/WR/Sharpe vs N. If performance is already FLAT
from N~150 up to the full pool, we are in the diminishing-returns regime and
adding MORE breadth would not help — answering the breadth question with data we
already have, no download. If it is still rising at the full pool, breadth binds.

Replay-only (cache, adaptive6_pt40), MC paired bootstrap vs the full pool.

Usage:
    PYTHONPATH=src python scripts/v3/38_breadth_ablation.py
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
    HOLD_DAYS,
    N_COHORTS,
    REBALANCE_EVERY,
    _simulate_trade,
)

from app.features.tradability import tradable_mask  # noqa: E402

_spec = importlib.util.spec_from_file_location("mc", ROOT / "scripts/v3/34_mc_validate.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

DECISION_FP = ROOT / "data" / "v3_benchmarks" / "v4_filter_decision.json"
BASE_CACHE = ROOT / "data" / "v3_benchmarks" / "cache" / "v4_nometa"
CAPS = [50, 100, 150, 200, 300, 10_000]   # 10_000 = full pool (no cap)
TOP_K = 8
SEED = 42


def evaluate(cap, ohlcv, mp, ma, cb, policy) -> pd.DataFrame:
    meta = json.loads((BASE_CACHE / "meta.json").read_text())
    rng = np.random.default_rng(SEED)
    rows = []
    for fm in meta:
        td = pd.read_parquet(BASE_CACHE / f"fold_{fm['fold']:02d}.parquet")
        td["date"] = pd.to_datetime(td["date"])
        dates = sorted(td.date.unique())
        port, trades = [], []
        for d in dates[::REBALANCE_EVERY]:
            day = td[td.date == d]
            day = day[tradable_mask(day, mp, ma)]
            if len(day) > cap:                       # subsample down to the cap
                day = day.iloc[rng.permutation(len(day))[:cap]]
            if len(day) < TOP_K:
                continue
            picks = day.sort_values("pred", ascending=False).head(TOP_K)
            rets = []
            for _, row in picks.iterrows():
                t_oh = ohlcv[(ohlcv.ticker == row["ticker"]) & (ohlcv.date >= d)]
                if len(t_oh) < 2:
                    continue
                prices = t_oh.head(HOLD_DAYS + 1)["close"].values
                atr = float(row.get("atr_pct_20d", 0.03))
                if not np.isfinite(atr) or atr <= 0:
                    atr = 0.03
                rets.append(_simulate_trade(prices, atr, policy) - 2 * cb / 1e4)
            if rets:
                port.append(float(np.mean(rets)))
                trades.extend(rets)
        if not port:
            rows.append({"fold": fm["fold"], "total_return": 0.0, "sharpe": 0.0,
                         "win_rate": 0.0, "n_trades": 0})
            continue
        streams = [port[c::N_COHORTS] for c in range(N_COHORTS)]
        stream_cum = [float((1 + pd.Series(s)).prod()) for s in streams if s]
        eff = pd.Series([r / N_COHORTS for r in port])
        sharpe = (float((eff.mean() / eff.std()) * np.sqrt(252 / REBALANCE_EVERY))
                  if eff.std() > 0 else 0.0)
        rows.append({
            "fold": fm["fold"],
            "total_return": float(np.mean(stream_cum)) - 1 if stream_cum else 0.0,
            "sharpe": sharpe, "win_rate": float(np.mean([t > 0 for t in trades])),
            "n_trades": len(trades),
        })
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    decision = json.loads(DECISION_FP.read_text())
    mp, ma, cb = decision["min_price"], decision["min_adv_usd"], decision["cost_bps"]
    pol = mc.policy_from(decision["exit_policy_adaptive"])
    ohlcv = pd.read_parquet(ROOT / "data/processed/ohlcv_smallcap.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    full = evaluate(10_000, ohlcv, mp, ma, cb, pol)
    print(f"\n  Breadth ablation — subsample candidates to N, then top-8 "
          f"(adaptive6_pt40, {cb:g}bps, seed {SEED})\n")
    print(f"  {'cap N':>6} {'ret':>7} {'WR':>4} {'Sharpe':>6} {'+folds':>6} "
          f"{'vs full (diff [95% CI])':>28}")
    for cap in CAPS:
        df = full if cap == 10_000 else evaluate(cap, ohlcv, mp, ma, cb, pol)
        wr = (df["win_rate"] * df["n_trades"]).sum() / df["n_trades"].sum()
        pos = int((df["total_return"] > 0).sum())
        if cap == 10_000:
            tag = "(full pool = baseline)"
        else:
            pb = mc.paired_bootstrap(df["total_return"].values, full["total_return"].values)
            tag = f"{pb['mean_diff']:+.1%} [{pb['ci_lo']:+.1%},{pb['ci_hi']:+.1%}]"
        label = "full" if cap == 10_000 else str(cap)
        print(f"  {label:>6} {df['total_return'].mean():+7.1%} {wr:4.0%} "
              f"{df['sharpe'].mean():+6.2f} {pos:>4}/16 {tag:>28}")
    print("\n  Flat from N~150 up -> not breadth-starved (more breadth won't help).")
    print(f"  Runtime: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
