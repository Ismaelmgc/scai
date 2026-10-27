"""Liquidcap backtest through the SAME engine the live book runs (app.engine.cohort_book).

Results depend on the panel snapshot: a fresh yfinance pull re-adjusts past prices for
dividends and drops names that left the index since (2026-10: 22.1% -> 21.0%/yr on the
same predictions), so compare variants on the SAME panel.

Replays the purged walk-forward out-of-sample predictions (harness cache
data/liquidcap/cache/<spec>/fold_XX.parquet, produced by scripts/v3/_v3_harness.py)
session by session with `cohort_book.run`. Local research tool: needs the
gitignored panel data/liquidcap/ohlcv_sp500.parquet and the prediction cache.

    PYTHONPATH=src python scripts/liquidcap/backtest_cohorts.py [--cost-bps 5] [--no-sector-cap]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from app.engine import cohort_book as cb  # noqa: E402

DATA = ROOT / "data" / "liquidcap"


def load(spec: str):
    cache = DATA / "cache" / spec
    meta = json.loads((cache / "meta.json").read_text())
    pred = pd.concat([pd.read_parquet(cache / f"fold_{m['fold']:02d}.parquet") for m in meta],
                     ignore_index=True)
    pred["date"] = pd.to_datetime(pred["date"])
    pred = pred.drop_duplicates(["date", "ticker"], keep="last")
    pred = pred[(pred.close >= 1.5) & (pred.adv_usd_20d >= 500_000)]   # tradability gate
    o = pd.read_parquet(DATA / "ohlcv_sp500.parquet", columns=["date", "ticker", "close"])
    o["date"] = pd.to_datetime(o["date"])
    sectors = json.loads((DATA / "gics_sectors.json").read_text(encoding="utf-8"))
    dates = sorted(set(pred.date) & set(o.date))
    o = o[o.date.isin(dates)]
    closes = {str(d.date()): dict(zip(g.ticker, g.close, strict=True))
              for d, g in o.groupby("date")}
    ranked = {str(d.date()): [cb.Candidate(r.ticker, r.pred, r.atr_pct_20d, sectors.get(r.ticker))
                              for r in g.sort_values("pred", ascending=False).itertuples()]
              for d, g in pred[pred.date.isin(dates)].groupby("date")}
    return [str(d.date()) for d in dates], closes, ranked


def report(dates: list[str], curve: list[float], state: cb.BookState) -> dict:
    e = pd.Series(curve, pd.to_datetime(dates))
    r = e.pct_change().dropna()
    w12 = (e / e.shift(12) - 1).dropna()
    t = pd.DataFrame(state.closed_trades)
    return {"anual%": 100 * (e.iloc[-1] ** (252 / len(r)) - 1),
            "vol%": 100 * r.std() * np.sqrt(252),
            "sharpe": r.mean() / r.std() * np.sqrt(252), "maxDD%": 100 * (e / e.cummax() - 1).min(),
            "peor_12s%": 100 * w12.min(), "trades": len(t), "win%": 100 * (t.pnl_pct > 0).mean(),
            "salidas": t.exit_reason.value_counts().to_dict()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="GI_fs15_fund_20d")
    ap.add_argument("--cost-bps", type=float, default=5.0, help="per side (harness flat = 5)")
    ap.add_argument("--no-sector-cap", action="store_true")
    args = ap.parse_args()
    cfg = replace(cb.BookConfig(), cost_bps=args.cost_bps,
                  max_per_sector=None if args.no_sector_cap else 2)
    dates, closes, ranked = load(args.spec)
    state, curve = cb.run(dates, closes, ranked, 1.0, cfg)
    print(f"{len(dates)} sesiones {dates[0]}..{dates[-1]} · coste {cfg.cost_bps} pb/lado · "
          f"límite sector {cfg.max_per_sector}")
    for k, v in report(dates, curve, state).items():
        print(f"  {k:10s} {v:.2f}" if isinstance(v, float) else f"  {k:10s} {v}")


if __name__ == "__main__":
    main()
