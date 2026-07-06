"""LIQUIDCAP — execution study: enter at signal-day CLOSE vs next-day OPEN.

The frozen backtest (+1.69%/mo flat 5bps) enters each pick at the CLOSE of the
signal day (harness `prices[0] = close_reb`). But the signal for date T uses
features that embed T's close, so a close-T fill is the theoretical ceiling
(mildly look-ahead: you need the close to compute the pick, then trade at that
same close). The realistic, look-ahead-free fill is the NEXT session's OPEN
(decide after T's close, submit overnight, fill at T+1 open) — exactly what the
live morning job does, and executable on IBKR via a MOO order.

This re-evaluates the SAME purged walk-forward picks (cached GI_fs15_fund_20d
predictions — no retraining) under both entry conventions and measures the gap:
how much of the edge survives waiting for the open. Same top-8, hold 20d, ATR
trail + pt40 exits, tradable gate, cohort compounding, both cost models. Paired
bootstrap on the per-fold return difference.

Usage:
    PYTHONPATH=src python scripts/liquidcap/exp_entry_timing.py
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

import _v3_harness as h  # noqa: E402

_spec = importlib.util.spec_from_file_location("mc", ROOT / "scripts/v3/34_mc_validate.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

DATA = ROOT / "data" / "liquidcap"
CACHE = DATA / "cache" / "GI_fs15_fund_20d"
COST_BPS = 5.0
MIN_PRICE, MIN_ADV = 1.5, 500_000

# Match the frozen GI spec globals used by the harness evaluator.
h.V2_TARGET = "fwd_ret_20d_sector_rel"
h.HOLD_DAYS = 20
h.REBALANCE_EVERY = 5
h.N_COHORTS = h.HOLD_DAYS // h.REBALANCE_EVERY  # 4
POLICY = h.ExitPolicy(profit_target=0.40, time_stop=h.HOLD_DAYS)


def build_ohlcv_index(ohlcv: pd.DataFrame) -> dict:
    """{ticker: (dates[np.datetime64], open[float], close[float])} sorted by date."""
    idx = {}
    for t, g in ohlcv.sort_values("date").groupby("ticker"):
        idx[t] = (g["date"].values.astype("datetime64[ns]"),
                  g["open"].values.astype(float), g["close"].values.astype(float))
    return idx


def price_path(oh_idx: dict, ticker: str, reb_date: pd.Timestamp,
               hold: int, mode: str) -> np.ndarray | None:
    """Entry + hold price path. CLOSE: [close_T, close_T+1, ...]. OPEN: entry at
    the next session's open, then that session's closes: [open_T+1, close_T+1, ...]."""
    arr = oh_idx.get(ticker)
    if arr is None:
        return None
    dates, opens, closes = arr
    i = int(np.searchsorted(dates, np.datetime64(reb_date)))
    if i >= len(dates) or dates[i] != np.datetime64(reb_date):
        return None
    if mode == "close":
        path = closes[i:i + hold + 1]
        return path if len(path) >= 2 else None
    # open: need the NEXT session to enter at its open
    if i + 1 >= len(dates):
        return None
    entry = opens[i + 1]
    tail = closes[i + 1:i + 1 + hold]
    if not np.isfinite(entry) or len(tail) < 1:
        return None
    return np.concatenate([[entry], tail])


def eval_fold(td: pd.DataFrame, oh_idx: dict, policy, cost_bps: float,
              spread: bool, mode: str) -> dict:
    """Mirror of h._evaluate_fold's trade loop, parametrised by entry `mode`."""
    dates = sorted(td.date.unique())
    reb_dates = dates[::h.REBALANCE_EVERY]
    port_returns, trades = [], []
    for reb in reb_dates:
        day = td[td.date == reb]
        day = day[h.tradable_mask(day, MIN_PRICE, MIN_ADV)]
        if len(day) < h.TOP_K:
            continue
        picks = day.sort_values("pred", ascending=False).head(h.TOP_K)
        period = []
        for _, row in picks.iterrows():
            prices = price_path(oh_idx, row["ticker"], reb, h.HOLD_DAYS, mode)
            if prices is None:
                continue
            atr = float(row.get("atr_pct_20d", 0.03))
            if not np.isfinite(atr) or atr <= 0:
                atr = 0.03
            ret = h._simulate_trade(prices, atr, policy)
            cost_rt = 2 * cost_bps / 1e4
            if spread:
                sp = row.get("cs_spread_20d", np.nan)
                if np.isfinite(sp) and sp > 0:
                    cost_rt = max(cost_rt, min(float(sp), 0.10))
            ret -= cost_rt
            trades.append(ret)
            period.append(ret)
        if period:
            port_returns.append(float(np.mean(period)))

    if not port_returns:
        return {"total_return": 0.0, "sharpe": 0.0, "win_rate": 0.0}
    streams = [port_returns[c::h.N_COHORTS] for c in range(h.N_COHORTS)]
    stream_cum = [float((1 + pd.Series(s)).prod()) for s in streams if s]
    total = float(np.mean(stream_cum)) - 1 if stream_cum else 0.0
    eff = pd.Series([r / h.N_COHORTS for r in port_returns])
    sharpe = float((eff.mean() / eff.std()) * np.sqrt(252 / h.REBALANCE_EVERY)) if eff.std() > 0 else 0.0
    wr = float(np.mean([t > 0 for t in trades])) if trades else 0.0
    return {"total_return": total, "sharpe": sharpe, "win_rate": wr}


def summarize(label: str, df: pd.DataFrame) -> None:
    r = df["total_return"].mean()
    mo = (1 + r) ** (1 / 3) - 1
    print(f"  {label:22} {r:+7.2%}/f  {mo:+6.2%}/mo  Sh{df['sharpe'].mean():5.2f}  "
          f"WR{df['win_rate'].mean():4.0%}")


def main() -> None:
    t0 = time.time()
    meta = json.loads((CACHE / "meta.json").read_text())
    ohlcv = pd.read_parquet(DATA / "ohlcv_sp500.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    ohlcv = ohlcv[ohlcv.ticker != "SPY"]
    oh_idx = build_ohlcv_index(ohlcv)
    print(f"  {len(meta)} folds | {len(oh_idx)} tickers | indexed in "
          f"{time.time() - t0:.0f}s\n")

    for cost_label, spread in [("flat 5bps", False), ("spread-aware", True)]:
        rows_close, rows_open, parity = [], [], []
        for fm in meta:
            td = pd.read_parquet(CACHE / f"fold_{fm['fold']:02d}.parquet")
            td["date"] = pd.to_datetime(td["date"])
            bounds = (pd.Timestamp(fm["test_start"]), pd.Timestamp(fm["test_end"]))
            # Parity check: my close-entry vs the harness's own evaluator.
            ref = h._evaluate_fold(td, ohlcv, bounds, MIN_PRICE, MIN_ADV,
                                   POLICY, COST_BPS, spread)
            c = eval_fold(td, oh_idx, POLICY, COST_BPS, spread, "close")
            o = eval_fold(td, oh_idx, POLICY, COST_BPS, spread, "open")
            parity.append(abs(c["total_return"] - ref["total_return"]))
            rows_close.append({**c, "fold": fm["fold"]})
            rows_open.append({**o, "fold": fm["fold"]})
        dc, do = pd.DataFrame(rows_close), pd.DataFrame(rows_open)

        print(f"  === cost {cost_label} ===")
        print(f"  (parity |mine-harness| max={max(parity):.2e} — close entry "
              f"reproduces the harness)")
        summarize("CLOSE (T close)", dc)
        summarize("OPEN  (T+1 open)", do)
        pb = mc.paired_bootstrap(do["total_return"].values, dc["total_return"].values)
        d_mo_close = (1 + dc["total_return"].mean()) ** (1 / 3) - 1
        d_mo_open = (1 + do["total_return"].mean()) ** (1 / 3) - 1
        print(f"  Δ(open-close): {pb['mean_diff']:+.2%}/f "
              f"[{pb['ci_lo']:+.2%},{pb['ci_hi']:+.2%}]  "
              f"p(open≤close)={pb['p_le0']:.2f}  "
              f"| monthly {d_mo_close:+.2%} -> {d_mo_open:+.2%}\n")

    print(f"  Runtime {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
