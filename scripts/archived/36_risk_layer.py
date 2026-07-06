"""V4 experiment: risk / position-sizing layer (no retrain, cache replay).

Round-2 showed idio_vol_60d / pct_from_52w_high carry real DOWNSIDE info (they cut
drawdown) but HURT as ranking features. The hypothesis here: their value is in
RISK MANAGEMENT, not selection. Two overlays on the deployed predictions
(baseline cache v4_nometa), both leak-free (features known at T) and MC-validated:

  (A) Downside filter — from the tradable candidates, drop the most crash-prone
      (top-q idio_vol_60d, or the names farthest below their 52w high) BEFORE the
      top-K cut. Reuses the overlay_col/overlay_exclude_q hook in _evaluate_fold.

  (B) Vol-targeting sizing — weight the K picks proportional to 1/idio_vol instead
      of equal weight (dedicated evaluator; the harness is equal-weight only).

Verdict via 34_mc_validate paired bootstrap vs the equal-weight baseline. The bar:
improve WR / Sharpe / drawdown without giving back return, and survive the MC CI.

Usage:
    PYTHONPATH=src python scripts/v3/36_risk_layer.py
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
TOP_K = 8


def evaluate_sized(cache_dir, ohlcv, extra, mp, ma, cb, policy, k, weight="equal") -> pd.DataFrame:
    """Replay cache with position weighting (equal or inverse-idio-vol)."""
    meta = json.loads((Path(cache_dir) / "meta.json").read_text())
    rows = []
    for fm in meta:
        td = pd.read_parquet(Path(cache_dir) / f"fold_{fm['fold']:02d}.parquet")
        td["date"] = pd.to_datetime(td["date"])
        td = td.merge(extra, on=["date", "ticker"], how="left")
        dates = sorted(td.date.unique())
        port, trades = [], []
        for d in dates[::REBALANCE_EVERY]:
            day = td[td.date == d]
            day = day[tradable_mask(day, mp, ma)]
            if len(day) < k:
                continue
            picks = day.sort_values("pred", ascending=False).head(k)
            rets, ws = [], []
            for _, row in picks.iterrows():
                t_oh = ohlcv[(ohlcv.ticker == row["ticker"]) & (ohlcv.date >= d)]
                if len(t_oh) < 2:
                    continue
                prices = t_oh.head(HOLD_DAYS + 1)["close"].values
                atr = float(row.get("atr_pct_20d", 0.03))
                if not np.isfinite(atr) or atr <= 0:
                    atr = 0.03
                r = _simulate_trade(prices, atr, policy) - 2 * cb / 1e4
                iv = row.get("idio_vol_60d", np.nan)
                w = (1.0 / iv if (weight == "invvol" and np.isfinite(iv) and iv > 0)
                     else 1.0 if weight == "equal" else np.nan)
                rets.append(r)
                ws.append(w)
            rets, ws = np.array(rets), np.array(ws)
            m = np.isfinite(rets) & np.isfinite(ws)
            rets, ws = rets[m], ws[m]
            if len(rets) == 0 or ws.sum() <= 0:
                continue
            ws = ws / ws.sum()
            port.append(float((ws * rets).sum()))
            trades.extend(rets.tolist())
        if not port:
            rows.append({"fold": fm["fold"], "total_return": 0.0, "sharpe": 0.0,
                         "win_rate": 0.0, "max_dd": 0.0, "n_trades": 0})
            continue
        streams = [port[c::N_COHORTS] for c in range(N_COHORTS)]
        stream_cum = [float((1 + pd.Series(s)).prod()) for s in streams if s]
        eff = pd.Series([r / N_COHORTS for r in port])
        cum = (1 + eff).cumprod()
        sharpe = (float((eff.mean() / eff.std()) * np.sqrt(252 / REBALANCE_EVERY))
                  if eff.std() > 0 else 0.0)
        rows.append({
            "fold": fm["fold"],
            "total_return": float(np.mean(stream_cum)) - 1 if stream_cum else 0.0,
            "sharpe": sharpe,
            "win_rate": float(np.mean([t > 0 for t in trades])),
            "max_dd": float(((cum - cum.cummax()) / cum.cummax()).min()),
            "n_trades": len(trades),
        })
    return pd.DataFrame(rows)


def line(df: pd.DataFrame, base: pd.DataFrame, label: str) -> None:
    wr = (df["win_rate"] * df["n_trades"]).sum() / max(df["n_trades"].sum(), 1)
    pb = mc.paired_bootstrap(df["total_return"].values, base["total_return"].values)
    pos = int((df["total_return"] > 0).sum())
    flag = "  <-CI>0" if pb["ci_lo"] > 0 else ""
    print(f"  {label:26s} ret {df['total_return'].mean():+6.1%}  WR {wr:4.0%}  "
          f"Sharpe {df['sharpe'].mean():+5.2f}  worstDD {df['max_dd'].min():+6.1%}  "
          f"+f {pos}/{len(df)}  diff {pb['mean_diff']:+5.1%} "
          f"[{pb['ci_lo']:+.1%},{pb['ci_hi']:+.1%}]{flag}")


def main() -> None:
    t0 = time.time()
    decision = json.loads(DECISION_FP.read_text())
    mp, ma, cb = decision["min_price"], decision["min_adv_usd"], decision["cost_bps"]
    pol = mc.policy_from(decision["exit_policy_adaptive"])
    ohlcv = pd.read_parquet(ROOT / "data/processed/ohlcv_smallcap.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    fe = pd.read_parquet(ROOT / "data/processed/features_smallcap.parquet",
                         columns=["date", "ticker", "idio_vol_60d", "pct_from_52w_high"])
    fe["date"] = pd.to_datetime(fe["date"])
    fe["neg_52w"] = -fe["pct_from_52w_high"]
    extra = fe[["date", "ticker", "idio_vol_60d", "neg_52w"]]

    base = evaluate_sized(BASE_CACHE, ohlcv, extra, mp, ma, cb, pol, TOP_K, "equal")
    print(f"\n  Risk layer vs equal-weight baseline (adaptive6_pt40, top-{TOP_K}, {cb:g}bps)\n")
    line(base, base, "baseline (equal-weight)")

    print("\n  (A) downside FILTER (drop crash-prone before top-K):")
    for q in (0.1, 0.2, 0.3):
        df = mc.replay_per_fold(BASE_CACHE, ohlcv, mp, ma, cb, pol, TOP_K, extra=extra,
                                overlay_col="idio_vol_60d", overlay_exclude_q=q)
        line(df, base, f"drop top-{q:.0%} idio_vol")
    for q in (0.1, 0.2, 0.3):
        df = mc.replay_per_fold(BASE_CACHE, ohlcv, mp, ma, cb, pol, TOP_K, extra=extra,
                                overlay_col="neg_52w", overlay_exclude_q=q)
        line(df, base, f"drop bottom-{q:.0%} 52w-high")

    print("\n  (B) vol-targeting SIZING (weight ~ 1/idio_vol):")
    inv = evaluate_sized(BASE_CACHE, ohlcv, extra, mp, ma, cb, pol, TOP_K, "invvol")
    line(inv, base, "inverse-vol weighted")

    print("\n  (CI>0 = improvement survives the MC paired bootstrap.)")
    print(f"  Runtime: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
