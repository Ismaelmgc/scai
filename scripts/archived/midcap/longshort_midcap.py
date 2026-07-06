"""#5+#6 synthesis: market-neutral LONG-SHORT on mid-caps ($2-10B), where short
borrow IS feasible (unlike small-caps, which blocked #5).

Replays the midcap_swing prediction cache (no retrain): each rebalance, long the
top-K and short the bottom-K by model score, held 20d. The return is the spread
top − bottom of the (date-relative) 20d target, so the market/date mean cancels →
market-neutral by construction. Net of 2-leg round-trip cost (15bps+spread each)
+ short borrow (~2%/yr general-collateral for liquid mid-caps). Cohort-compounded
per fold like the harness; MC vs zero.

A market-neutral L/S is also less survivorship-distorted than long-only: excluding
fallen names removes both long-leg inflation and the short-leg gain we'd have made
shorting them, so the spread is more robust than the long-only magnitude.

Usage:
    PYTHONPATH=src python scripts/midcap/longshort_midcap.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "v3"))
from _v3_harness import V2_TARGET  # noqa: E402

_spec = importlib.util.spec_from_file_location("mc", ROOT / "scripts/v3/34_mc_validate.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

CACHE = ROOT / "data" / "midcap" / "cache" / "midcap_swing"
DEC = json.loads((ROOT / "data/v3_benchmarks/v4_filter_decision.json").read_text())
MP, MA, CB = DEC["min_price"], DEC["min_adv_usd"], DEC["cost_bps"]
K, REB, N_COH = 8, 5, 4
BORROW_APR = 0.02   # general-collateral short borrow for liquid mid-caps
BORROW_20D = BORROW_APR * 20 / 252


def leg_cost(spreads):
    rt = 2 * CB / 1e4
    sp = np.nanmean(spreads) if len(spreads) else 0.0
    return max(rt, min(sp, 0.10)) if np.isfinite(sp) and sp > 0 else rt


def main():
    meta = json.loads((CACHE / "meta.json").read_text())
    rows = []
    for fm in meta:
        td = pd.read_parquet(CACHE / f"fold_{fm['fold']:02d}.parquet")
        td["date"] = pd.to_datetime(td["date"])
        td = td[(td["close"] >= MP) & (td["adv_usd_20d"] >= MA)].dropna(subset=[V2_TARGET])
        dates = sorted(td["date"].unique())[::REB]
        gross, net = [], []
        for d in dates:
            day = td[td["date"] == d]
            if len(day) < 2 * K:
                continue
            s = day.sort_values("pred", ascending=False)
            top, bot = s.head(K), s.tail(K)
            ls = top[V2_TARGET].mean() - bot[V2_TARGET].mean()
            cost = (leg_cost(top["cs_spread_20d"].values)
                    + leg_cost(bot["cs_spread_20d"].values) + BORROW_20D)
            gross.append(ls)
            net.append(ls - cost)
        if not gross:
            continue

        def comp(x):
            streams = [pd.Series(x[c::N_COH]) for c in range(N_COH)]
            cum = [float((1 + s).prod()) for s in streams if len(s)]
            return float(np.mean(cum)) - 1 if cum else 0.0
        eff = pd.Series([r / N_COH for r in net])
        sharpe = float(eff.mean() / eff.std() * np.sqrt(252 / REB)) if eff.std() > 0 else 0.0
        rows.append({"fold": fm["fold"], "gross": comp(gross), "net": comp(net),
                     "sharpe": sharpe, "wr": float(np.mean([n > 0 for n in net]))})
    df = pd.DataFrame(rows)
    pg = mc.one_sample_bootstrap(df["gross"].values)
    pn = mc.one_sample_bootstrap(df["net"].values)
    mo = (1 + df["net"].mean()) ** (1 / 3) - 1
    print(f"\n  Mid-cap market-neutral L/S (top-{K} long / bottom-{K} short, 20d, "
          f"{CB:g}bps/leg + {BORROW_APR:.0%} borrow)\n")
    print(f"  gross/fold : {df['gross'].mean():+.1%}  CI[{pg['ci_lo']:+.1%},{pg['ci_hi']:+.1%}]"
          f"{' CI>0' if pg['ci_lo'] > 0 else ''}")
    print(f"  net/fold   : {df['net'].mean():+.1%}  ({mo:+.1%}/mo)  "
          f"CI[{pn['ci_lo']:+.1%},{pn['ci_hi']:+.1%}]{' CI>0' if pn['ci_lo'] > 0 else ''}")
    print(f"  Sharpe     : {df['sharpe'].mean():.2f}   WR : {(df['wr']).mean():.0%}   "
          f"+folds {int((df['net'] > 0).sum())}/{len(df)}")
    print("\n  market-neutral by construction (date-mean cancels in top-bottom).")


if __name__ == "__main__":
    main()
