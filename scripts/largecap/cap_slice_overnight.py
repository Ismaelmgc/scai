"""Where does the overnight edge live? Slice the $2B-$50B universe by market-cap
tercile and re-evaluate (replay only, no retrain) the executable overnight model.

Addresses the worry that the edge is driven by the smaller / more-manipulable end
of the band. The whole-universe-trained model's predictions are reused; we just
restrict daily selection to each cap tercile and recompute net returns + spread.
A broad/uniform edge argues against a manipulation story; one concentrated in the
small, wide-spread tercile would be the fragile case.

Usage:
    PYTHONPATH=src python scripts/largecap/cap_slice_overnight.py
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

DATA = ROOT / "data" / "largecap"
CACHE = DATA / "cache" / "lc_overnight_exec"
TOP_K, PRICE_FLOOR = 8, 3.0


def agg(daily):
    s = pd.Series(daily)
    return float((1 + s).prod()) - 1


def replay(caps: dict, lo: float, hi: float):
    """Per-fold net returns restricting selection to tickers with lo<=cap<hi."""
    meta = json.loads((CACHE / "meta.json").read_text())
    g5, g10, wr, spr, ntr = [], [], [], [], 0
    for fm in meta:
        td = pd.read_parquet(CACHE / f"fold_{fm['fold']:02d}.parquet")
        td["date"] = pd.to_datetime(td["date"])
        td["cap"] = td["ticker"].map(caps)
        d5, d10 = [], []
        for _, day in td.groupby("date"):
            day = day[(day["close"] >= PRICE_FLOOR) & (day["cap"] >= lo) & (day["cap"] < hi)]
            if len(day) < TOP_K:
                continue
            picks = day.sort_values("pred", ascending=False).head(TOP_K)
            r = picks[V2_TARGET].to_numpy()
            m = np.isfinite(r)
            r = r[m]
            if len(r) == 0:
                continue
            d5.append((r - 2 * 5 / 1e4).mean())
            d10.append((r - 2 * 10 / 1e4).mean())
            sp = picks["cs_spread_20d"].to_numpy()[m]
            spr.extend(sp[np.isfinite(sp)].tolist())
            wr.extend((r > 0).tolist())
            ntr += len(r)
        if d5:
            g5.append(agg(d5))
            g10.append(agg(d10))
    return np.array(g5), np.array(g10), wr, spr, ntr


def main() -> None:
    uni = pd.read_parquet(DATA / "universe_largecap.parquet")
    caps = dict(zip(uni["ticker"], uni["market_cap"], strict=False))
    q1, q2 = uni["market_cap"].quantile([1 / 3, 2 / 3]).tolist()
    bands = [
        (f"low  $2B-{q1/1e9:.0f}B", 0, q1),
        (f"mid  {q1/1e9:.0f}-{q2/1e9:.0f}B", q1, q2),
        (f"high {q2/1e9:.0f}-50B", q2, 1e15),
        ("ALL  $2B-50B", 0, 1e15),
    ]
    print(f"\n  Overnight (exec, 1d-lag) edge by market-cap tercile  "
          f"(terciles at ${q1/1e9:.1f}B / ${q2/1e9:.1f}B)\n")
    print(f"  {'band':16s} {'net5/fold':>10} {'net5/mo':>8} {'net10/fold':>11} "
          f"{'WR':>5} {'medSpr':>7} {'+folds':>7}  {'net5 95% CI':>20}")
    for name, lo, hi in bands:
        g5, g10, wr, spr, ntr = replay(caps, lo, hi)
        if len(g5) == 0:
            print(f"  {name:16s}  (too few names per day)")
            continue
        p = mc.one_sample_bootstrap(g5)
        mo = (1 + g5.mean()) ** (1 / 3) - 1
        medspr = float(np.median(spr)) if spr else float("nan")
        wrm = float(np.mean(wr)) if wr else float("nan")
        flag = " CI>0" if p["ci_lo"] > 0 else ""
        print(f"  {name:16s} {g5.mean():+10.1%} {mo:+8.1%} {g10.mean():+11.1%} "
              f"{wrm:5.0%} {medspr:7.2%} {int((g5>0).sum()):>4}/{len(g5)}  "
              f"[{p['ci_lo']:+.1%},{p['ci_hi']:+.1%}]{flag}")
    print("\n  edge uniform across terciles -> broad effect (not manipulation-driven);")
    print("  concentrated in 'low' (wide medSpr) -> the fragile/expensive case.")


if __name__ == "__main__":
    main()
