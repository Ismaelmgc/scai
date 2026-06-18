"""Critical diagnostics on the overnight large-cap result: is the net edge real
or tail-/fold-driven? Reads the cached folds, computes per-fold returns at each
cost tier, gross & net win rates, +folds count, MC CIs, and checks whether a few
clipped (+25%) days dominate the gross compound.
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

CACHE = ROOT / "data" / "largecap" / "cache"
TOP_K, PRICE_FLOOR = 8, 3.0
TIERS = {"gross": 0, "net5": 5, "net10": 10, "net30": 30}


def fold_daily(name: str):
    """Return list of (per-fold cum return dict) and the pooled daily portfolio rets."""
    meta = json.loads((CACHE / f"lc_{name}" / "meta.json").read_text())
    per_fold, pooled_daily = [], []
    for fm in meta:
        td = pd.read_parquet(CACHE / f"lc_{name}" / f"fold_{fm['fold']:02d}.parquet")
        daily = {k: [] for k in TIERS}
        gross_trades = []
        for _, day in td.groupby("date"):
            day = day[day["close"] >= PRICE_FLOOR]
            if len(day) < TOP_K:
                continue
            r = day.sort_values("pred", ascending=False).head(TOP_K)[V2_TARGET].to_numpy()
            r = r[np.isfinite(r)]
            if len(r) == 0:
                continue
            gross_trades.extend(r.tolist())
            for k, b in TIERS.items():
                daily[k].append((r - 2 * b / 1e4).mean())
        if not daily["gross"]:
            continue
        row = {"fold": fm["fold"]}
        for k in TIERS:
            s = pd.Series(daily[k])
            row[k] = float((1 + s).prod()) - 1
        row["gross_wr"] = float(np.mean([t > 0 for t in gross_trades]))
        row["n_days"] = len(daily["gross"])
        per_fold.append(row)
        pooled_daily.extend(daily["gross"])
    return pd.DataFrame(per_fold), np.array(pooled_daily)


for name in ("overnight", "intraday"):
    df, daily = fold_daily(name)
    print(f"\n══ {name} ══  ({len(df)} folds)")
    print(f"  gross WR (per-trade): {df['gross_wr'].mean():.1%}")
    for k in TIERS:
        p = mc.one_sample_bootstrap(df[k].values)
        pos = int((df[k] > 0).sum())
        flag = " <-CI>0" if p["ci_lo"] > 0 else ""
        print(f"  {k:6s} {df[k].mean():+7.1%}/fold  [{p['ci_lo']:+.1%},{p['ci_hi']:+.1%}] "
              f"+folds {pos}/{len(df)}{flag}")
    # Tail concentration: how much of the gross compound comes from the best days?
    s = pd.Series(np.sort(daily)[::-1])
    top5 = s.head(5).sum()
    print(f"  daily port rets: mean {daily.mean():+.3%}  std {daily.std():.3%}  "
          f"max {daily.max():+.2%}  min {daily.min():+.2%}")
    print(f"  sum of all daily rets {s.sum():+.1%}; top-5 days alone {top5:+.1%} "
          f"({top5/s.sum()*100:.0f}% of total)")
