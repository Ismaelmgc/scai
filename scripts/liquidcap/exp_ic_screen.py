"""LIQUIDCAP feature research — cheap IC screen (no retraining).

For each candidate feature, measures on the SAME purged OOS test folds as the
frozen GI model (cached predictions):
  IC_std   mean daily Spearman(candidate, fwd_ret_20d_sector_rel)  — raw power
  IC_marg  partial Spearman controlling for the GI prediction      — does it add
           anything BEYOND the current 25 features?

A candidate with IC_marg ≈ 0 is already captured by GI (skip). Only those with a
consistent, non-trivial marginal IC earn a full-harness test. Sign is free — the
tree model uses either direction; we report it for interpretation.

Usage:
    PYTHONPATH=src python scripts/liquidcap/exp_ic_screen.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data" / "liquidcap"
CACHE = DATA / "cache" / "GI_fs15_fund_20d"
TARGET = "fwd_ret_20d_sector_rel"
CANDIDATES = ["high_52w_prox", "resid_mom_12_1", "rev_5d", "beta_252d",
              "skew_60d", "mom_radj"]


def partial_ic(cand: np.ndarray, targ: np.ndarray, ctrl: np.ndarray) -> float:
    """Spearman partial correlation of cand vs targ, controlling for ctrl.
    Residualise ranks of cand and targ on rank of ctrl, then correlate."""
    if len(cand) < 12:
        return np.nan
    rc, rt, rk = rankdata(cand), rankdata(targ), rankdata(ctrl)
    A = np.c_[np.ones_like(rk), rk]
    # residuals after regressing each on the control rank
    res_c = rc - A @ np.linalg.lstsq(A, rc, rcond=None)[0]
    res_t = rt - A @ np.linalg.lstsq(A, rt, rcond=None)[0]
    if np.std(res_c) == 0 or np.std(res_t) == 0:
        return np.nan
    return float(np.corrcoef(res_c, res_t)[0, 1])


def main() -> None:
    feats = pd.read_parquet(DATA / "features_research.parquet",
                            columns=["date", "ticker"] + CANDIDATES)
    feats["date"] = pd.to_datetime(feats["date"])
    meta = json.loads((CACHE / "meta.json").read_text())

    rows = {c: {"std": [], "marg": []} for c in CANDIDATES}
    for fm in meta:
        td = pd.read_parquet(CACHE / f"fold_{fm['fold']:02d}.parquet")
        td["date"] = pd.to_datetime(td["date"])
        td = td.merge(feats, on=["date", "ticker"], how="left")
        for d in sorted(td.date.unique()):
            day = td[td.date == d]
            if len(day) < 15:
                continue
            for c in CANDIDATES:
                sub = day[[c, TARGET, "pred"]].dropna()
                if len(sub) < 15:
                    continue
                ic, _ = spearmanr(sub[c], sub[TARGET])
                mic = partial_ic(sub[c].values, sub[TARGET].values, sub["pred"].values)
                if np.isfinite(ic):
                    rows[c]["std"].append(ic)
                if np.isfinite(mic):
                    rows[c]["marg"].append(mic)

    print(f"\n  {'candidate':16} {'IC_std':>9} {'IC_marg':>9} {'hit%':>6} {'n':>6}")
    print("  " + "-" * 50)
    res = []
    for c in CANDIDATES:
        s = np.array(rows[c]["std"]); m = np.array(rows[c]["marg"])
        ic_std = float(np.mean(s)) if len(s) else 0.0
        ic_marg = float(np.mean(m)) if len(m) else 0.0
        hit = float(np.mean(np.sign(m) == np.sign(ic_marg))) if len(m) else 0.0
        res.append((c, ic_std, ic_marg, hit, len(m)))
    for c, ic_std, ic_marg, hit, n in sorted(res, key=lambda x: -abs(x[2])):
        flag = " <-- keep" if abs(ic_marg) >= 0.010 else ""
        print(f"  {c:16} {ic_std:+9.4f} {ic_marg:+9.4f} {hit:5.0%} {n:6d}{flag}")
    print("\n  IC_marg = partial Spearman vs GI's prediction (marginal power).")
    print("  keep threshold |IC_marg| >= 0.010 with consistent sign.")


if __name__ == "__main__":
    main()
