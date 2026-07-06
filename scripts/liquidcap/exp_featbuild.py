"""LIQUIDCAP feature research — build the EXTENDED matrix (production base + new
Lote-1 candidates), so the IC screen + harness can test whether any add edge.

Reuses the exact production anti-leak path (build_feature_matrix + _add_lag +
add_new_features) then appends price-only cross-sectional candidates with strong
academic backing (all windows END at T — no look-ahead):

  high_52w_prox   close / 252d-high         (George-Hwang 2004, 52-week high)
  resid_mom_12_1  Σ residual ret [t-252,t-21] (Blitz 2011, residual momentum)
  rev_5d          close/close_-5 - 1         (short-term weekly reversal)
  beta_252d       cov(ret,mkt)/var, 252d     (Frazzini-Pedersen low-beta)
  skew_60d        skew of daily ret, 60d     (Boyer-Mitton idiosyncratic skew)
  mom_radj        ret_252d / realized_vol    (risk-adjusted momentum)

Output: data/liquidcap/features_research.parquet (gitignored, regenerable).

Usage:
    PYTHONPATH=src python scripts/liquidcap/exp_featbuild.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "liquidcap"))

from run_smallcap_pipeline import _add_lag_features  # noqa: E402

from app.features.pipeline import build_feature_matrix  # noqa: E402
from build_features_liquidcap import add_new_features  # noqa: E402

DATA = ROOT / "data" / "liquidcap"
CANDIDATES = ["high_52w_prox", "resid_mom_12_1", "rev_5d", "beta_252d",
              "skew_60d", "mom_radj"]


def add_candidates(feats: pd.DataFrame, panel: pd.DataFrame,
                   spy: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_values(["ticker", "date"]).copy()
    g = p.groupby("ticker", group_keys=False)
    p["ret"] = g["close"].pct_change()
    spy = spy.sort_values("date")[["date", "close"]].copy()
    spy["mkt_ret"] = spy["close"].pct_change()
    p = p.merge(spy[["date", "mkt_ret"]], on="date", how="left")

    def roll(col, win, fn, mp=None):
        mp = mp or win // 2
        return (p.groupby("ticker", group_keys=False)[col]
                .rolling(win, min_periods=mp).agg(fn).reset_index(level=0, drop=True))

    # 52-week high proximity (1.0 = at the high)
    hi252 = roll("high", 252, "max", mp=200)
    p["high_52w_prox"] = p["close"] / hi252

    # residual momentum: rolling 60d beta -> residual ret -> Σ over [t-252,t-21]
    cov = (p.groupby("ticker", group_keys=False)
           .apply(lambda d: d["ret"].rolling(60, min_periods=30).cov(d["mkt_ret"]))
           .reset_index(level=0, drop=True))
    var = p["mkt_ret"].rolling(60, min_periods=30).var()
    beta60 = cov / var.reindex(cov.index)
    p["resid"] = p["ret"] - beta60 * p["mkt_ret"]
    resid_sum = roll("resid", 231, "sum", mp=150)  # 252-21 window length
    p["resid_mom_12_1"] = p.groupby("ticker", group_keys=False)["resid"].apply(
        lambda s: s.rolling(231, min_periods=150).sum().shift(21)).reset_index(level=0, drop=True)
    _ = resid_sum  # (kept explicit for clarity; the shifted version above is used)

    # weekly reversal
    p["rev_5d"] = p["close"] / g["close"].shift(5) - 1

    # long-window beta (low-beta anomaly)
    cov252 = (p.groupby("ticker", group_keys=False)
              .apply(lambda d: d["ret"].rolling(252, min_periods=150).cov(d["mkt_ret"]))
              .reset_index(level=0, drop=True))
    var252 = p["mkt_ret"].rolling(252, min_periods=150).var()
    p["beta_252d"] = cov252 / var252.reindex(cov252.index)

    # idiosyncratic skewness (lottery demand)
    p["skew_60d"] = roll("ret", 60, "skew", mp=40)

    # risk-adjusted 12m momentum
    ret252 = g["close"].shift(0) / g["close"].shift(252) - 1
    vol = roll("ret", 120, "std", mp=80) * np.sqrt(252)
    p["mom_radj"] = ret252 / vol.replace(0, np.nan)

    return feats.merge(p[["date", "ticker"] + CANDIDATES], on=["date", "ticker"], how="left")


def main() -> None:
    t0 = time.time()
    panel = pd.read_parquet(DATA / "ohlcv_sp500.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel.ticker != "SPY"]
    spy = pd.read_parquet(DATA / "spy.parquet")
    spy["date"] = pd.to_datetime(spy["date"])
    mem = pd.read_parquet(DATA / "membership_sp500.parquet")
    print(f"  panel {len(panel):,} rows / {panel.ticker.nunique()} tickers", flush=True)

    universe = [{"ticker": t} for t in panel["ticker"].unique()]
    feats = build_feature_matrix(panel, fundamentals=None, market_df=spy,
                                 universe=universe, horizons=[1, 5, 10, 20])
    feats = _add_lag_features(feats)
    feats = add_new_features(feats, panel, spy)
    print(f"  base+new built ({(time.time()-t0)/60:.1f} min); adding candidates", flush=True)
    feats = add_candidates(feats, panel, spy)

    # PIT membership filter (LAST, so rolling windows saw full history)
    feats["date"] = pd.to_datetime(feats["date"])
    n0 = len(feats)
    keep = pd.Series(False, index=feats.index)
    for tkr, grp in mem.groupby("ticker"):
        m = feats["ticker"] == tkr
        if not m.any():
            continue
        ok = pd.Series(False, index=feats.index[m])
        for _, sp in grp.iterrows():
            ok |= (feats.loc[m, "date"] >= sp["start"]) & (feats.loc[m, "date"] <= sp["end"])
        keep.loc[ok.index[ok]] = True
    feats = feats[keep]
    print(f"  PIT filter: {n0:,} -> {len(feats):,} rows", flush=True)

    fp = DATA / "features_research.parquet"
    feats.to_parquet(fp, index=False)
    cov = {c: float(feats[c].notna().mean()) for c in CANDIDATES}
    print(f"  saved {fp.name} ({(time.time()-t0)/60:.1f} min)")
    print("  candidate coverage:", {k: round(v, 2) for k, v in cov.items()})


if __name__ == "__main__":
    main()
