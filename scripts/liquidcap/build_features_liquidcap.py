"""LIQUIDCAP — feature matrix for the PIT S&P 500 panel.

Reuses the production anti-leak path (build_feature_matrix + _add_lag_features,
26 base features minus vwap/transactions unavailable here) and adds 6 NEW
point-in-time families computed from trailing bars only (windows END at T):

  gap_avg_20d            mean overnight gap (open/prev_close-1), 20d
  gap_vol_20d            std of overnight gaps, 20d
  intraday_avg_20d       mean intraday move (close/open-1), 20d
  idio_vol_60d           60d std of (ret - beta_60d*mkt_ret) residuals
  dollar_vol_ratio_20v60 ADV20/ADV60 - 1 (volume trend)
  range_pos_20d          (close - min low 20d) / (max high - min low, 20d)

Rows are then filtered to PIT membership (ticker,date within a spell): training
and selection only ever see names that were IN the index at that date — the
delisted/removed stay up to their removal date (anti-survivorship), and no name
enters before its addition (no selection-date bias).

Output: data/liquidcap/features_liquidcap.parquet

Usage:
    PYTHONPATH=src python scripts/liquidcap/build_features_liquidcap.py
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

from run_smallcap_pipeline import _add_lag_features  # noqa: E402

from app.features.pipeline import build_feature_matrix  # noqa: E402

DATA = ROOT / "data" / "liquidcap"

NEW_FEATURES = ["gap_avg_20d", "gap_vol_20d", "intraday_avg_20d",
                "idio_vol_60d", "dollar_vol_ratio_20v60", "range_pos_20d"]


def add_new_features(feats: pd.DataFrame, panel: pd.DataFrame,
                     spy: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_values(["ticker", "date"]).copy()
    g = p.groupby("ticker", group_keys=False)
    p["prev_close"] = g["close"].shift(1)
    p["gap"] = p["open"] / p["prev_close"] - 1
    p["intraday"] = p["close"] / p["open"] - 1
    p["ret"] = g["close"].pct_change()
    p["dv"] = p["close"] * p["volume"]

    spy = spy.sort_values("date")[["date", "close"]].copy()
    spy["mkt_ret"] = spy["close"].pct_change()
    p = p.merge(spy[["date", "mkt_ret"]], on="date", how="left")

    def roll(col, win, fn):
        return getattr(p.groupby("ticker", group_keys=False)[col]
                       .rolling(win, min_periods=win // 2), fn)().reset_index(level=0, drop=True)

    p["gap_avg_20d"] = roll("gap", 20, "mean")
    p["gap_vol_20d"] = roll("gap", 20, "std")
    p["intraday_avg_20d"] = roll("intraday", 20, "mean")
    p["adv20"] = roll("dv", 20, "mean")
    p["adv60"] = roll("dv", 60, "mean")
    p["dollar_vol_ratio_20v60"] = p["adv20"] / p["adv60"] - 1
    lo20 = roll("low", 20, "min")
    hi20 = roll("high", 20, "max")
    p["range_pos_20d"] = (p["close"] - lo20) / (hi20 - lo20).replace(0, np.nan)
    # idio vol: residual of ret vs rolling 60d beta (both windows end at T)
    cov = (p.groupby("ticker", group_keys=False)
           .apply(lambda d: d["ret"].rolling(60, min_periods=30).cov(d["mkt_ret"]))
           .reset_index(level=0, drop=True))
    var = p["mkt_ret"].rolling(60, min_periods=30).var()
    beta = cov / var.reindex(cov.index)
    p["resid"] = p["ret"] - beta * p["mkt_ret"]
    p["idio_vol_60d"] = roll("resid", 60, "std")

    return feats.merge(p[["date", "ticker"] + NEW_FEATURES], on=["date", "ticker"], how="left")


def main() -> None:
    t0 = time.time()
    panel = pd.read_parquet(DATA / "ohlcv_sp500.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel.ticker != "SPY"]
    spy = pd.read_parquet(DATA / "spy.parquet")
    spy["date"] = pd.to_datetime(spy["date"])
    mem = pd.read_parquet(DATA / "membership_sp500.parquet")
    print(f"  panel {len(panel):,} rows / {panel.ticker.nunique()} tickers; "
          f"membership spells {len(mem)}")

    universe = [{"ticker": t} for t in panel["ticker"].unique()]  # Unknown sector
    feats = build_feature_matrix(panel, fundamentals=None, market_df=spy,
                                 universe=universe, horizons=[1, 5, 10, 20])
    feats = _add_lag_features(feats)
    feats = add_new_features(feats, panel, spy)

    # PIT membership filter — LAST, so rolling windows saw the full history.
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
    print(f"  PIT filter: {n0:,} -> {len(feats):,} rows")

    fp = DATA / "features_liquidcap.parquet"
    feats.to_parquet(fp, index=False)
    print(f"  saved {fp.name}: {feats.ticker.nunique()} tickers, "
          f"{feats.date.min().date()} -> {feats.date.max().date()}, "
          f"{(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
