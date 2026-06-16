"""Build NEW accumulation/distribution + advanced-momentum features.

These families are genuinely absent from the 517-col feature parquet (which has
OBV/VWAP/Amihud but no Chaikin Money Flow, Money Flow Index, Force Index, Ease of
Movement, Volume-Price Trend, nor 12-1 / market-adjusted momentum). All are
computed PIT-safe (trailing only) and made cross-sectionally comparable (bounded
or volume/price-normalized) so the LambdaRank can split on raw values per date.

Writes data/accum_features.parquet (date, ticker, + new cols).

Usage:
    PYTHONPATH=src python scripts/build_accum_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "data" / "accum_features.parquet"

NEW_FEATURES = [
    "cmf_20", "cmf_60", "mfi_14", "force_index_norm", "eom_14",
    "vpt_mom_20", "adl_slope_20", "mom_12_1", "mom_mktadj_60", "mom_mktadj_120",
]


def _per_ticker(g: pd.DataFrame) -> pd.DataFrame:
    h, lo, c, v = g["high"], g["low"], g["close"], g["volume"]
    o = g.copy()
    rng = (h - lo).replace(0, np.nan)

    # ── Accumulation / distribution ──
    mfm = ((c - lo) - (h - c)) / rng         # money-flow multiplier in [-1, 1]
    mfv = mfm * v                            # money-flow volume
    o["cmf_20"] = mfv.rolling(20).sum() / v.rolling(20).sum()
    o["cmf_60"] = mfv.rolling(60).sum() / v.rolling(60).sum()
    adl = mfv.cumsum()
    o["adl_slope_20"] = (adl - adl.shift(20)) / v.rolling(20).sum()

    tp = (h + lo + c) / 3                     # Money Flow Index (14)
    rmf = tp * v
    up = tp > tp.shift(1)
    pos = rmf.where(up, 0.0).rolling(14).sum()
    neg = rmf.where(~up, 0.0).rolling(14).sum().replace(0, np.nan)
    o["mfi_14"] = 100 - 100 / (1 + pos / neg)

    relvol = v / v.rolling(20).mean()         # relative volume
    ret1 = c.pct_change()
    o["force_index_norm"] = (ret1 * relvol).ewm(span=13, adjust=False).mean()
    o["vpt_mom_20"] = (ret1 * relvol).rolling(20).sum()

    dm = ((h + lo) / 2) - ((h.shift(1) + lo.shift(1)) / 2)   # Ease of Movement
    box = relvol / rng.replace(0, np.nan)
    o["eom_14"] = (dm / c / box.replace(0, np.nan)).rolling(14).mean()

    # ── Advanced momentum ──
    o["mom_12_1"] = c.shift(21) / c.shift(252) - 1          # 12-1 (skip last month)
    return o


def main() -> None:
    oh = pd.read_parquet(ROOT / "data/processed/ohlcv_smallcap.parquet")
    oh["date"] = pd.to_datetime(oh["date"])
    oh = oh.sort_values(["ticker", "date"])
    print(f"OHLCV: {len(oh):,} rows, {oh['ticker'].nunique()} tickers")

    out = oh.groupby("ticker", group_keys=False).apply(_per_ticker)

    # Market-adjusted momentum (stock minus SPY over the window), PIT-safe.
    spy = pd.read_parquet(ROOT / "data/processed/smallcap_spy.parquet")[["date", "close"]]
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date")
    spy["mkt_60"] = spy["close"] / spy["close"].shift(60) - 1
    spy["mkt_120"] = spy["close"] / spy["close"].shift(120) - 1
    out = out.merge(spy[["date", "mkt_60", "mkt_120"]], on="date", how="left")
    g = out.groupby("ticker")["close"]
    out["mom_mktadj_60"] = (out["close"] / g.shift(60) - 1) - out["mkt_60"]
    out["mom_mktadj_120"] = (out["close"] / g.shift(120) - 1) - out["mkt_120"]

    keep = ["date", "ticker"] + NEW_FEATURES
    res = out[keep].replace([np.inf, -np.inf], np.nan)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT, index=False)
    print(f"Saved {len(res):,} rows x {len(NEW_FEATURES)} new features -> {OUT}")
    print("Non-null coverage:")
    for f in NEW_FEATURES:
        print(f"  {f:18s} {res[f].notna().mean():.1%}")


if __name__ == "__main__":
    main()
