"""Build price-reaction PEAD features + standalone IC screen.

Coverage gate (download_earnings.py) showed 86% coverage on the ACTIVE universe
(delisted names lack a current CIK). Before investing in a full feature module +
harness run, screen the signal's standalone per-date IC on the TRADABLE cross-
section (active names, where coverage is good). If it is noise like short-interest
/insider were, stop here.

Signal (no EPS/analyst data): the market's own surprise = abnormal return around
the earnings announcement, then the documented drift continues for ~1-2 months.
  react   = (close_{d+1}/close_{d-1} - 1) - (spy_{d+1}/spy_{d-1} - 1)   at event d
  avail   = d+1  (PIT-safe: reaction known only the day after)
  pead_react      = signed reaction of the most recent event within 60d
  pead_react_decay= react * (1 - days_since/60)   (drift fades)
  pead_days_since = sessions since the event

Writes data/pead_features.parquet (if the screen is run with `save`).

Usage:
    PYTHONPATH=src python scripts/build_pead_features.py          # screen only
    PYTHONPATH=src python scripts/build_pead_features.py save     # screen + write parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "pead_features.parquet"
TARGET = "fwd_ret_20d_sector_rel"
DRIFT_DAYS = 60


def build() -> pd.DataFrame:
    ev = pd.read_parquet(ROOT / "data/earnings.parquet")
    ev = ev[ev["kind"] == "8K_202"][["ticker", "date"]].copy()
    ev["date"] = pd.to_datetime(ev["date"])

    oh = pd.read_parquet(ROOT / "data/processed/ohlcv_smallcap.parquet",
                         columns=["date", "ticker", "close"])
    oh["date"] = pd.to_datetime(oh["date"])
    oh = oh.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = oh.groupby("ticker")
    oh["c_m1"] = g["close"].shift(1)
    oh["c_p1"] = g["close"].shift(-1)
    oh["avail"] = g["date"].shift(-1)          # next trading day = availability

    spy = pd.read_parquet(ROOT / "data/processed/smallcap_spy.parquet")[["date", "close"]]
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date")
    spy["spy_react"] = spy["close"].shift(-1) / spy["close"].shift(1) - 1
    oh = oh.merge(spy[["date", "spy_react"]], on="date", how="left")

    # reaction measured ON the event date d (uses d-1 and d+1 closes)
    e = ev.merge(oh[["ticker", "date", "c_m1", "c_p1", "avail", "spy_react"]],
                 on=["ticker", "date"], how="inner")
    e["react"] = (e["c_p1"] / e["c_m1"] - 1) - e["spy_react"]
    e = e.dropna(subset=["react", "avail"])
    e = e[["ticker", "date", "avail", "react"]].rename(columns={"date": "event_date"})
    e = e.sort_values("avail")

    # attach the most recent already-available event to every (ticker, date)
    base = oh[["ticker", "date"]].sort_values("date")
    merged = pd.merge_asof(base, e, left_on="date", right_on="avail",
                           by="ticker", direction="backward")
    dsince = (merged["date"] - merged["event_date"]).dt.days
    merged["pead_days_since"] = dsince
    fresh = dsince <= DRIFT_DAYS
    merged["pead_react"] = merged["react"].where(fresh)
    merged["pead_react_decay"] = (merged["react"] * (1 - dsince / DRIFT_DAYS)).where(fresh)
    return merged[["date", "ticker", "pead_react", "pead_react_decay", "pead_days_since"]]


def screen(feats: pd.DataFrame) -> None:
    cols = ["date", "ticker", TARGET, "close", "adv_usd_20d"]
    f = pd.read_parquet(ROOT / "data/processed/features_smallcap.parquet", columns=cols)
    f["date"] = pd.to_datetime(f["date"])
    df = f.merge(feats, on=["date", "ticker"], how="left")
    trad = df[(df["close"] >= 1.5) & (df["adv_usd_20d"] >= 500_000)].dropna(subset=[TARGET])
    print(f"\n  tradable rows: {len(trad):,}  | PEAD non-null on tradable: "
          f"{trad['pead_react'].notna().mean():.0%}")
    print(f"  {'feature':18s} {'med IC':>8} {'mean IC':>8}  (per-date Spearman vs target)")
    for c in ["pead_react", "pead_react_decay"]:
        ics = []
        for _, gdf in trad[["date", c, TARGET]].dropna().groupby("date"):
            if len(gdf) >= 20:
                ics.append(spearmanr(gdf[c], gdf[TARGET]).correlation)
        ics = np.array([x for x in ics if x == x])
        print(f"  {c:18s} {np.median(ics):+8.4f} {np.mean(ics):+8.4f}")


def main() -> None:
    feats = build()
    screen(feats)
    if "save" in sys.argv:
        feats.to_parquet(OUT, index=False)
        print(f"\n  saved -> {OUT}")


if __name__ == "__main__":
    main()
