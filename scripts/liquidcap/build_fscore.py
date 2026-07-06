"""LIQUIDCAP feature research — Piotroski F-score (2000) from EDGAR facts.

9-point fundamental quality/value score, point-in-time (filing-date anchored,
usable NEXT day). Components (each 0/1), YoY = vs the filing ~4 quarters back:

  profitability : ROA>0, CFO>0, ΔROA>0, accrual (CFO>NI)
  leverage/liq  : ΔLeverage<0, ΔCurrentRatio>0, no new shares
  efficiency    : ΔOpMargin>0, ΔAssetTurnover>0

Reuses build_fundamentals' EDGAR parsing. Its components (roa, accruals, debt,
margin, asset growth) are ALREADY individual features, so the score's value is
whether the COMPOSITE adds marginal power a tree can't already assemble.

Output: data/liquidcap/fscore_daily.parquet (date, ticker, f_score)
Usage:
    PYTHONPATH=src python scripts/liquidcap/build_fscore.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "liquidcap"))

from build_fundamentals import latest_per_filing  # noqa: E402

DATA = ROOT / "data" / "liquidcap"
FACTS_FP = DATA / "edgar_facts_raw.parquet"
OUT_FP = DATA / "fscore_daily.parquet"


def main() -> None:
    t0 = time.time()
    facts = pd.read_parquet(FACTS_FP)
    facts["start_date"] = pd.to_datetime(facts.get("start_date"), errors="coerce")

    spec = [("net_income", True), ("cfo", True), ("revenue", True),
            ("revenue_contracts", True), ("operating_income", True),
            ("total_assets", False), ("current_assets", False),
            ("current_liabilities", False), ("long_term_debt", False),
            ("shares_outstanding", False)]
    parts = None
    for concept, ann in spec:
        p = latest_per_filing(facts, concept, ann)
        parts = p if parts is None else parts.merge(p, on=["ticker", "filed"], how="outer")
    f = parts.sort_values(["ticker", "filed"]).reset_index(drop=True)
    f["revenue"] = f["revenue"].fillna(f["revenue_contracts"])
    g = f.groupby("ticker", group_keys=False)
    for c in [c for c, _ in spec]:
        f[c] = g[c].ffill()

    # Level ratios
    f["roa"] = f["net_income"] / f["total_assets"].replace(0, np.nan)
    f["lev"] = f["long_term_debt"].fillna(0) / f["total_assets"].replace(0, np.nan)
    f["curr"] = f["current_assets"] / f["current_liabilities"].replace(0, np.nan)
    f["margin"] = f["operating_income"] / f["revenue"].replace(0, np.nan)
    f["turn"] = f["revenue"] / f["total_assets"].replace(0, np.nan)

    # YoY prior (≈4 filings back)
    def lag(col):
        return g[col].shift(4)

    p1 = (f["roa"] > 0).astype(float)
    p2 = (f["cfo"] > 0).astype(float)
    p3 = (f["roa"] > lag("roa")).astype(float)
    p4 = (f["cfo"] > f["net_income"]).astype(float)                 # accrual
    p5 = (f["lev"] < lag("lev")).astype(float)                      # ΔLeverage<0
    p6 = (f["curr"] > lag("curr")).astype(float)                    # ΔCurrentRatio>0
    p7 = (f["shares_outstanding"] <= lag("shares_outstanding")).astype(float)  # no new shares
    p8 = (f["margin"] > lag("margin")).astype(float)               # ΔMargin>0
    p9 = (f["turn"] > lag("turn")).astype(float)                   # ΔAssetTurnover>0
    comp = pd.concat([p1, p2, p3, p4, p5, p6, p7, p8, p9], axis=1)
    # require at least the 4 profitability signals present to score (else NaN)
    valid = f[["roa", "cfo"]].notna().all(axis=1) & lag("roa").notna()
    f["f_score"] = np.where(valid, comp.sum(axis=1), np.nan)

    panel = pd.read_parquet(DATA / "ohlcv_sp500.parquet", columns=["date", "ticker"])
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel.ticker != "SPY"].sort_values("date")
    ff = f.rename(columns={"filed": "date"})[["date", "ticker", "f_score"]].sort_values("date")
    merged = pd.merge_asof(panel, ff, on="date", by="ticker", allow_exact_matches=False)
    out = merged.dropna(subset=["f_score"])[["date", "ticker", "f_score"]]
    out.to_parquet(OUT_FP, index=False)
    print(f"  saved {OUT_FP.name}: {len(out):,} rows, coverage "
          f"{merged['f_score'].notna().mean():.0%}, "
          f"mean f_score {out['f_score'].mean():.2f}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
