"""LIQUIDCAP — free point-in-time fundamentals from SEC EDGAR (no Tiingo paid).

Downloads XBRL company facts for every PIT S&P 500 member (reuses the
production sec_edgar client — filing_date anchored, so a ratio only becomes
visible the day the 10-K/10-Q was actually filed) and computes the classic
cross-sectional ratios the literature supports:

  earnings_yield   annualized net income / market cap   (E/P — inverse P/E)
  cfo_yield        annualized CFO / market cap           (cash P/E)
  sales_yield      annualized revenue / market cap       (inverse P/S)
  book_to_market   equity / market cap                   (value)
  debt_to_equity   (LT + current debt) / equity          (leverage)
  roa              annualized net income / total assets  (profitability)
  op_margin        operating income / revenue            (profitability)
  accruals         (net income - CFO) / assets           (Sloan 1996)
  asset_growth     YoY total assets change               (Cooper et al 2008)
  earnings_chg     annualized NI vs ~1y-ago filing / assets (PEAD-ish drift)

Flow concepts are annualized by filing-period duration (365/days), so 10-K and
10-Q values are comparable. Market-cap denominators are applied DAILY at merge
time (shares from the latest filing × that day's close), so valuation ratios
move with price between filings — as they do in reality.

KNOWN HAZARD (measured and printed): the SEC ticker->CIK registry is CURRENT;
delisted members may not resolve -> their fundamentals are NaN. If coverage of
removed names is materially below current names, the fundamentals config must
be judged with that asymmetry in mind (a model could learn "has fundamentals ~
survived").

Output: data/liquidcap/fundamentals_daily.parquet (date, ticker, 10 ratios)
Usage:
    PYTHONPATH=src python scripts/liquidcap/build_fundamentals.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from app.data.free_sources import sec_edgar  # noqa: E402

DATA = ROOT / "data" / "liquidcap"
FACTS_FP = DATA / "edgar_facts_raw.parquet"
OUT_FP = DATA / "fundamentals_daily.parquet"

RATIOS = ["earnings_yield", "cfo_yield", "sales_yield", "book_to_market",
          "debt_to_equity", "roa", "op_margin", "accruals", "asset_growth",
          "earnings_chg"]


def annualize(df: pd.DataFrame) -> pd.DataFrame:
    """Scale flow values (income/CFO/revenue) to annual run-rate by duration."""
    dur = (df["end_date"] - df["start_date"]).dt.days
    scale = (365.0 / dur.clip(lower=80)).where(dur.notna(), 1.0)
    df = df.copy()
    df["value_ann"] = df["value"] * scale
    return df


def latest_per_filing(facts: pd.DataFrame, concept: str, annualized: bool):
    d = facts[(facts.concept == concept) & facts.form.isin(["10-K", "10-Q"])].copy()
    if d.empty:
        return pd.DataFrame(columns=["ticker", "filed", concept])
    if annualized:
        d = annualize(d)
        col = "value_ann"
    else:
        col = "value"
    # keep the most recent period reported in each filing
    d = d.sort_values(["ticker", "filed", "end_date"])
    d = d.groupby(["ticker", "filed"], as_index=False).last()
    return d[["ticker", "filed", col]].rename(columns={col: concept})


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-facts", action="store_true",
                    help="re-download ALL company facts (picks up new filings)")
    args = ap.parse_args()
    if args.refresh_facts and FACTS_FP.exists():
        FACTS_FP.unlink()

    t0 = time.time()
    mem = pd.read_parquet(DATA / "membership_sp500.parquet")
    tickers = sorted(mem.ticker.unique())
    current = set(mem[mem["end"] >= mem["end"].max()]["ticker"])
    removed = set(tickers) - current

    # Delisted-CIK overrides recovered by company name (recover_delisted_ciks.py)
    ov_fp = DATA / "cik_overrides.json"
    overrides = json.loads(ov_fp.read_text()) if ov_fp.exists() else {}

    cik = sec_edgar.get_cik_map(tickers, overrides=overrides)
    cov_cur = len([t for t in current if t in cik]) / len(current)
    cov_rem = len([t for t in removed if t in cik]) / len(removed)
    print(f"  CIK coverage (+{len(overrides)} overrides): current {cov_cur:.0%} "
          f"({len(current)}), REMOVED {cov_rem:.0%} ({len(removed)})"
          f"  <- survivorship asymmetry check")

    if FACTS_FP.exists():
        facts = pd.read_parquet(FACTS_FP)
        print(f"  (facts cached: {len(facts):,} rows, {facts.ticker.nunique()} tickers)")
        todo = [t for t in tickers if t in cik and t not in set(facts.ticker)]
        if todo:
            print(f"  appending {len(todo)} newly-mapped tickers")
            extra = sec_edgar.download_company_facts(todo, delay=0.12,
                                                     cik_overrides=overrides)
            if not extra.empty:
                facts = pd.concat([facts, extra], ignore_index=True)
                facts.to_parquet(FACTS_FP, index=False)
    else:
        facts = sec_edgar.download_company_facts(tickers, delay=0.12,
                                                 cik_overrides=overrides)
        facts.to_parquet(FACTS_FP, index=False)
        print(f"  facts: {len(facts):,} rows, {facts.ticker.nunique()} tickers")
    facts["start_date"] = pd.to_datetime(facts.get("start_date"), errors="coerce")

    # One row per (ticker, filed) with the concepts we need
    parts = None
    spec = [("net_income", True), ("cfo", True), ("revenue", True),
            ("revenue_contracts", True),  # ASC-606 tag — half the filers use it
            ("operating_income", True), ("equity", False), ("total_assets", False),
            ("long_term_debt", False), ("current_debt", False),
            ("shares_outstanding", False)]
    for concept, ann in spec:
        p = latest_per_filing(facts, concept, ann)
        parts = p if parts is None else parts.merge(p, on=["ticker", "filed"], how="outer")
    f = parts.sort_values(["ticker", "filed"]).reset_index(drop=True)
    # tag-standardization fallback: Revenues | RevenueFromContractWithCustomer...
    f["revenue"] = f["revenue"].fillna(f["revenue_contracts"])
    # forward-fill balance-sheet items within ticker (a 10-Q may omit some)
    g = f.groupby("ticker", group_keys=False)
    for c in [c for c, _ in spec]:
        f[c] = g[c].ffill()

    f["roa"] = f["net_income"] / f["total_assets"].replace(0, np.nan)
    f["op_margin"] = f["operating_income"] / f["revenue"].replace(0, np.nan)
    f["debt_to_equity"] = ((f["long_term_debt"].fillna(0) + f["current_debt"].fillna(0))
                           / f["equity"].replace(0, np.nan))
    f["accruals"] = (f["net_income"] - f["cfo"]) / f["total_assets"].replace(0, np.nan)
    f["asset_growth"] = g["total_assets"].pct_change(4)   # ~4 filings = 1y
    f["earnings_chg"] = (f["net_income"] - g["net_income"].shift(4)) \
        / f["total_assets"].replace(0, np.nan)

    # Daily as-of merge (available from the filing date onward)
    panel = pd.read_parquet(DATA / "ohlcv_sp500.parquet",
                            columns=["date", "ticker", "close"])
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel.ticker != "SPY"].sort_values("date")
    f = f.rename(columns={"filed": "date"}).sort_values("date")
    keep = ["date", "ticker", "net_income", "cfo", "revenue", "equity",
            "shares_outstanding", "roa", "op_margin", "debt_to_equity",
            "accruals", "asset_growth", "earnings_chg"]
    merged = pd.merge_asof(panel, f[keep], on="date", by="ticker",
                           allow_exact_matches=False)  # filed day -> usable NEXT day

    mcap = merged["close"] * merged["shares_outstanding"]
    merged["earnings_yield"] = merged["net_income"] / mcap
    merged["cfo_yield"] = merged["cfo"] / mcap
    merged["sales_yield"] = merged["revenue"] / mcap
    merged["book_to_market"] = merged["equity"] / mcap

    out = merged[["date", "ticker"] + RATIOS]
    out.to_parquet(OUT_FP, index=False)
    non_nan = out[RATIOS].notna().mean()
    print(f"\n  saved {OUT_FP.name}: {len(out):,} rows")
    print("  coverage per ratio:\n" + non_nan.to_string())
    print(f"  Runtime {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
