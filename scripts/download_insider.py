"""Download SEC Form 4 insider OPEN-MARKET BUYS → data/insider_form4.parquet.

Insider buying (especially cluster buys by officers) is a documented, price-
orthogonal small-cap signal: open-market purchases (Form 4 transaction code 'P')
predict positive abnormal returns. Unlike short interest it doesn't collide with
the squeeze effect, so it's a cleaner orthogonal candidate.

Source: SEC's free quarterly "Form 345" structured data sets. We read only the
purchases (NONDERIV_TRANS TRANS_CODE='P', acquired) joined to SUBMISSION for the
issuer ticker + FILING_DATE, restricted to our universe, aggregated per
(ticker, filing_date): number of buy filings (cluster proxy) + shares + $ value.

PIT NOTE: Form 4 must be filed within ~2 business days of the trade and is public
on the FILING_DATE. The feature builder lags availability to filing_date + 1
business day (never the transaction date).

Usage:
    PYTHONPATH=src python scripts/download_insider.py
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from app.data.free_sources.sec_edgar import HEADERS  # noqa: E402

OUT = ROOT / "data" / "insider_form4.parquet"
BASE = ("https://www.sec.gov/files/structureddata/data/"
        "insider-transactions-data-sets/{y}q{q}_form345.zip")


def _quarters(start_year: int = 2021, start_q: int = 2) -> list[tuple[int, int]]:
    out = []
    y, q = start_year, start_q
    end_y, end_q = 2026, 2  # try through 2026q2 (skipped if not yet posted)
    while (y, q) <= (end_y, end_q):
        out.append((y, q))
        q += 1
        if q > 4:
            q, y = 1, y + 1
    return out


def main() -> None:
    uni = pd.read_parquet(ROOT / "data/processed/smallcap_universe.parquet")
    universe = set(uni["ticker"].astype(str).str.upper())
    print(f"Universe: {len(universe)} tickers")

    frames = []
    with httpx.Client(headers=HEADERS, timeout=120, follow_redirects=True) as c:
        for y, q in _quarters():
            url = BASE.format(y=y, q=q)
            try:
                resp = c.get(url)
                if resp.status_code != 200:
                    print(f"  {y}q{q}: HTTP {resp.status_code} — skip")
                    continue
                z = zipfile.ZipFile(io.BytesIO(resp.content))
            except Exception as e:  # noqa: BLE001
                print(f"  {y}q{q}: ERROR {str(e)[:50]} — skip")
                continue

            sub = pd.read_csv(z.open("SUBMISSION.tsv"), sep="\t", dtype=str,
                              usecols=["ACCESSION_NUMBER", "FILING_DATE",
                                       "ISSUERTRADINGSYMBOL"])
            nt = pd.read_csv(z.open("NONDERIV_TRANS.tsv"), sep="\t", dtype=str,
                             usecols=["ACCESSION_NUMBER", "TRANS_CODE",
                                      "TRANS_ACQUIRED_DISP_CD", "TRANS_SHARES",
                                      "TRANS_PRICEPERSHARE"])
            buys = nt[(nt["TRANS_CODE"] == "P") & (nt["TRANS_ACQUIRED_DISP_CD"] == "A")].copy()
            buys["shares"] = pd.to_numeric(buys["TRANS_SHARES"], errors="coerce")
            buys["price"] = pd.to_numeric(buys["TRANS_PRICEPERSHARE"], errors="coerce")
            buys["value"] = buys["shares"] * buys["price"]
            # one buy event = one filing (accession): sum its transactions
            per_acc = buys.groupby("ACCESSION_NUMBER").agg(
                shares=("shares", "sum"), value=("value", "sum")).reset_index()
            m = per_acc.merge(sub, on="ACCESSION_NUMBER", how="left")
            m["ticker"] = m["ISSUERTRADINGSYMBOL"].str.upper()
            m = m[m["ticker"].isin(universe)]
            if not m.empty:
                frames.append(m[["ticker", "FILING_DATE", "ACCESSION_NUMBER",
                                 "shares", "value"]])
            print(f"  {y}q{q}: {len(m)} universe buy-filings")

    if not frames:
        print("No data downloaded — aborting.")
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    df["filing_date"] = pd.to_datetime(df["FILING_DATE"], errors="coerce")
    df = df.dropna(subset=["filing_date", "ticker"])
    agg = df.groupby(["ticker", "filing_date"]).agg(
        n_buy_filings=("ACCESSION_NUMBER", "nunique"),
        buy_shares=("shares", "sum"),
        buy_value=("value", "sum"),
    ).reset_index().sort_values(["ticker", "filing_date"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(OUT, index=False)
    lo, hi = agg["filing_date"].min().date(), agg["filing_date"].max().date()
    print(f"\nSaved {len(agg):,} (ticker,date) rows for {agg['ticker'].nunique()} "
          f"tickers -> {OUT}")
    print(f"  Date range: {lo} -> {hi}")
    print(f"  Total buy filings: {int(agg['n_buy_filings'].sum()):,}")


if __name__ == "__main__":
    main()
