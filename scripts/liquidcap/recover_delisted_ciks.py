"""LIQUIDCAP — recover CIKs for delisted S&P 500 members by company name.

The SEC ticker->CIK registry (company_tickers.json) only lists CURRENT
registrants, so ~57% of our removed members don't resolve and their EDGAR
fundamentals would be NaN — a survivorship-shaped asymmetry ("has fundamentals
~ survived") that could fake a fundamentals edge. Fix, for free: the removed
companies' NAMES are in the Wikipedia changes table, and the SEC publishes
cik-lookup-data.txt with every registrant name EVER (incl. dead ones).
Normalized name matching recovers most of the missing CIKs.

Output: data/liquidcap/cik_overrides.json {ticker: cik} — consumed by
build_fundamentals.py. Unmatched tickers are printed for the record.

Usage:
    PYTHONPATH=src python scripts/liquidcap/recover_delisted_ciks.py [--html FILE]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from app.data.free_sources.sec_edgar import HEADERS, get_cik_map  # noqa: E402

DATA = ROOT / "data" / "liquidcap"
LOOKUP_URL = "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt"

# suffix noise that differs between Wikipedia and SEC registrant names
_STOP = {"INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "LTD",
         "PLC", "THE", "HOLDINGS", "HOLDING", "GROUP", "COS", "COMPANIES",
         "TRUST", "LP", "LLC", "SA", "NV", "DE", "CL", "A", "B", "NEW"}


def norm(name: str) -> str:
    s = re.sub(r"[^A-Z0-9 ]", " ", str(name).upper())
    toks = [t for t in s.split() if t not in _STOP]
    return " ".join(toks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default=None, help="local Wikipedia snapshot")
    args = ap.parse_args()

    # 1. removed ticker -> company name from the Wikipedia changes table
    if args.html:
        html = Path(args.html).read_text(encoding="utf-8")
    else:
        html = (DATA / "sp500_wiki_snapshot.html").read_text(encoding="utf-8")
    ch = pd.read_html(html)[1]
    ch.columns = ["_".join(c) if isinstance(c, tuple) else c for c in ch.columns]
    ch = ch.rename(columns={"Removed_Ticker": "ticker", "Removed_Security": "name"})
    ch = ch.dropna(subset=["ticker", "name"])
    ch["ticker"] = ch["ticker"].str.upper().str.replace(".", "-", regex=False)
    names = dict(zip(ch["ticker"], ch["name"]))

    mem = pd.read_parquet(DATA / "membership_sp500.parquet")
    tickers = sorted(mem.ticker.unique())
    have = set(get_cik_map(tickers))
    missing = [t for t in tickers if t not in have and t in names]
    print(f"  missing CIKs with a known company name: {len(missing)}")

    # 2. SEC master name->CIK index (every registrant ever)
    lookup_fp = DATA / "cik-lookup-data.txt"
    if not lookup_fp.exists():
        r = requests.get(LOOKUP_URL, headers=HEADERS, timeout=120)
        r.raise_for_status()
        lookup_fp.write_bytes(r.content)
    by_norm: dict[str, str] = {}
    for line in lookup_fp.read_text(encoding="latin-1").splitlines():
        # format: COMPANY NAME:CIK:
        parts = line.rsplit(":", 2)
        if len(parts) < 2 or not parts[1].strip().isdigit():
            continue
        by_norm.setdefault(norm(parts[0]), parts[1].strip())

    overrides: dict[str, str] = {}
    unmatched: list[str] = []
    for t in missing:
        n = norm(names[t])
        cik = by_norm.get(n)
        if cik is None:  # relaxed: unique prefix match
            cands = {v for k, v in by_norm.items() if k.startswith(n) and n}
            cik = cands.pop() if len(cands) == 1 else None
        if cik:
            overrides[t] = cik
        else:
            unmatched.append(f"{t} ({names[t]})")

    fp = DATA / "cik_overrides.json"
    fp.write_text(json.dumps(overrides, indent=2))
    print(f"  recovered {len(overrides)}/{len(missing)} -> {fp.name}")
    if unmatched:
        print(f"  unmatched ({len(unmatched)}): {unmatched[:12]}"
              f"{' ...' if len(unmatched) > 12 else ''}")


if __name__ == "__main__":
    main()
