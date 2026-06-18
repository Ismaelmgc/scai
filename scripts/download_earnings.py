"""Download SEC earnings-announcement dates -> data/earnings.parquet (PEAD).

Post-earnings-announcement drift (PEAD) is the strongest horizon-matched anomaly
left untested: the drift concentrates over ~1-2 months, which fits our 20d target.
We avoid fragile EPS/analyst data entirely and use a PRICE-REACTION construction:
the earnings DATE (event) + the abnormal return around it (the market's own
surprise). This script does Phase-0 = pull the event dates and report coverage.

Source: SEC submissions API (free, ~10 req/s with a User-Agent). For each universe
ticker we map ticker->CIK via company_tickers.json, then read the recent filings
and keep the genuine earnings-announcement events:
  * 8-K with item 2.02 (Results of Operations) = the earnings press release, the
    true announcement date and the cleanest PEAD event;
  * 10-Q / 10-K filing dates as a secondary/fallback event.

PIT NOTE: the announcement date is public same day; the feature builder will use
the reaction only from date+1 onward.

Usage:
    PYTHONPATH=src python scripts/download_earnings.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from app.data.free_sources.sec_edgar import HEADERS  # noqa: E402

OUT = ROOT / "data" / "earnings.parquet"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUB_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
START = pd.Timestamp("2021-01-01")


def _ticker_cik_map(client: httpx.Client, universe: set[str]) -> dict[str, int]:
    data = client.get(TICKERS_URL).json()
    out: dict[str, int] = {}
    for row in data.values():
        t = str(row["ticker"]).upper()
        if t in universe:
            out[t] = int(row["cik_str"])
    return out


def _events_for(client: httpx.Client, cik: int) -> list[tuple[pd.Timestamp, str]]:
    r = client.get(SUB_URL.format(cik=cik))
    if r.status_code != 200:
        return []
    recent = r.json().get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    items = recent.get("items", [""] * len(forms))
    ev = []
    for form, d, it in zip(forms, dates, items, strict=False):
        ts = pd.to_datetime(d, errors="coerce")
        if ts is pd.NaT or ts < START:
            continue
        if form == "8-K" and "2.02" in (it or ""):
            ev.append((ts, "8K_202"))
        elif form in ("10-Q", "10-K"):
            ev.append((ts, form))
    return ev


def main() -> None:
    uni = pd.read_parquet(ROOT / "data/processed/smallcap_universe.parquet")
    universe = set(uni["ticker"].astype(str).str.upper())
    print(f"Universe: {len(universe)} tickers")

    rows = []
    with httpx.Client(headers=HEADERS, timeout=60, follow_redirects=True) as c:
        cik_map = _ticker_cik_map(c, universe)
        print(f"Mapped ticker->CIK: {len(cik_map)}/{len(universe)} "
              f"(unmapped are mostly delisted)")
        for i, (t, cik) in enumerate(sorted(cik_map.items())):
            try:
                for ts, kind in _events_for(c, cik):
                    rows.append({"ticker": t, "date": ts, "kind": kind})
            except Exception as e:  # noqa: BLE001
                print(f"  {t}: ERROR {str(e)[:40]}")
            time.sleep(0.11)  # be polite (~9 req/s)
            if (i + 1) % 200 == 0:
                print(f"  ...{i + 1}/{len(cik_map)} tickers")

    if not rows:
        print("No events downloaded - aborting.")
        sys.exit(1)

    df = pd.DataFrame(rows).drop_duplicates().sort_values(["ticker", "date"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    # ── coverage gate ──
    er = df[df["kind"] == "8K_202"]
    per_t = er.groupby("ticker").size()
    print(f"\nSaved {len(df):,} events ({df['ticker'].nunique()} tickers) -> {OUT}")
    print(f"  Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print("  --- coverage gate (8-K item 2.02 = earnings releases) ---")
    print(f"  tickers with >=1 earnings release : {er['ticker'].nunique()}/{len(universe)} "
          f"({er['ticker'].nunique()/len(universe):.0%})")
    print(f"  tickers with >=8 (>=2yr quarterly): {(per_t >= 8).sum()}")
    print(f"  median releases per covered ticker: {per_t.median():.0f}")
    print(f"  total earnings releases           : {len(er):,}")
    print(f"  (10-Q/10-K fallback events        : {(df['kind'] != '8K_202').sum():,})")


if __name__ == "__main__":
    main()
