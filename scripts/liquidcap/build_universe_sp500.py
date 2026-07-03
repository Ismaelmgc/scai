"""LIQUIDCAP experiment — point-in-time S&P 500 membership (anti-survivorship).

Reconstructs WHO was in the index at every date from Wikipedia's current
constituents + the historical changes table, walking changes backwards from
today. Names removed in 2019 (bankrupt, acquired, faded) are in the panel up
to their removal date — the exact opposite of the June mid-cap universe, whose
current-listing selection produced ICs that grew toward the selection date
(see scripts/midcap/purged_midcap.py).

Output: data/liquidcap/membership_sp500.parquet — one row per membership spell
(ticker, start, end), clipped to [FLOOR, today]. Tickers normalized to Tiingo
style ('.' -> '-').

Usage:
    PYTHONPATH=src python scripts/liquidcap/build_universe_sp500.py [--html FILE]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "liquidcap"
WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FLOOR = pd.Timestamp("2014-01-01")


def load_tables(html_fp: str | None):
    if html_fp:
        html = Path(html_fp).read_text(encoding="utf-8")
    else:
        r = requests.get(WIKI, headers={"User-Agent": "Mozilla/5.0 (research)"}, timeout=60)
        r.raise_for_status()
        html = r.text
        (DATA / "sp500_wiki_snapshot.html").write_text(html, encoding="utf-8")
    tables = pd.read_html(html)
    return tables[0], tables[1]


def norm(t) -> str | None:
    if not isinstance(t, str) or not t.strip():
        return None
    return t.strip().upper().replace(".", "-")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default=None, help="local snapshot instead of fetching")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)

    cur, ch = load_tables(args.html)
    today = pd.Timestamp.today().normalize()

    ch = ch.copy()
    ch.columns = ["_".join(c) if isinstance(c, tuple) else c for c in ch.columns]
    ch = ch.rename(columns={
        "Effective Date_Effective Date": "date",
        "Added_Ticker": "added", "Removed_Ticker": "removed"})
    ch["date"] = pd.to_datetime(ch["date"], errors="coerce")
    ch = ch.dropna(subset=["date"]).sort_values("date", ascending=False)

    # Walk backwards from the current membership, opening/closing spells.
    open_end: dict[str, pd.Timestamp] = {}
    spells: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for t in cur["Symbol"]:
        t = norm(t)
        if t:
            open_end[t] = today
    for _, row in ch.iterrows():
        d = row["date"]
        if d < FLOOR - pd.Timedelta(days=370):
            break  # spells still open at FLOOR get clipped below anyway
        a, r = norm(row.get("added")), norm(row.get("removed"))
        if a and a in open_end:
            spells.append((a, d, open_end.pop(a)))
        if r and r not in open_end:
            open_end[r] = d
    for t, end in open_end.items():
        spells.append((t, FLOOR, end))

    df = pd.DataFrame(spells, columns=["ticker", "start", "end"])
    df["start"] = df["start"].clip(lower=FLOOR)
    df = df[df["end"] > df["start"]].sort_values(["ticker", "start"]).reset_index(drop=True)

    removed_only = sorted(set(df.ticker) - {norm(t) for t in cur["Symbol"]})
    print(f"  spells: {len(df)}  unique tickers: {df.ticker.nunique()}")
    print(f"  current members: {cur.shape[0]}  removed-in-window: {len(removed_only)}")
    print(f"  window: {df.start.min().date()} -> {df.end.max().date()}")
    fp = DATA / "membership_sp500.parquet"
    df.to_parquet(fp, index=False)
    print(f"  saved {fp}")


if __name__ == "__main__":
    main()
