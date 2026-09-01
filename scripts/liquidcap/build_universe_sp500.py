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


def _from_changes(cur, ch, today) -> pd.DataFrame:
    """Full point-in-time reconstruction from Wikipedia's changes table."""
    ch = ch.copy()
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
    return df[df["end"] > df["start"]].sort_values(["ticker", "start"]).reset_index(drop=True)


def _incremental(cur, today, fp) -> pd.DataFrame:
    """Wikipedia dropped the "Selected changes" table (~2026), so a full rebuild
    is impossible. Refresh the committed membership incrementally: extend the
    open spell of every ticker still current to `today`, freeze the open spell of
    a departed ticker at the last refresh date, add a fresh spell for new
    entrants, and preserve every historical (removed) spell — anti-survivorship
    intact. Needs the committed membership file to exist."""
    old = pd.read_parquet(fp)
    old["start"] = pd.to_datetime(old["start"]); old["end"] = pd.to_datetime(old["end"])
    old_max = old["end"].max()
    new_current = {norm(t) for t in cur["Symbol"] if norm(t)}
    rows = []
    open_tickers = set()
    for r in old.itertuples(index=False):
        if r.end >= old_max:                    # currently-open spell
            open_tickers.add(r.ticker)
            end = today if r.ticker in new_current else old_max   # extend or freeze
            rows.append((r.ticker, r.start, end))
        else:                                   # historical (removed) — preserve verbatim
            rows.append((r.ticker, r.start, r.end))
    for t in sorted(new_current - open_tickers):   # new entrants (join date unknown -> last refresh)
        rows.append((t, old_max, today))
    df = pd.DataFrame(rows, columns=["ticker", "start", "end"])
    return df[df["end"] > df["start"]].sort_values(["ticker", "start"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default=None, help="local snapshot instead of fetching")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)

    cur, ch = load_tables(args.html)
    today = pd.Timestamp.today().normalize()
    fp = DATA / "membership_sp500.parquet"

    ch = ch.copy()
    ch.columns = ["_".join(c) if isinstance(c, tuple) else c for c in ch.columns]
    ch = ch.rename(columns={
        "Effective Date_Effective Date": "date",
        "Added_Ticker": "added", "Removed_Ticker": "removed"})

    if "date" in ch.columns:
        df = _from_changes(cur, ch, today)
    elif fp.exists():
        print("  WARN: no changes table on Wikipedia — incremental refresh from committed membership")
        df = _incremental(cur, today, fp)
    else:
        raise SystemExit("  ABORT: no Wikipedia changes table AND no committed membership to refresh from")

    removed_only = sorted(set(df.ticker) - {norm(t) for t in cur["Symbol"]})
    print(f"  spells: {len(df)}  unique tickers: {df.ticker.nunique()}")
    print(f"  current members: {cur.shape[0]}  removed-in-window: {len(removed_only)}")
    print(f"  window: {df.start.min().date()} -> {df.end.max().date()}")
    df.to_parquet(fp, index=False)
    print(f"  saved {fp}")


if __name__ == "__main__":
    main()
