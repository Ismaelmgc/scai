"""Download FINRA short-interest history for the universe → data/short_interest.parquet.

Short interest is a classic, price-orthogonal small-cap signal (high SI / high
days-to-cover predicts weaker forward returns). FINRA collects it twice a month
(settlement ~15th and month-end); we pull it via the Massive/Polygon endpoint
`/stocks/v1/short-interest` (fields: settlement_date, short_interest,
avg_daily_volume, days_to_cover) while the paid plan is active, so the backtest
has full 2021→present history. Coverage check: 100% of the active universe.

PIT NOTE: only `settlement_date` is provided. FINRA disseminates ~8 business
days AFTER the settlement date, so the feature builder must lag availability
(settlement + buffer) — never treat settlement_date as the as-of date here.

Usage:
    PYTHONPATH=src python scripts/download_short_interest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from app.data.massive import MassiveClient  # noqa: E402

OUT = ROOT / "data" / "short_interest.parquet"
FIELDS = ("ticker", "settlement_date", "short_interest", "avg_daily_volume", "days_to_cover")


def main() -> None:
    uni = pd.read_parquet(ROOT / "data/processed/smallcap_universe.parquet")
    tickers = sorted(uni["ticker"].astype(str).unique())
    print(f"Universe: {len(tickers)} tickers. Pulling short interest history...")

    client = MassiveClient()
    rows: list[dict] = []
    ok = miss = err = 0
    for i, tk in enumerate(tickers, 1):
        try:
            res = client.get_all_pages(
                "/stocks/v1/short-interest",
                {"ticker": tk, "sort": "settlement_date.asc", "limit": 50000},
                max_pages=5,
            )
        except Exception as e:  # noqa: BLE001 — one bad ticker shouldn't kill the run
            err += 1
            print(f"  [{i}/{len(tickers)}] {tk}: ERROR {str(e)[:60]}")
            continue
        if not res:
            miss += 1
            continue
        ok += 1
        for r in res:
            rows.append({k: r.get(k) for k in FIELDS})
        if i % 100 == 0:
            print(f"  [{i}/{len(tickers)}] data:{ok} empty:{miss} err:{err} rows:{len(rows)}")

    df = pd.DataFrame(rows, columns=list(FIELDS))
    df = df.dropna(subset=["settlement_date", "short_interest"])
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    df = df.sort_values(["ticker", "settlement_date"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    lo, hi = df["settlement_date"].min().date(), df["settlement_date"].max().date()
    print(f"\nSaved {len(df):,} rows for {df['ticker'].nunique()} tickers -> {OUT}")
    print(f"  Date range: {lo} -> {hi}")
    print(f"  tickers with data: {ok} | empty: {miss} | errors: {err}")


if __name__ == "__main__":
    main()
