"""Small-cap daily data update (incremental Polygon download)."""
from __future__ import annotations

import os
from datetime import timedelta

import pandas as pd

from app.utils import get_logger

log = get_logger(__name__)


def update_ohlcv(cfg, predict_to: str) -> pd.DataFrame:
    """Download only the missing OHLCV bars since last update."""
    from app.data.massive import AggregatesAPI, MassiveClient
    from app.data.store.parquet_store import ParquetStore

    store = ParquetStore()
    ohlcv = store.read("ohlcv_smallcap")
    uni_df = store.read("smallcap_universe")
    # Only download updates for active tickers (delisted ones won't have new bars)
    active_mask = (uni_df["active"] == True if "active" in uni_df.columns
                   else pd.Series(True, index=uni_df.index))
    tickers = uni_df[active_mask]["ticker"].tolist()

    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    last_date = ohlcv["date"].max().date()
    predict_to_date = pd.Timestamp(predict_to).date()

    # Check for tickers lagging behind the global max date
    last_per_ticker = ohlcv.groupby("ticker")["date"].max()
    stale_tickers = last_per_ticker[last_per_ticker < ohlcv["date"].max() - pd.Timedelta(days=2)]
    n_stale = len(stale_tickers)

    # Skip the download only if we already hold the target day's bar. Comparing
    # against (today - 1) meant a run holding yesterday's bar short-circuited and
    # never fetched today's — the cron captured no data for the day (fixed
    # 2026-06-12). predict_to already encodes weekend/holiday handling.
    if last_date >= predict_to_date and n_stale == 0:
        print(f"  OHLCV already current ({last_date})")
        return ohlcv

    if n_stale > 0:
        print(f"  ⚠ {n_stale} tickers desfasados: {', '.join(stale_tickers.index[:10])}")

    print(f"  Updating OHLCV: {last_date} → {predict_to} ({len(tickers)} tickers)")

    # Rate from MASSIVE_CALLS_PER_MINUTE (default 50, paid plan). When the
    # Polygon plan is downgraded to free, set this env to 5 — no code change
    # needed; the daily download just paces slower (~12s/ticker, job still
    # completes within the Actions 6h limit).
    cpm = int(os.environ.get("MASSIVE_CALLS_PER_MINUTE", "50"))
    client = MassiveClient(calls_per_minute=cpm)
    aggs = AggregatesAPI(client)

    # Import the downloaders from the main pipeline
    from app.data.smallcap_ohlcv import download_ohlcv, download_ohlcv_grouped

    # Per-ticker pull for the small set grouped_daily can't cover from the global
    # frontier: brand-new tickers (no bars) + individually stale ones (own last bar
    # behind the rest, e.g. post-halt). Restricted to ACTIVE tickers — the 728
    # delisted names are perpetually "stale" but must never be re-fetched. Done
    # first so each one's own gap is filled, then grouped tops up recent sessions.
    new_tickers = [t for t in tickers if t not in last_per_ticker.index]
    catch_up = sorted(set(new_tickers) | (set(stale_tickers.index) & set(tickers)))
    if catch_up:
        print(f"  per-ticker backfill: {len(catch_up)} new/stale "
              f"({', '.join(catch_up[:10])}{'…' if len(catch_up) > 10 else ''})")
        ohlcv = download_ohlcv(
            aggs, catch_up,
            train_start=(last_date - timedelta(days=5)).isoformat(),
            predict_to=predict_to,
            existing_ohlcv=ohlcv,
        )

    # Bulk incremental for the WHOLE universe via grouped_daily: ~1 call/session
    # (whole market, filtered to our tickers) instead of one call per ticker —
    # ~12s vs ~1h on the free Polygon tier (5/min). See download_ohlcv_grouped.
    ohlcv = download_ohlcv_grouped(aggs, tickers, ohlcv, predict_to)

    # Also update SPY
    existing_spy = store.read("smallcap_spy") if store.exists("smallcap_spy") else None
    spy_from = (last_date + timedelta(days=1)).isoformat()
    try:
        spy_bars = aggs.get_custom_bars("SPY", from_date=spy_from,
                                         to_date=predict_to, adjusted=True)
    except Exception as e:
        log.warning("spy_download_failed", error=str(e))
        spy_bars = []
    if spy_bars:
        spy_rows = [{"date": pd.Timestamp(b.trading_date), "ticker": "SPY",
                     "open": b.open, "high": b.high, "low": b.low,
                     "close": b.close, "volume": b.volume} for b in spy_bars]
        spy_new = pd.DataFrame(spy_rows)
        if existing_spy is not None and not existing_spy.empty:
            existing_spy["date"] = pd.to_datetime(existing_spy["date"])
            spy_df = pd.concat([existing_spy, spy_new], ignore_index=True)
            spy_df = spy_df.drop_duplicates(subset=["date"], keep="last")
        else:
            spy_df = spy_new
        store.write("smallcap_spy", spy_df)

    # Also update IWM (Russell 2000) — the investable benchmark for the small-cap
    # books on the dashboard chart/alpha (SPY stays as the feature market-proxy).
    # First run self-backfills the full history to match SPY's span; then incremental.
    existing_iwm = store.read("smallcap_iwm") if store.exists("smallcap_iwm") else None
    if existing_iwm is not None and not existing_iwm.empty:
        existing_iwm["date"] = pd.to_datetime(existing_iwm["date"])
        iwm_from = (existing_iwm["date"].max() + timedelta(days=1)).date().isoformat()
    else:
        iwm_from = "2021-06-09"  # backfill to match smallcap_spy's start
    try:
        iwm_bars = aggs.get_custom_bars("IWM", from_date=iwm_from,
                                        to_date=predict_to, adjusted=True)
    except Exception as e:
        log.warning("iwm_download_failed", error=str(e))
        iwm_bars = []
    if iwm_bars:
        iwm_rows = [{"date": pd.Timestamp(b.trading_date), "ticker": "IWM",
                     "open": b.open, "high": b.high, "low": b.low,
                     "close": b.close, "volume": b.volume} for b in iwm_bars]
        iwm_new = pd.DataFrame(iwm_rows)
        if existing_iwm is not None and not existing_iwm.empty:
            iwm_df = pd.concat([existing_iwm, iwm_new], ignore_index=True)
            iwm_df = iwm_df.drop_duplicates(subset=["date"], keep="last")
        else:
            iwm_df = iwm_new
        store.write("smallcap_iwm", iwm_df)

    client.close()
    store.write("ohlcv_smallcap", ohlcv)
    print(f"  ✓ OHLCV: {len(ohlcv):,} rows, {ohlcv['date'].max().date()}")

    # Data-gap watch: name ACTIVE tickers chronically behind the panel (missing the
    # last >=2 sessions). A transient 1-day miss self-heals via the stale catch-up,
    # so it's NOT flagged; a ticker stuck >=2 sessions is a real gap. The daily
    # "316/317" count alone can't tell transient noise from a broken ticker — this
    # surfaces the NAMES + how many sessions each is behind in the log/Telegram.
    sessions = sorted(pd.to_datetime(ohlcv["date"]).dt.normalize().unique())
    active_set = set(tickers)
    if len(sessions) >= 3:
        cutoff = sessions[-2]  # last bar older than this => missing >=2 sessions
        last_bar = ohlcv[ohlcv["ticker"].isin(active_set)].groupby("ticker")["date"].max()
        stale = last_bar[last_bar < cutoff]
        if len(stale):
            behind = sorted(((tk, int(sum(s > d for s in sessions))) for tk, d in stale.items()),
                            key=lambda x: -x[1])
            names = ", ".join(f"{tk}(-{n})" for tk, n in behind[:15])
            print(f"  ⚠ {len(stale)}/{len(active_set)} active tickers STALE >=2 sessions "
                  f"[ticker(-sessions_behind)]: {names}{' ...' if len(stale) > 15 else ''}")
        else:
            print(f"  ✓ all {len(active_set)} active tickers current (<=1 session behind)")
    return ohlcv

