"""Incremental small-cap OHLCV download from Polygon ("massive" client).

Extracted verbatim from the retired V2 research pipeline; used by
scripts/daily_pipeline.py::update_ohlcv.
"""
from __future__ import annotations

import pandas as pd


def download_ohlcv(aggs, tickers, train_start, predict_to, existing_ohlcv=None):
    """Download OHLCV bars incrementally — only fetches missing days.

    If existing_ohlcv is provided, determines the last available date per
    ticker and only downloads bars after that date.  New tickers get a
    full history download.
    """
    import time as _time

    all_dfs = []
    failed = []
    errors = 0
    skipped = 0
    downloaded = 0
    target_end = pd.Timestamp(predict_to)
    _dl_start = _time.monotonic()

    # Build per-ticker last-date index from existing data
    last_dates: dict[str, pd.Timestamp] = {}
    if existing_ohlcv is not None and not existing_ohlcv.empty:
        existing_ohlcv["date"] = pd.to_datetime(existing_ohlcv["date"])
        last_dates = existing_ohlcv.groupby("ticker")["date"].max().to_dict()

    for ticker in tickers:
        last = last_dates.get(ticker)

        # Already up-to-date? Skip only if we already have the target day's bar.
        # A 1-day tolerance here meant a run holding yesterday's bar would never
        # fetch today's — the daily cron silently captured no new data for the
        # most recent trading day (bug fixed 2026-06-12). predict_to already
        # encodes weekend/holiday handling, so an exact compare is safe.
        if last is not None and last >= target_end:
            skipped += 1
            continue

        # Determine from_date: day after last known, or full history
        if last is not None:
            from_date = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            label = "incremental"
        else:
            from_date = train_start
            label = "full"

        try:
            bars = aggs.get_custom_bars(
                ticker, from_date=from_date, to_date=predict_to, adjusted=True,
            )
        except Exception as e:
            # A per-ticker HTTP error (e.g. 403 after a plan downgrade, or a
            # transient 5xx) must NOT abort the whole daily run — skip and carry
            # on, exactly like the SPY download below already does. The run then
            # completes on whatever bars succeeded (or on the existing data).
            errors += 1
            failed.append(f"{ticker}(err)")
            if errors <= 3:
                print(f"    ✗ {ticker:6s} — download error: {str(e)[:90]}")
            continue

        if label == "full" and (not bars or len(bars) < 30):
            failed.append(f"{ticker}({len(bars) if bars else 0})")
            continue

        if bars:
            rows = [{
                "date": pd.Timestamp(b.trading_date),
                "ticker": b.ticker,
                "open": b.open, "high": b.high, "low": b.low,
                "close": b.close, "volume": b.volume,
                "vwap": b.vwap, "transactions": b.transactions,
            } for b in bars]
            all_dfs.append(pd.DataFrame(rows))
            downloaded += 1
            print(f"    ✓ {ticker:6s} — {len(bars):>4d} new bars  "
                  f"({bars[0].trading_date} → {bars[-1].trading_date})  [{label}]")
        else:
            downloaded += 1
            print(f"    · {ticker:6s} — 0 new bars (already current)")

        if downloaded > 0 and downloaded % 10 == 0:
            elapsed = _time.monotonic() - _dl_start
            total_to_download = len(tickers) - skipped
            pct = downloaded / total_to_download if total_to_download else 1
            eta_secs = (elapsed / pct - elapsed) if pct > 0 else 0
            eta_min = eta_secs / 60
            print(f"      ── {downloaded}/{total_to_download} descargados "
                  f"({elapsed/60:.1f} min) · ETA ~{eta_min:.0f} min ──")

    if skipped:
        print(f"    ⏩ {skipped} tickers already up-to-date (skipped)")
    if errors:
        print(f"\n  ⚠ {errors} tickers errored on download (HTTP/auth, e.g. 403 after a "
              f"plan downgrade) — ran with the bars that succeeded. If this is most of "
              f"the universe, the data source/plan needs attention.")
    if failed:
        print(f"\n  ⚠ Failed/insufficient: {', '.join(failed)}")

    new_data = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    # Merge new data with existing
    if existing_ohlcv is not None and not existing_ohlcv.empty and not new_data.empty:
        combined = pd.concat([existing_ohlcv, new_data], ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"])
        combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
        return combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    elif existing_ohlcv is not None and not existing_ohlcv.empty:
        return existing_ohlcv
    else:
        return new_data


# Business days of overlap re-pulled by the grouped download each run so a
# session left under-covered (a per-ticker backfill advancing the frontier before
# the grouped bar publishes) is topped up on a later run. Cheap: ~3 extra calls.
GROUPED_LOOKBACK_BDAYS = 3


def download_ohlcv_grouped(aggs, tickers, existing_ohlcv, predict_to):
    """Incremental OHLCV update via the grouped-daily endpoint.

    One call returns the WHOLE US market for a single date, so refreshing the
    latest session(s) for the entire universe costs ~1 call PER DAY instead of one
    call PER TICKER — the difference between ~12s and ~1h on the free tier (5/min).

    Re-pulls a short lookback window (not only sessions strictly after the stored
    frontier) so an under-covered session self-heals. Per-ticker stragglers and
    brand-new tickers that need deeper history are handled by download_ohlcv()
    separately. Returns the combined frame (existing + new), deduped, or existing
    unchanged.
    """
    import time as _time

    ticker_set = set(tickers)
    existing_ohlcv = existing_ohlcv.copy()
    existing_ohlcv["date"] = pd.to_datetime(existing_ohlcv["date"])
    have_dates = set(existing_ohlcv["date"].dt.date)
    last_date = existing_ohlcv["date"].max().date()
    target = pd.Timestamp(predict_to).date()

    # Re-pull a short lookback window instead of only sessions STRICTLY AFTER the
    # stored frontier. The per-ticker backfill (new/stale names) can advance the
    # stored max onto a session grouped hasn't covered yet: on 2026-07-01 that left
    # the whole universe missing except ECHO, and "strictly after max" would skip
    # that session forever. Overlapping the last few sessions + dedup(keep=last)
    # tops up any partial day once its grouped bar is finally published.
    start = min(pd.Timestamp(last_date),
                pd.Timestamp(target) - pd.tseries.offsets.BDay(GROUPED_LOOKBACK_BDAYS))
    days = [d.date() for d in pd.bdate_range(start, target)]
    if not days:
        print("  grouped: already current, nothing to fetch")
        return existing_ohlcv

    print(f"  grouped: {len(days)} session(s) {days[0]} → {days[-1]} "
          f"(1 call/day, whole market → filter {len(ticker_set)} tickers)")
    rows, errors, past_errors, got = [], 0, 0, 0
    t0 = _time.monotonic()
    for d in days:
        try:
            bars = aggs.get_grouped_daily(d, adjusted=True)
        except Exception as e:
            errors += 1
            # A 403 on TODAY's in-progress bar (d == target) is benign: free Polygon
            # only publishes a session's grouped bar hours after the close (next
            # morning), so the current day legitimately isn't available yet. Ditto a
            # session we ALREADY have (re-pulled by the lookback window) — a transient
            # error there must not tank the run. Only a PAST session we DON'T yet hold
            # erroring is a real failure — count those apart to fail loudly on them.
            if d < target and d not in have_dates:
                past_errors += 1
            if errors <= 3:
                print(f"    ✗ {d} — grouped error: {str(e)[:90]}")
            continue
        if not bars:
            continue  # weekend/holiday → no session
        got += 1
        for b in bars:
            if b.ticker in ticker_set:
                rows.append({
                    "date": pd.Timestamp(b.trading_date), "ticker": b.ticker,
                    "open": b.open, "high": b.high, "low": b.low,
                    "close": b.close, "volume": b.volume,
                    "vwap": b.vwap, "transactions": b.transactions,
                })
    if errors:
        print(f"    ⚠ {errors} session(s) errored (HTTP/auth) — used the rest.")
    print(f"    ✓ {got} session(s), {len(rows):,} ticker-rows ({_time.monotonic()-t0:.0f}s)")

    # FAIL LOUD: if a PAST session (one that should already be published) errored,
    # do NOT return silently with stale data — that produced a GREEN run on
    # 2026-06-29 that committed no new bars. Raise so the GitHub run goes red.
    if past_errors > 0:
        raise RuntimeError(
            f"grouped: {past_errors} PAST session(s) errored and were not fetched — "
            f"data NOT updated past {last_date}. Likely an auth/plan/rate issue or the "
            f"free tier hasn't published yet. Failing loudly instead of committing stale data."
        )

    if not rows:
        return existing_ohlcv
    combined = pd.concat([existing_ohlcv, pd.DataFrame(rows)], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
    return combined.sort_values(["ticker", "date"]).reset_index(drop=True)
