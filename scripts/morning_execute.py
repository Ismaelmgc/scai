"""Morning execution: fill pending BUY signals at today's open, intraday.

Previously a queued BUY was only filled when the post-close pipeline ran. Since
the entry price is the day's OPEN (already fixed at 09:30 ET), waiting until the
close changes nothing about the price — only when it's recorded. This job fills
the pending signals shortly after the open so the dashboard shows the positions
in the morning.

Scope is deliberately minimal and consistent with the backtest:
- ONLY `execute_pending` at the open (Finnhub REST `/quote` open). It does NOT
  run exits, generate signals, or advance the day index / last_update — the EOD
  pipeline still does all of that. Calling only execute_pending leaves the state
  identical to the all-at-EOD flow (entry_day_idx unchanged; the EOD
  update_positions still increments the day index and checks exits).
- Idempotent: once filled, pending is empty, so a second run (or the EOD
  pipeline) is a no-op. A ticker without a Finnhub open stays pending → the EOD
  pipeline fills it from the official Polygon open (safe fallback).

Exits stay close-based on purpose: a 16-fold backtest (2026-06-15) showed
intraday exits roughly halve the return and crater Sharpe (2.79→1.15) — noisy
small-cap wicks shake you out. So this job touches entries only.

Run (CI, weekdays after the open):
    PYTHONPATH=src python scripts/morning_execute.py
Dry run (no Supabase writes):
    PYTHONPATH=src python scripts/morning_execute.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from app.data import supabase_store  # noqa: E402
from app.data.free_sources import finnhub  # noqa: E402
from app.paper_trading import PaperTrader  # noqa: E402
from app.utils import notify_telegram  # noqa: E402
from app.web import dashboard_data  # noqa: E402

# (strategy, pt_dir, adaptive_stop, ohlcv_path)  ohlcv_path=None -> small-cap parquet.
# Adaptive-Stop book retired 2026-08-27 (poor live performance).
# Liquidcap left the open-fill 2026-08-27: it now enters at close(S) (MOC-sim) in
# its own nightly job, so there is nothing to fill at the open.
STRATEGIES = [
    ("baseline", dashboard_data.PAPER_TRADING_DIR, False, None),
]


def _ohlcv_for(path: Path | None) -> pd.DataFrame:
    """Committed EOD panel used to mark the view. Small-cap default, or the
    liquidcap S&P 500 panel when a path is given (may be absent in CI until the
    daily job has committed it — then the view keeps its last Supabase state)."""
    if path is None:
        return dashboard_data._load_ohlcv()
    if not path.exists():
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close"])
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _market_open_et(now: datetime) -> bool:
    et = now.astimezone(ZoneInfo("America/New_York"))
    if et.weekday() >= 5:
        return False
    mins = et.hour * 60 + et.minute
    return 9 * 60 + 30 <= mins <= 16 * 60


def _price_frame(tickers: list[str], today_ts: pd.Timestamp) -> pd.DataFrame:
    """Build a one-day OHLCV-ish frame from Finnhub quotes: `open` for the fill,
    `close` (= current price) to mark held positions, large volume (no cap).

    Holiday guard (2026-07-03 incident): on a market holiday Finnhub serves the
    LAST session's quote with a truthy `open` — filling at it records
    yesterday's open as today's entry. Only accept quotes whose timestamp is
    from TODAY (ET); stale ones are dropped so the signal stays pending and the
    EOD pipeline fills it from the official Polygon open of the next session.
    """
    quotes = finnhub.get_quotes(tickers)
    rows = []
    for t, q in quotes.items():
        if not q.get("open"):  # need a real open to fill at
            continue
        ts = q.get("ts", 0)
        q_date = (pd.Timestamp(ts, unit="s", tz="UTC")
                  .tz_convert(ZoneInfo("America/New_York")).date() if ts else None)
        if q_date != today_ts.date():
            print(f"    {t}: stale quote ({q_date or 'no ts'}) — market closed "
                  f"today? keeping pending")
            continue
        rows.append({"date": today_ts, "ticker": t,
                     "open": q["open"], "close": q["price"], "volume": 1e12})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill pending BUYs at the open")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan only; no Supabase writes")
    parser.add_argument("--force", action="store_true",
                        help="Run even if the market is closed (testing)")
    args = parser.parse_args()

    now = datetime.now(ZoneInfo("America/New_York"))
    if not _market_open_et(now) and not args.force:
        print(f"  Market closed (ET {now:%Y-%m-%d %H:%M}). Nothing to do.")
        return
    today = now.date().isoformat()
    today_ts = pd.Timestamp(today)

    if not supabase_store.is_configured():
        print("  Supabase not configured — aborting.")
        return

    # Collect fills across strategies (dedupe by ticker) so we send ONE Telegram
    # message with entry prices. Each strategy loads its own EOD panel below.
    all_fills: dict[str, float] = {}

    for strat, pt_dir, adaptive, ohlcv_path in STRATEGIES:
        state = supabase_store.read_state(strat)
        if state is None:
            print(f"  [{strat}] no state in Supabase — skip")
            continue
        ohlcv = _ohlcv_for(ohlcv_path)  # committed EOD panel, for the view

        p_path = pt_dir / "portfolio.json"
        p_path.parent.mkdir(parents=True, exist_ok=True)
        p_path.write_text(json.dumps(state))  # hydrate local file
        pt = PaperTrader.load_or_create(
            str(p_path), initial_capital=1000.0, max_positions=8,
            holding_period_days=20, adaptive_stop=adaptive, profit_target=0.40,
        )

        pending = [s["ticker"] for s in pt.state.pending_signals]
        if not pending:
            print(f"  [{strat}] no pending signals — skip")
            continue

        held = [p["ticker"] for p in pt.state.positions]
        prices = _price_frame(sorted(set(pending + held)), today_ts)
        have = set(prices["ticker"]) if not prices.empty else set()
        missing = [t for t in pending if t not in have]

        if args.dry_run:
            print(f"  [{strat}] DRY: would fill {sorted(set(pending) & have)} "
                  f"at open; no open for {missing} (stay pending)")
            continue

        entered = pt.execute_pending(prices, today)
        pt.save()
        supabase_store.write_state(strat, asdict(pt.state))
        # Refresh the view so the just-filled BUY shows as an OPEN position, not
        # stuck "pending". Mark with the EOD panel when present; otherwise with the
        # live Finnhub quotes we already fetched (liquidcap's ohlcv_sp500 panel is
        # gitignored → ABSENT here). Marking with `prices` (live) instead of the
        # empty panel avoids the degraded view (every position at its entry price,
        # P/L 0%). Only skip if we have NO prices at all. The 03:00 daily job later
        # rewrites the view from the full EOD panel with true day-change references.
        mark = ohlcv if not ohlcv.empty else prices
        if not mark.empty:
            view = dashboard_data.build_view(mark, pt_dir, adaptive, strategy=strat)
            if view is not None:
                supabase_store.write_dashboard_view(strat, view)
        else:
            print(f"  [{strat}] no prices to mark — view left to the daily job")
        print(f"  [{strat}] filled {entered or '[]'} at open"
              + (f"; still pending (no open): {missing}" if missing else ""))
        for tk in entered or []:
            pos = next((p for p in pt.state.positions if p["ticker"] == tk), None)
            if pos:
                all_fills[tk] = float(pos["entry_price"])

    # One Telegram alert per fill event, with the actual entry (open) price.
    if all_fills:
        lines = "\n".join(f"  • <b>{t}</b> @ ${p:.2f}" for t, p in sorted(all_fills.items()))
        notify_telegram(f"🟢 <b>Señales ejecutadas al open</b> — {today}\n{lines}")


if __name__ == "__main__":
    main()
