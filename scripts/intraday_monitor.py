"""Intraday trailing stop monitor via Polygon snapshots (15-min delay).

Uses get_full_market_snapshot() to check current prices of open positions
against their trailing stops. If a stop is hit, marks position for exit.

Designed to run via cron every 30 min during market hours (9:45–15:45 ET):
    */30 10-15 * * 1-5  cd /path/to/SCAI && DYLD_LIBRARY_PATH=.local/lib PYTHONPATH=src .venv/bin/python scripts/intraday_monitor.py

Uses 1 Polygon API call per run (full market snapshot filtered to open tickers).
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from app.data.massive.client import MassiveClient
from app.data.massive.snapshots import SnapshotsAPI
from app.paper_trading import PaperTrader
from app.utils import get_logger

log = get_logger(__name__)

PORTFOLIO_PATH = "data/paper_trading/portfolio.json"
INTRADAY_LOG = "data/paper_trading/intraday_log.jsonl"

ET = ZoneInfo("America/New_York")


def _market_open() -> bool:
    """Check if US market is likely open (rough check, no holiday calendar)."""
    now_et = datetime.now(ET)
    if now_et.weekday() >= 5:
        return False
    market_open = now_et.replace(hour=9, minute=30, second=0)
    market_close = now_et.replace(hour=16, minute=0, second=0)
    return market_open <= now_et <= market_close


def _log_event(event: dict) -> None:
    """Append event to intraday log."""
    p = Path(INTRADAY_LOG)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")


def run_intraday_check(force: bool = False) -> None:
    """Check trailing stops for all open positions using Polygon snapshots."""
    if not force and not _market_open():
        print("Market closed — skipping. Use --force to override.")
        return

    trader = PaperTrader.load_or_create(PORTFOLIO_PATH)
    positions = trader.state.positions

    if not positions:
        print("No open positions — nothing to monitor.")
        return

    tickers = [p["ticker"] for p in positions]
    print(f"Checking {len(tickers)} positions: {', '.join(tickers)}")

    # 1 API call: snapshot for all open tickers
    client = MassiveClient()
    snaps_api = SnapshotsAPI(client)
    snapshots = snaps_api.get_full_market_snapshot(tickers=tickers)

    snap_map = {s.ticker: s for s in snapshots}
    now_str = datetime.now(timezone.utc).isoformat()

    alerts = []
    updates = []

    for pos in positions:
        ticker = pos["ticker"]
        snap = snap_map.get(ticker)
        if not snap or not snap.last_trade_price:
            print(f"  {ticker}: no snapshot data — skipping")
            continue

        price = snap.last_trade_price
        high = pos["high_price"]
        trail_pct = pos["trailing_stop_pct"]
        entry = pos["entry_price"]

        # Update watermark if new high
        new_high = max(high, price)
        if new_high > high:
            pos["high_price"] = round(new_high, 4)
            updates.append(ticker)

        # Check trailing stop
        trail_trigger = new_high * (1 - trail_pct)
        pnl_pct = (price / entry - 1) * 100

        status = "OK"
        if price <= trail_trigger:
            status = "STOP_HIT"
            alerts.append({
                "ticker": ticker,
                "price": price,
                "trail_trigger": round(trail_trigger, 4),
                "high_price": round(new_high, 4),
                "entry_price": entry,
                "pnl_pct": round(pnl_pct, 2),
            })

        print(f"  {ticker}: ${price:.2f} (entry ${entry:.2f}, "
              f"P&L {pnl_pct:+.1f}%, high ${new_high:.2f}, "
              f"trail@ ${trail_trigger:.2f}) → {status}")

    # Save updated watermarks
    if updates:
        trader.save()
        print(f"\n  Watermarks updated: {', '.join(updates)}")

    # Log & alert
    event = {
        "timestamp": now_str,
        "positions_checked": len(positions),
        "alerts": len(alerts),
        "watermarks_updated": updates,
    }

    if alerts:
        print(f"\n  ⚠ {len(alerts)} TRAILING STOP(S) HIT:")
        for a in alerts:
            print(f"    {a['ticker']}: ${a['price']:.2f} ≤ trail ${a['trail_trigger']:.2f} "
                  f"(P&L {a['pnl_pct']:+.1f}%)")
            event[f"stop_{a['ticker']}"] = a
        print("\n  → Positions will be closed at next daily pipeline run.")
        print("    To close immediately, run daily_pipeline.py now.")

        # Mark stopped positions so daily pipeline knows
        stop_file = Path("data/paper_trading/intraday_stops.json")
        existing = json.loads(stop_file.read_text()) if stop_file.exists() else []
        for a in alerts:
            if a["ticker"] not in [e["ticker"] for e in existing]:
                existing.append({
                    "ticker": a["ticker"],
                    "stop_price": a["price"],
                    "timestamp": now_str,
                })
        stop_file.write_text(json.dumps(existing, indent=2, default=str))
    else:
        print("\n  ✓ All positions within trailing stop range.")

    _log_event(event)


if __name__ == "__main__":
    force = "--force" in sys.argv
    run_intraday_check(force=force)
