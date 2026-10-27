"""What every paper book does after it has processed its sessions — in one place.

Each book (liquidcap, btc_trend, bounce, illiquid) has its own engine, but they all
publish the same way: state -> Supabase, signals, NAV points, the render-ready
dashboard view, and the official closes into `live_prices` (so the web matches
the run until the live worker takes over).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.data import supabase_store


def publish_book(strategy: str, state: dict, ohlcv: pd.DataFrame, pt_dir: Path, *,
                 nav: dict[str, float] | None = None, nav_date: str | None = None,
                 signals: list[dict] | None = None, bench: pd.DataFrame | None = None,
                 bench2: tuple[str, pd.DataFrame] | None = None,
                 write_state: bool = True) -> dict | None:
    """Persist + publish one book; returns the dashboard view (None if not buildable).

    NAV: pass `nav` (session -> book value, e.g. every session caught up in this run)
    and it is written BEFORE the view, so the chart already includes it. Otherwise
    one point is written for `nav_date` (default: the latest bar) from the view's
    total value — that point shows on the chart from the next run.
    """
    from app.web import dashboard_data

    if write_state:
        supabase_store.write_state(strategy, state)
    if signals:
        supabase_store.upsert_signals(strategy, signals)
    for d, v in (nav or {}).items():
        supabase_store.upsert_nav(strategy, d, round(float(v), 2))
    pt_dir.mkdir(parents=True, exist_ok=True)
    view = dashboard_data.build_view(ohlcv, pt_dir, strategy, bench=bench, bench2=bench2)
    if view is None:
        return None
    supabase_store.write_dashboard_view(strategy, view)
    if not nav:
        d = nav_date or str(pd.to_datetime(ohlcv["date"]).max().date())
        supabase_store.upsert_nav(strategy, d, float(view["paper"]["total_value"]))
    supabase_store.upsert_live_prices(
        {p["ticker"]: p["current_price"] for p in view["paper"].get("positions", [])})
    return view
