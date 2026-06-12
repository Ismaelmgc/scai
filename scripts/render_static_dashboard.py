"""Render the dashboard to a standalone static HTML for GitHub Pages.

Reuses the exact data-loading logic from `app.web.server` so the static
snapshot matches the live dashboard. The live-only buttons (run pipeline,
refresh prices) are hidden via `static_mode=True`.

Output: ./site/index.html  (+ .nojekyll so Pages serves files verbatim)

Usage:
    PYTHONPATH=src python scripts/render_static_dashboard.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: E402

from app.data import supabase_store  # noqa: E402
from app.web import server  # noqa: E402


def main() -> None:
    ohlcv = server._load_ohlcv()
    paper = server._load_paper_trading(ohlcv, server.PAPER_TRADING_DIR)
    paper_adaptive = server._load_paper_trading(
        ohlcv, server.PAPER_TRADING_ADAPTIVE_DIR, adaptive_stop=True
    )
    signals = server._load_signal_history(server.PAPER_TRADING_DIR)
    signals_adaptive = server._load_signal_history(server.PAPER_TRADING_ADAPTIVE_DIR)
    data_info = server._get_data_freshness(ohlcv)

    # Anon (publishable) creds embedded so the Pages snapshot can poll Supabase
    # live (Phase 2b). Empty when unconfigured → live-refresh script is skipped.
    supabase_url, supabase_anon_key = supabase_store.public_config()

    tpl_dir = ROOT / "src" / "app" / "web" / "templates"
    env = Environment(
        loader=FileSystemLoader(str(tpl_dir)),
        autoescape=select_autoescape(["html"]),
    )
    html = env.get_template("dashboard.html").render(
        request=None,
        paper=paper,
        paper_adaptive=paper_adaptive,
        signals=signals,
        signals_adaptive=signals_adaptive,
        data_info=data_info,
        now=datetime.now().strftime("%Y-%m-%d %H:%M"),
        static_mode=True,
        supabase_url=supabase_url,
        supabase_anon_key=supabase_anon_key,
    )
    if supabase_url and supabase_anon_key:
        print("  Live refresh enabled (Supabase anon key embedded)")
    else:
        print("  Live refresh disabled (no Supabase anon config)")

    out_dir = ROOT / "site"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")
    print(f"  Wrote {out_dir / 'index.html'} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
