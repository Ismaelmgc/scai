"""Render the dashboard to a standalone static HTML for GitHub Pages.

Reuses the exact data-loading logic from `app.web.server` so the static
snapshot matches the live dashboard. The live-only buttons (run pipeline,
refresh prices) are hidden via `static_mode=True`.

Output: ./site/index.html  (+ .nojekyll so Pages serves files verbatim)

Usage:
    PYTHONPATH=src python scripts/render_static_dashboard.py
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: E402

from app.data import supabase_store  # noqa: E402
from app.data.free_sources import finnhub  # noqa: E402
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

    # Finnhub token embedded so the snapshot can stream live prices over the
    # Finnhub WebSocket (WebSockets aren't subject to CORS). Empty → no live prices.
    finnhub_token = finnhub.public_token()

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
        finnhub_token=finnhub_token,
        logo=server._asset_data_uri(server.LOGO_PATH),
        favicon=server._asset_data_uri(server.FAVICON_PATH),
    )
    if supabase_url and supabase_anon_key:
        print("  Live refresh enabled (Supabase anon key embedded)")
    else:
        print("  Live refresh disabled (no Supabase anon config)")
    if finnhub_token:
        print("  Live prices enabled (Finnhub token embedded)")
    else:
        print("  Live prices disabled (no FINNHUB_TOKEN)")

    out_dir = ROOT / "site"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")
    print(f"  Wrote {out_dir / 'index.html'} ({len(html):,} bytes)")

    # PWA: copy icons + write the web manifest alongside index.html so the Pages
    # site is installable on a phone home screen.
    static_dir = ROOT / "src" / "app" / "web" / "static"
    icons = []
    for name in ("icon-180.png", "icon-192.png", "icon-512.png"):
        src_icon = static_dir / name
        if src_icon.exists():
            shutil.copyfile(src_icon, out_dir / name)
            icons.append(name)
    manifest = {
        "name": "SCAI",
        "short_name": "SCAI",
        "description": "Dashboard de paper trading — small-caps US",
        "start_url": "./",
        "scope": "./",
        "display": "standalone",
        "background_color": "#0a0c12",
        "theme_color": "#0a0c12",
        "icons": [
            {"src": "icon-192.png", "sizes": "192x192",
             "type": "image/png", "purpose": "any maskable"},
            {"src": "icon-512.png", "sizes": "512x512",
             "type": "image/png", "purpose": "any maskable"},
        ],
    }
    (out_dir / "manifest.webmanifest").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  PWA: manifest + {len(icons)} icons copied ({', '.join(icons)})")


if __name__ == "__main__":
    main()
