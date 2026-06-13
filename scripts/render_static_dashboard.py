"""Render the login shell to a standalone static HTML for GitHub Pages.

The published page ships with NO trade data: it logs in via Supabase Auth and
then fetches the render-ready `dashboard_view` (RLS authenticated-only) to paint
the dashboard client-side. So this just renders the shell with the embedded
public config (Supabase URL + anon key, Finnhub token, brand assets) and copies
the PWA manifest + icons.

Output: ./site/index.html  (+ .nojekyll, manifest, icons)

Usage:
    PYTHONPATH=src python scripts/render_static_dashboard.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: E402

from app.web import server  # noqa: E402


def main() -> None:
    ctx = server.shell_context(static_mode=True)

    tpl_dir = ROOT / "src" / "app" / "web" / "templates"
    env = Environment(
        loader=FileSystemLoader(str(tpl_dir)),
        autoescape=select_autoescape(["html"]),
    )
    html = env.get_template("dashboard.html").render(request=None, **ctx)

    if ctx["supabase_url"] and ctx["supabase_anon_key"]:
        print("  Login enabled (Supabase URL + anon key embedded)")
    else:
        print("  WARNING: no Supabase config — login will not work")
    if ctx["finnhub_token"]:
        print("  Live prices enabled (Finnhub token embedded)")

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
