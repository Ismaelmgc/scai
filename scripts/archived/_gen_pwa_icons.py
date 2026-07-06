"""One-off: generate PWA icons (180/192/512) from the gold-bull favicon.

Dark safe-zone background (#0a0c12, matches the dashboard theme) with the bull
centred and padded so maskable launchers don't clip it. Run locally; the PNGs
are committed and embedded into the static Pages snapshot by render_static_dashboard.
"""
from pathlib import Path

from PIL import Image

STATIC = Path(__file__).resolve().parents[1] / "src" / "app" / "web" / "static"
BG = (10, 12, 18, 255)

src = Image.open(STATIC / "scai_favicon.png").convert("RGBA")
bbox = src.getbbox()
bull = src.crop(bbox) if bbox else src


def make(size: int, pad_frac: float) -> Image.Image:
    canvas = Image.new("RGBA", (size, size), BG)
    inner = int(size * (1 - 2 * pad_frac))
    w, h = bull.size
    scale = min(inner / w, inner / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    b = bull.resize((nw, nh), Image.LANCZOS)
    canvas.alpha_composite(b, ((size - nw) // 2, (size - nh) // 2))
    return canvas.convert("RGB")


make(180, 0.14).save(STATIC / "icon-180.png")  # apple-touch
make(192, 0.18).save(STATIC / "icon-192.png")  # maskable safe-zone
make(512, 0.18).save(STATIC / "icon-512.png")  # maskable safe-zone
print("icons written:", [p.name for p in STATIC.glob("icon-*.png")])
