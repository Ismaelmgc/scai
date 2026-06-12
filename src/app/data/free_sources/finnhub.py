"""Finnhub connector — free real-time US stock quotes.

Used as the free replacement for Polygon snapshots (paid) for live prices:
- Server-side REST `/quote` (the intraday trailing-stop monitor).
- The dashboard streams live prices client-side via Finnhub's WebSocket
  (``wss://ws.finnhub.io``); ``public_token`` exposes the token so the static
  Pages snapshot can open that socket (the token is embedded in public HTML,
  same trust model as the Supabase anon key).

Finnhub does NOT send CORS headers, so the REST endpoints here are for
server-side use only; the browser uses the WebSocket (not subject to CORS).

The token comes from the environment (GitHub Actions secret) or, locally, from
the project ``.env`` (unprefixed ``FINNHUB_TOKEN``). Absent token → no-ops.
"""
from __future__ import annotations

import os
from pathlib import Path

import httpx

from app.utils import get_logger

log = get_logger(__name__)

_ROOT = Path(__file__).resolve().parents[4]
_BASE_URL = "https://finnhub.io/api/v1"
_TIMEOUT = 15.0


def _env(name: str) -> str:
    """Read an env var, falling back to the project .env (unprefixed)."""
    val = os.environ.get(name)
    if val:
        return val
    env_file = _ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                if k.strip() == name:
                    return v.strip()
    return ""


def _token() -> str:
    return _env("FINNHUB_TOKEN")


def is_configured() -> bool:
    """True when a Finnhub token is present."""
    return bool(_token())


def public_token() -> str:
    """Token for the client-side WebSocket (embedded in public HTML), or ""."""
    return _token()


def get_quote(ticker: str, client: httpx.Client | None = None) -> dict | None:
    """Fetch a real-time quote for one ticker via GET /quote.

    Returns a dict with price/change/pct/high/low/open/prev_close, or None if
    unconfigured or the symbol has no data (Finnhub returns price 0 then).
    """
    token = _token()
    if not token:
        return None
    owns = client is None
    client = client or httpx.Client(timeout=_TIMEOUT)
    try:
        r = client.get(f"{_BASE_URL}/quote",
                        params={"symbol": ticker, "token": token})
        r.raise_for_status()
        d = r.json()
    except Exception as e:  # noqa: BLE001 — best-effort, never break the caller
        log.warning("finnhub_quote_failed", ticker=ticker, error=str(e)[:80])
        return None
    finally:
        if owns:
            client.close()

    price = d.get("c")
    if not price:  # 0 / None → no data for this symbol
        return None
    return {
        "ticker": ticker,
        "price": float(price),
        "change": float(d.get("d") or 0),
        "change_percent": float(d.get("dp") or 0),
        "high": float(d.get("h") or 0),
        "low": float(d.get("l") or 0),
        "open": float(d.get("o") or 0),
        "prev_close": float(d.get("pc") or 0),
    }


def get_quotes(tickers: list[str]) -> dict[str, dict]:
    """Fetch quotes for several tickers (one REST call each, ≤60/min free).

    Returns {ticker: quote_dict} for tickers that returned data.
    """
    if not is_configured() or not tickers:
        return {}
    out: dict[str, dict] = {}
    with httpx.Client(timeout=_TIMEOUT) as client:
        for t in tickers:
            q = get_quote(t, client=client)
            if q:
                out[t] = q
    log.info("finnhub_quotes", requested=len(tickers), returned=len(out))
    return out
