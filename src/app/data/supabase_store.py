"""Supabase persistence for paper-trading state (single source of truth).

Replaces the git-committed JSON/parquet state files. The pipeline writes with
the service_role key (bypasses RLS); the public dashboard reads with the anon
key. Credentials come from the environment (GitHub Actions secrets) or, locally,
from the project ``.env`` (the app's pydantic Settings use a ``SCAI_`` prefix, so
the unprefixed SUPABASE_* vars are read directly here).

If credentials are absent the module degrades to no-ops (with a warning) so the
pipeline still runs locally without Supabase.
"""
from __future__ import annotations

import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np

from app.utils import get_logger

log = get_logger(__name__)

_ROOT = Path(__file__).resolve().parents[3]
_TABLES = ("portfolio_state", "trades", "signals", "nav_history")
_TIMEOUT = 30.0


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


def _base_url() -> str:
    return _env("SUPABASE_URL").rstrip("/")


def _service_key() -> str:
    return _env("SUPABASE_SERVICE_KEY")


def _anon_key() -> str:
    return _env("SUPABASE_ANON_KEY")


def _read_key() -> str:
    """Key for reads: the service key when available (CI pipeline / local with
    .env), else the anon key (Pages render step, which only carries the anon
    key). RLS grants anon SELECT on every table, so reads work with either;
    writes still require the service key."""
    return _service_key() or _anon_key()


def is_configured() -> bool:
    """True when URL + service key are present (writes possible)."""
    return bool(_base_url() and _service_key())


def _read_configured() -> bool:
    """True when reads are possible (URL + any key — service or anon)."""
    return bool(_base_url() and _read_key())


def public_config() -> tuple[str, str]:
    """URL + anon (publishable) key for the client-side dashboard.

    The anon key is safe to embed in the public Pages HTML: RLS grants it
    read-only access. Returns ``("", "")`` when either is unset, so callers
    can skip emitting the live-refresh script.
    """
    return _base_url(), _anon_key()


def _headers(extra: dict[str, str] | None = None,
             key: str | None = None) -> dict[str, str]:
    key = key or _service_key()
    h = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if extra:
        h.update(extra)
    return h


def _json_safe(obj):
    """Recursively make a payload JSON-serialisable by httpx, whose encoder is
    strict (``allow_nan=False``): non-finite floats (NaN / ±Inf) become None and
    numpy scalars become native Python types. A single NaN anywhere in the state
    / view / signals payload (e.g. a win-rate with no closed trades, or a missing
    live price) otherwise raises "Out of range float values are not JSON
    compliant" and fails the whole daily job."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()  # numpy scalar -> python float/int/bool
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


# Cloudflare/Supabase throw transient 5xx (a 525 SSL-handshake blip once killed a
# whole daily run on read_state). Retry those + network errors with backoff so a
# momentary infra hiccup no longer fails the pipeline.
_RETRY_STATUS = {502, 503, 504, 520, 521, 522, 523, 524, 525, 527, 530}
_MAX_TRIES = 4


def _request(method: str, url: str, **kwargs) -> httpx.Response:
    """httpx request with retry/backoff on transient Cloudflare/Supabase errors."""
    fn = getattr(httpx, method.lower())  # httpx.get / httpx.post
    for attempt in range(_MAX_TRIES):
        last = attempt == _MAX_TRIES - 1
        try:
            r = fn(url, timeout=_TIMEOUT, **kwargs)
        except httpx.TransportError:
            if last:
                raise
            time.sleep(2 ** attempt)
            continue
        if r.status_code in _RETRY_STATUS and not last:
            log.warning("supabase_transient_retry", status=r.status_code,
                        attempt=attempt + 1, url=url.rsplit("/", 1)[-1])
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r
    raise RuntimeError("unreachable")  # pragma: no cover


def _post(table: str, rows: list[dict], on_conflict: str | None = None,
          resolution: str = "merge-duplicates") -> None:
    """Insert/upsert rows into a table via PostgREST."""
    if not rows:
        return
    url = f"{_base_url()}/rest/v1/{table}"
    params = {"on_conflict": on_conflict} if on_conflict else {}
    headers = _headers({"Prefer": f"resolution={resolution}"})
    _request("POST", url, json=_json_safe(rows), params=params, headers=headers)


# ── Public API ───────────────────────────────────────────────

def write_state(strategy: str, state: dict) -> None:
    """Upsert the full portfolio state (one row per strategy)."""
    if not is_configured():
        log.warning("supabase_not_configured", op="write_state", strategy=strategy)
        return
    now = datetime.now(UTC).isoformat()
    _post("portfolio_state",
          [{"strategy": strategy, "state": state, "updated_at": now}],
          on_conflict="strategy")


def read_state(strategy: str) -> dict | None:
    """Fetch the portfolio state row for a strategy, or None if absent.

    Reads with the anon key when no service key is present (Pages render),
    relying on the RLS read policy.
    """
    if not _read_configured():
        return None
    url = f"{_base_url()}/rest/v1/portfolio_state"
    params = {"strategy": f"eq.{strategy}", "select": "state", "limit": "1"}
    r = _request("GET", url, params=params, headers=_headers(key=_read_key()))
    data = r.json()
    return data[0]["state"] if data else None


def append_trades(strategy: str, trades: list[dict]) -> None:
    """Append closed trades (idempotent: dups on strategy/ticker/entry/exit ignored)."""
    if not is_configured() or not trades:
        return
    cols = ("ticker", "entry_date", "exit_date", "entry_price", "exit_price",
            "shares", "pnl_pct", "pnl_usd", "exit_reason", "days_held")
    rows = [{"strategy": strategy, **{c: t.get(c) for c in cols}} for t in trades]
    _post("trades", rows, on_conflict="strategy,ticker,entry_date,exit_date",
          resolution="ignore-duplicates")


def upsert_signals(strategy: str, signals: list[dict]) -> None:
    """Upsert daily signals (one row per strategy/date/ticker)."""
    if not is_configured() or not signals:
        return
    cols = ("signal_date", "ticker", "score", "recommendation",
            "was_traded", "skip_reason", "actual_ret_20d")
    rows = [{"strategy": strategy, **{c: s.get(c) for c in cols}} for s in signals]
    _post("signals", rows, on_conflict="strategy,signal_date,ticker")


def update_signal_outcomes(strategy: str, outcomes: list[dict]) -> None:
    """Patch ``actual_ret_20d`` on existing signal rows once the 20d horizon has
    elapsed. Only those three keys are sent, so the merge-upsert leaves score /
    recommendation / skip_reason untouched (PostgREST updates only payload cols).

    outcomes: list of {signal_date, ticker, actual_ret_20d}.
    """
    if not is_configured() or not outcomes:
        return
    rows = [{"strategy": strategy, "signal_date": o["signal_date"],
             "ticker": o["ticker"], "actual_ret_20d": o["actual_ret_20d"]}
            for o in outcomes]
    _post("signals", rows, on_conflict="strategy,signal_date,ticker")


def upsert_nav(strategy: str, date: str, portfolio_value: float) -> None:
    """Upsert one daily NAV point for the equity chart."""
    if not is_configured():
        return
    _post("nav_history",
          [{"strategy": strategy, "date": date, "portfolio_value": portfolio_value}],
          on_conflict="strategy,date")


def write_dashboard_view(strategy: str, view: dict) -> None:
    """Upsert the render-ready dashboard view (one row per strategy).

    The logged-in client reads this single row to paint the dashboard, so raw
    trade tables never reach the browser and no client-side computation is needed.
    """
    if not is_configured():
        log.warning("supabase_not_configured", op="write_dashboard_view", strategy=strategy)
        return
    now = datetime.now(UTC).isoformat()
    _post("dashboard_view",
          [{"strategy": strategy, "view": view, "updated_at": now}],
          on_conflict="strategy")


def _get(table: str, params: dict) -> list[dict]:
    r = _request("GET", f"{_base_url()}/rest/v1/{table}", params=params,
                 headers=_headers(key=_read_key()))
    return r.json()


def read_nav(strategy: str) -> list[dict]:
    """Daily NAV points (date, portfolio_value) ascending, for the equity chart."""
    if not _read_configured():
        return []
    return _get("nav_history", {
        "strategy": f"eq.{strategy}",
        "select": "date,portfolio_value",
        "order": "date.asc",
    })


def read_signals(strategy: str, limit: int = 50) -> list[dict]:
    """Most recent signals for a strategy (newest first)."""
    if not _read_configured():
        return []
    return _get("signals", {
        "strategy": f"eq.{strategy}",
        "select": "signal_date,ticker,score,recommendation,was_traded,skip_reason,actual_ret_20d",
        "order": "signal_date.desc,score.desc",
        "limit": str(limit),
    })


def read_signals_since(strategy: str, since_date: str, page: int = 1000) -> list[dict]:
    """All signals for a strategy on/after ``since_date`` (full cross-section).

    Paginated so it returns the whole window even past PostgREST's 1000-row cap —
    the daily cross-section is ~320 names, so a few weeks already exceeds one page.
    Used by the live-IC monitor, which needs every scored name per date.
    """
    if not _read_configured():
        return []
    out: list[dict] = []
    offset = 0
    while True:
        batch = _get("signals", {
            "strategy": f"eq.{strategy}",
            "signal_date": f"gte.{since_date}",
            "select": "signal_date,ticker,score,was_traded,actual_ret_20d",
            "order": "signal_date.asc,ticker.asc",
            "limit": str(page),
            "offset": str(offset),
        })
        out.extend(batch)
        if len(batch) < page:
            return out
        offset += page
