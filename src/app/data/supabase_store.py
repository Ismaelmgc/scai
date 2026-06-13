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

import os
from datetime import UTC, datetime
from pathlib import Path

import httpx

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


def _post(table: str, rows: list[dict], on_conflict: str | None = None,
          resolution: str = "merge-duplicates") -> None:
    """Insert/upsert rows into a table via PostgREST."""
    if not rows:
        return
    url = f"{_base_url()}/rest/v1/{table}"
    params = {"on_conflict": on_conflict} if on_conflict else {}
    headers = _headers({"Prefer": f"resolution={resolution}"})
    r = httpx.post(url, json=rows, params=params, headers=headers, timeout=_TIMEOUT)
    r.raise_for_status()


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
    r = httpx.get(url, params=params, headers=_headers(key=_read_key()), timeout=_TIMEOUT)
    r.raise_for_status()
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


def upsert_nav(strategy: str, date: str, portfolio_value: float) -> None:
    """Upsert one daily NAV point for the equity chart."""
    if not is_configured():
        return
    _post("nav_history",
          [{"strategy": strategy, "date": date, "portfolio_value": portfolio_value}],
          on_conflict="strategy,date")


def _get(table: str, params: dict) -> list[dict]:
    r = httpx.get(f"{_base_url()}/rest/v1/{table}", params=params,
                  headers=_headers(key=_read_key()), timeout=_TIMEOUT)
    r.raise_for_status()
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
