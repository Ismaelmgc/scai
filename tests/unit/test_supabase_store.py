"""Tests for the Supabase paper-trading store (request shapes + no-op fallback)."""
import pytest

from app.data import supabase_store as s


class _FakeResp:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return [{"state": {"cash": 1000.0}}]


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(s, "_base_url", lambda: "https://x.supabase.co")
    monkeypatch.setattr(s, "_service_key", lambda: "svc-key")


def test_not_configured_is_noop(monkeypatch):
    monkeypatch.setattr(s, "_base_url", lambda: "")
    monkeypatch.setattr(s, "_service_key", lambda: "")

    def _boom(*a, **k):  # must never be called when unconfigured
        raise AssertionError("httpx called while not configured")

    monkeypatch.setattr(s.httpx, "post", _boom)
    assert s.is_configured() is False
    s.write_state("baseline", {"cash": 1000.0})  # no exception, no call
    s.upsert_nav("baseline", "2026-06-12", 1000.0)


def test_write_state_upserts_on_strategy(configured, monkeypatch):
    captured = {}

    def _post(url, json, params, headers, timeout):
        captured.update(url=url, json=json, params=params, headers=headers)
        return _FakeResp()

    monkeypatch.setattr(s.httpx, "post", _post)
    s.write_state("baseline", {"cash": 1000.0})

    assert captured["url"].endswith("/rest/v1/portfolio_state")
    assert captured["params"] == {"on_conflict": "strategy"}
    assert captured["json"][0]["strategy"] == "baseline"
    assert captured["json"][0]["state"] == {"cash": 1000.0}
    assert "merge-duplicates" in captured["headers"]["Prefer"]


def test_append_trades_dedup(configured, monkeypatch):
    captured = {}

    def _post(url, json, params, headers, timeout):
        captured.update(url=url, json=json, params=params, headers=headers)
        return _FakeResp()

    monkeypatch.setattr(s.httpx, "post", _post)
    s.append_trades("adaptive", [{
        "ticker": "AAA", "entry_date": "2026-06-11", "exit_date": "2026-06-12",
        "entry_price": 2.0, "exit_price": 2.2, "shares": 50, "pnl_pct": 0.1,
        "pnl_usd": 10.0, "exit_reason": "profit_target", "days_held": 1,
        "side": "long",  # extra field must be dropped
    }])

    assert captured["url"].endswith("/rest/v1/trades")
    assert captured["params"] == {"on_conflict": "strategy,ticker,entry_date,exit_date"}
    assert "ignore-duplicates" in captured["headers"]["Prefer"]
    row = captured["json"][0]
    assert row["strategy"] == "adaptive"
    assert "side" not in row  # only mapped columns are sent


def test_read_state_returns_payload(configured, monkeypatch):
    monkeypatch.setattr(s.httpx, "get", lambda *a, **k: _FakeResp())
    assert s.read_state("baseline") == {"cash": 1000.0}


def test_write_dashboard_view_upserts_on_strategy(configured, monkeypatch):
    captured = {}

    def _post(url, json, params, headers, timeout):
        captured.update(url=url, json=json, params=params)
        return _FakeResp()

    monkeypatch.setattr(s.httpx, "post", _post)
    s.write_dashboard_view("baseline", {"paper": {"cash": 1000.0}, "signals": []})

    assert captured["url"].endswith("/rest/v1/dashboard_view")
    assert captured["params"] == {"on_conflict": "strategy"}
    assert captured["json"][0]["strategy"] == "baseline"
    assert captured["json"][0]["view"]["paper"] == {"cash": 1000.0}


def test_write_dashboard_view_noop_when_unconfigured(monkeypatch):
    monkeypatch.setattr(s, "_base_url", lambda: "")
    monkeypatch.setattr(s, "_service_key", lambda: "")
    monkeypatch.setattr(s.httpx, "post",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("posted")))
    s.write_dashboard_view("baseline", {"x": 1})  # no exception, no call
