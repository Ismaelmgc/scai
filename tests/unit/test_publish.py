"""publish_book: write order (NAV before the view when known) and idempotent re-runs."""
import pandas as pd

from app.books import publish
from app.data import supabase_store
from app.web import dashboard_data

OHLCV = pd.DataFrame({"date": pd.to_datetime(["2026-10-26"]), "ticker": ["AAA"], "close": [10.0]})


def _wire(monkeypatch, calls):
    for name in ["write_state", "upsert_signals", "upsert_nav", "write_dashboard_view",
                 "upsert_live_prices"]:
        monkeypatch.setattr(supabase_store, name,
                            lambda *a, _n=name, **k: calls.append((_n, a)))
    view = {"paper": {"total_value": 1234.5,
                      "positions": [{"ticker": "AAA", "current_price": 10.0}]}}
    monkeypatch.setattr(dashboard_data, "build_view",
                        lambda *a, **k: calls.append(("build_view", a)) or view)


def test_nav_written_before_view_when_provided(monkeypatch, tmp_path):
    calls = []
    _wire(monkeypatch, calls)
    ohlcv = OHLCV
    publish.publish_book("x", {"cash": 1.0}, ohlcv, tmp_path,
                         nav={"2026-10-23": 1200.0, "2026-10-26": 1234.5}, signals=[{"t": 1}])
    order = [c[0] for c in calls]
    assert order.index("upsert_nav") < order.index("build_view")
    assert [c[1][1] for c in calls if c[0] == "upsert_nav"] == ["2026-10-23", "2026-10-26"]
    assert order[0] == "write_state" and "upsert_signals" in order
    assert calls[-1] == ("upsert_live_prices", ({"AAA": 10.0},))


def test_rerun_without_state_write_uses_view_value(monkeypatch, tmp_path):
    calls = []
    _wire(monkeypatch, calls)
    ohlcv = OHLCV
    publish.publish_book("x", {}, ohlcv, tmp_path, write_state=False, nav_date="2026-10-26")
    names = [c[0] for c in calls]
    assert "write_state" not in names and "upsert_signals" not in names
    assert ("upsert_nav", ("x", "2026-10-26", 1234.5)) in calls
