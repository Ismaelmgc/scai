"""Tests for the Finnhub live-quote connector (parsing + no-op fallback)."""
from app.data.free_sources import finnhub as f


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_not_configured_is_noop(monkeypatch):
    monkeypatch.setattr(f, "_token", lambda: "")

    def _boom(*a, **k):  # must never hit the network when unconfigured
        raise AssertionError("httpx called while not configured")

    monkeypatch.setattr(f.httpx, "Client", _boom)
    assert f.is_configured() is False
    assert f.public_token() == ""
    assert f.get_quote("AAPL") is None
    assert f.get_quotes(["AAPL", "MSFT"]) == {}


def test_get_quote_parses_fields(monkeypatch):
    monkeypatch.setattr(f, "_token", lambda: "tok")

    class _Client:
        def __init__(self, *a, **k):
            pass

        def get(self, url, params):
            assert params["symbol"] == "AAPL"
            assert params["token"] == "tok"
            return _FakeResp({"c": 12.5, "d": 0.5, "dp": 4.0,
                              "h": 13.0, "l": 11.8, "o": 12.0, "pc": 12.0})

        def close(self):
            pass

    monkeypatch.setattr(f.httpx, "Client", _Client)
    q = f.get_quote("AAPL")
    assert q["ticker"] == "AAPL"
    assert q["price"] == 12.5
    assert q["change_percent"] == 4.0
    assert q["prev_close"] == 12.0


def test_get_quote_zero_price_is_none(monkeypatch):
    """Finnhub returns c=0 for symbols with no data → treat as missing."""
    monkeypatch.setattr(f, "_token", lambda: "tok")

    class _Client:
        def __init__(self, *a, **k):
            pass

        def get(self, url, params):
            return _FakeResp({"c": 0, "d": None, "dp": None,
                              "h": 0, "l": 0, "o": 0, "pc": 0})

        def close(self):
            pass

    monkeypatch.setattr(f.httpx, "Client", _Client)
    assert f.get_quote("ZZZZ") is None


def test_get_quotes_maps_by_ticker(monkeypatch):
    monkeypatch.setattr(f, "_token", lambda: "tok")
    prices = {"AAA": 1.0, "BBB": 2.0}

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def get(self, url, params):
            return _FakeResp({"c": prices[params["symbol"]]})

        def close(self):
            pass

    monkeypatch.setattr(f.httpx, "Client", _Client)
    out = f.get_quotes(["AAA", "BBB"])
    assert set(out) == {"AAA", "BBB"}
    assert out["AAA"]["price"] == 1.0
    assert out["BBB"]["price"] == 2.0
