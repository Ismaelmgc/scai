"""Tests for Massive data layer — using mocked HTTP responses.

All tests run without an API key using httpx_mock or monkeypatched responses.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from app.data.massive.client import MassiveClient, RateLimiter
from app.data.massive.schemas import (
    DailyBar,
    DividendRecord,
    QuoteRecord,
    SnapshotTicker,
    SplitRecord,
    TickerDetail,
    TickerRecord,
    TradeRecord,
)


# ── Fixtures ────────────────────────────────────────────────
@pytest.fixture
def client():
    """Create a MassiveClient with a fake API key (no real calls)."""
    return MassiveClient(api_key="test_key_12345", calls_per_minute=600)


@pytest.fixture
def mock_response():
    """Factory for httpx Response mocks."""
    def _make(json_data, status_code=200):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_data
        resp.request = MagicMock()
        return resp
    return _make


# ── Rate Limiter Tests ──────────────────────────────────────
class TestRateLimiter:
    def test_first_call_no_wait(self):
        limiter = RateLimiter(calls_per_minute=60)
        import time
        start = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - start
        # First call should be near-instant
        assert elapsed < 0.1

    def test_interval_calculation(self):
        limiter = RateLimiter(calls_per_minute=5)
        assert limiter._interval == 12.0

    def test_zero_cpm_no_crash(self):
        limiter = RateLimiter(calls_per_minute=0)
        limiter.wait()  # Should not raise


# ── Client Tests ────────────────────────────────────────────
class TestMassiveClient:
    def test_init_with_key(self, client):
        assert client._key == "test_key_12345"
        assert client._base_url == "https://api.polygon.io"

    def test_init_no_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("app.data.massive.client.MassiveClient._load_from_settings", return_value=""):
                with pytest.raises(ValueError, match="API key required"):
                    MassiveClient(api_key="")

    def test_request_count_increments(self, client, mock_response):
        with patch.object(client._client, "get", return_value=mock_response({"results": []})):
            client.get("/v3/reference/tickers", {"limit": 1})
            assert client.request_count == 1

    def test_pagination(self, client, mock_response):
        page1 = mock_response({
            "results": [{"ticker": "AAPL"}],
            "next_url": "https://api.polygon.io/v3/reference/tickers?cursor=abc",
        })
        page2 = mock_response({
            "results": [{"ticker": "MSFT"}],
        })
        with patch.object(client._client, "get", side_effect=[page1, page2]):
            results = client.get_all_pages("/v3/reference/tickers")
            assert len(results) == 2
            assert results[0]["ticker"] == "AAPL"
            assert results[1]["ticker"] == "MSFT"

    def test_context_manager(self):
        with MassiveClient(api_key="test_key", calls_per_minute=600) as c:
            assert c._key == "test_key"


# ── Schema Tests ────────────────────────────────────────────
class TestSchemas:
    def test_ticker_record(self):
        t = TickerRecord(ticker="SOFI", name="SoFi Technologies", active=True)
        assert t.ticker == "SOFI"
        assert t.active is True

    def test_daily_bar(self):
        bar = DailyBar(
            ticker="HOOD",
            trading_date=date(2024, 1, 15),
            open=10.5,
            high=11.0,
            low=10.0,
            close=10.8,
            volume=1_000_000,
            vwap=10.6,
            transactions=5000,
        )
        assert bar.close == 10.8
        assert bar.adjusted is True  # default

    def test_quote_spread(self):
        q = QuoteRecord(
            ticker="TEST",
            bid_price=10.00,
            bid_size=100,
            ask_price=10.10,
            ask_size=200,
        )
        assert q.spread == pytest.approx(0.10)
        assert q.mid_price == pytest.approx(10.05)
        assert q.relative_spread == pytest.approx(0.10 / 10.05)

    def test_split_record(self):
        s = SplitRecord(
            ticker="TSLA",
            execution_date=date(2022, 8, 25),
            split_from=1.0,
            split_to=3.0,
        )
        assert s.split_to / s.split_from == 3.0

    def test_snapshot_ticker(self):
        snap = SnapshotTicker(
            ticker="PLTR",
            day_open=20.0,
            day_high=21.0,
            day_low=19.5,
            day_close=20.5,
            day_volume=50_000_000,
            change=0.5,
            change_percent=2.5,
        )
        assert snap.change_percent == 2.5


# ── Reference API Tests ─────────────────────────────────────
class TestReferenceAPI:
    def test_list_tickers(self, client, mock_response):
        from app.data.massive.reference import ReferenceAPI

        api = ReferenceAPI(client)
        fake_resp = mock_response({
            "results": [
                {"ticker": "SOFI", "name": "SoFi", "active": True, "market": "stocks", "locale": "us"},
                {"ticker": "HOOD", "name": "Robinhood", "active": True, "market": "stocks", "locale": "us"},
            ]
        })
        with patch.object(client._client, "get", return_value=fake_resp):
            tickers = api.list_tickers(limit=10)
            assert len(tickers) == 2
            assert tickers[0].ticker == "SOFI"

    def test_get_ticker_details(self, client, mock_response):
        from app.data.massive.reference import ReferenceAPI

        api = ReferenceAPI(client)
        fake_resp = mock_response({
            "results": {
                "ticker": "SOFI",
                "name": "SoFi Technologies Inc",
                "market_cap": 8_000_000_000,
                "sic_code": "6159",
                "active": True,
                "market": "stocks",
                "locale": "us",
            }
        })
        with patch.object(client._client, "get", return_value=fake_resp):
            detail = api.get_ticker_details("SOFI")
            assert detail is not None
            assert detail.market_cap == 8_000_000_000


# ── Aggregates API Tests ────────────────────────────────────
class TestAggregatesAPI:
    def test_get_custom_bars(self, client, mock_response):
        from app.data.massive.aggregates import AggregatesAPI

        api = AggregatesAPI(client)
        fake_resp = mock_response({
            "results": [
                {"o": 10.0, "h": 11.0, "l": 9.5, "c": 10.5, "v": 500000, "vw": 10.3, "n": 1000, "t": 1705363200000},
                {"o": 10.5, "h": 12.0, "l": 10.0, "c": 11.5, "v": 600000, "vw": 11.0, "n": 1200, "t": 1705449600000},
            ]
        })
        with patch.object(client._client, "get", return_value=fake_resp):
            bars = api.get_custom_bars("SOFI", from_date=date(2024, 1, 1), to_date=date(2024, 1, 31))
            assert len(bars) == 2
            assert bars[0].vwap == 10.3
            assert bars[0].transactions == 1000

    def test_get_grouped_daily(self, client, mock_response):
        from app.data.massive.aggregates import AggregatesAPI

        api = AggregatesAPI(client)
        fake_resp = mock_response({
            "results": [
                {"T": "SOFI", "o": 10.0, "h": 11.0, "l": 9.5, "c": 10.5, "v": 500000, "vw": 10.3, "n": 1000},
                {"T": "HOOD", "o": 20.0, "h": 21.0, "l": 19.0, "c": 20.5, "v": 800000, "vw": 20.1, "n": 2000},
            ]
        })
        with patch.object(client._client, "get", return_value=fake_resp):
            bars = api.get_grouped_daily(date(2024, 1, 15))
            assert len(bars) == 2
            assert bars[0].ticker == "SOFI"
            assert bars[1].ticker == "HOOD"


# ── Corporate Actions Tests ─────────────────────────────────
class TestCorporateActionsAPI:
    def test_get_splits(self, client, mock_response):
        from app.data.massive.corporate_actions import CorporateActionsAPI

        api = CorporateActionsAPI(client)
        fake_resp = mock_response({
            "results": [
                {"ticker": "TSLA", "execution_date": "2022-08-25", "split_from": 1, "split_to": 3},
            ]
        })
        with patch.object(client._client, "get", return_value=fake_resp):
            splits = api.get_splits(ticker="TSLA")
            assert len(splits) == 1
            assert splits[0].split_to == 3

    def test_get_dividends(self, client, mock_response):
        from app.data.massive.corporate_actions import CorporateActionsAPI

        api = CorporateActionsAPI(client)
        fake_resp = mock_response({
            "results": [
                {"ticker": "AAPL", "ex_dividend_date": "2024-02-09", "cash_amount": 0.24, "frequency": 4},
            ]
        })
        with patch.object(client._client, "get", return_value=fake_resp):
            divs = api.get_dividends(ticker="AAPL")
            assert len(divs) == 1
            assert divs[0].cash_amount == 0.24


# ── Validation Tests ────────────────────────────────────────
class TestValidation:
    def test_validate_no_data(self):
        """Validation should not crash when no data exists."""
        from app.data.massive.jobs import validate_data
        with patch("app.data.massive.jobs._get_store") as mock_store:
            store = MagicMock()
            store.read.side_effect = Exception("not found")
            mock_store.return_value = store
            issues = validate_data()
            assert "daily_bars" in issues
