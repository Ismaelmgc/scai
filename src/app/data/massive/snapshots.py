"""Snapshots endpoints: full market, single ticker, gainers/losers."""

from __future__ import annotations

from typing import Any

from app.data.massive.client import MassiveClient
from app.data.massive.schemas import SnapshotTicker
from app.utils import get_logger

log = get_logger(__name__)


def _parse_snapshot(raw: dict[str, Any]) -> SnapshotTicker:
    """Parse a single snapshot ticker from raw API response."""
    day = raw.get("day", {})
    prev = raw.get("prevDay", {})
    minute = raw.get("min", {})
    last_trade = raw.get("lastTrade", {})
    last_quote = raw.get("lastQuote", {})

    return SnapshotTicker(
        ticker=raw.get("ticker", ""),
        updated=raw.get("updated"),
        # Day
        day_open=day.get("o"),
        day_high=day.get("h"),
        day_low=day.get("l"),
        day_close=day.get("c"),
        day_volume=day.get("v"),
        day_vwap=day.get("vw"),
        # Previous day
        prev_open=prev.get("o"),
        prev_high=prev.get("h"),
        prev_low=prev.get("l"),
        prev_close=prev.get("c"),
        prev_volume=prev.get("v"),
        prev_vwap=prev.get("vw"),
        # Minute
        min_open=minute.get("o"),
        min_high=minute.get("h"),
        min_low=minute.get("l"),
        min_close=minute.get("c"),
        min_volume=minute.get("v"),
        min_vwap=minute.get("vw"),
        # Last trade
        last_trade_price=last_trade.get("p"),
        last_trade_size=last_trade.get("s"),
        last_trade_timestamp=last_trade.get("t"),
        # Last quote — p=bid, P=ask, s=bid_size, S=ask_size
        last_quote_bid=last_quote.get("p"),
        last_quote_ask=last_quote.get("P"),
        last_quote_bid_size=last_quote.get("s"),
        last_quote_ask_size=last_quote.get("S"),
        last_quote_timestamp=last_quote.get("t"),
        # Derived
        change=raw.get("todaysChange"),
        change_percent=raw.get("todaysChangePerc"),
        otc=raw.get("otc", False),
    )


class SnapshotsAPI:
    """Wrapper for Massive snapshot endpoints."""

    def __init__(self, client: MassiveClient) -> None:
        self._c = client

    def get_full_market_snapshot(
        self,
        *,
        tickers: list[str] | None = None,
        include_otc: bool = False,
    ) -> list[SnapshotTicker]:
        """Fetch full market snapshot from /v2/snapshot/locale/us/markets/stocks/tickers.

        If tickers is provided, filters to those specific tickers.
        One API call returns all tickers — useful for daily screening.
        """
        params: dict[str, Any] = {"include_otc": str(include_otc).lower()}
        if tickers:
            params["tickers"] = ",".join(tickers)

        data = self._c.get("/v2/snapshot/locale/us/markets/stocks/tickers", params)
        results = data.get("tickers", [])

        snapshots = [_parse_snapshot(r) for r in results]
        log.info("full_market_snapshot", count=len(snapshots))
        return snapshots

    def get_single_ticker_snapshot(self, ticker: str) -> SnapshotTicker | None:
        """Fetch single ticker snapshot from /v2/snapshot/locale/us/markets/stocks/tickers/{ticker}."""
        data = self._c.get(
            f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
        )
        result = data.get("ticker")
        if not result:
            # Try alternate response shape
            results = data.get("tickers", [])
            if results:
                result = results[0]
        if not result:
            return None
        return _parse_snapshot(result)

    def get_gainers(self, *, include_otc: bool = False) -> list[SnapshotTicker]:
        """Fetch top market gainers from /v2/snapshot/locale/us/markets/stocks/gainers.

        NOTE: Incorporates intraday movement — DO NOT use for historical training.
        """
        params: dict[str, Any] = {"include_otc": str(include_otc).lower()}
        data = self._c.get("/v2/snapshot/locale/us/markets/stocks/gainers", params)
        results = data.get("tickers", [])
        return [_parse_snapshot(r) for r in results]

    def get_losers(self, *, include_otc: bool = False) -> list[SnapshotTicker]:
        """Fetch top market losers from /v2/snapshot/locale/us/markets/stocks/losers.

        NOTE: Incorporates intraday movement — DO NOT use for historical training.
        """
        params: dict[str, Any] = {"include_otc": str(include_otc).lower()}
        data = self._c.get("/v2/snapshot/locale/us/markets/stocks/losers", params)
        results = data.get("tickers", [])
        return [_parse_snapshot(r) for r in results]
