"""Reference data endpoints: tickers, ticker details, types, events, IPOs, exchanges, conditions."""

from __future__ import annotations

from datetime import date
from typing import Any

from app.data.massive.client import MassiveClient
from app.data.massive.schemas import (
    Condition,
    Exchange,
    IPORecord,
    TickerDetail,
    TickerEvent,
    TickerRecord,
    TickerType,
)
from app.utils import get_logger

log = get_logger(__name__)


class ReferenceAPI:
    """Wrapper for Massive reference endpoints."""

    def __init__(self, client: MassiveClient) -> None:
        self._c = client

    # ── Tickers ─────────────────────────────────────────────
    def list_tickers(
        self,
        *,
        market: str = "stocks",
        locale: str = "us",
        active: bool | None = None,
        ticker: str | None = None,
        ticker_type: str | None = None,
        search: str | None = None,
        date_: date | None = None,
        sort: str = "ticker",
        order: str = "asc",
        limit: int = 1000,
    ) -> list[TickerRecord]:
        """Fetch paginated list of tickers from /v3/reference/tickers.

        Supports point-in-time via 'date_' param for survivorship-bias-free universes.
        """
        params: dict[str, Any] = {
            "market": market,
            "locale": locale,
            "sort": sort,
            "order": order,
            "limit": limit,
        }
        if active is not None:
            params["active"] = str(active).lower()
        if ticker:
            params["ticker"] = ticker
        if ticker_type:
            params["type"] = ticker_type
        if search:
            params["search"] = search
        if date_:
            params["date"] = date_.isoformat()

        raw = self._c.get_all_pages("/v3/reference/tickers", params)
        log.info("list_tickers", count=len(raw))
        return [TickerRecord(**r) for r in raw]

    def get_ticker_details(
        self,
        ticker: str,
        *,
        date_: date | None = None,
    ) -> TickerDetail | None:
        """Fetch ticker details from /v3/reference/tickers/{ticker}.

        Point-in-time via date_ param. Returns None if ticker not found.
        """
        params: dict[str, Any] = {}
        if date_:
            params["date"] = date_.isoformat()

        try:
            data = self._c.get(f"/v3/reference/tickers/{ticker}", params)
        except Exception as e:
            log.warning("ticker_details_failed", ticker=ticker, error=str(e))
            return None
        result = data.get("results")
        if not result:
            return None
        detail = TickerDetail(**result)
        if date_:
            detail.as_of_date = date_
        return detail

    def get_ticker_types(self) -> list[TickerType]:
        """Fetch all ticker types from /v3/reference/tickers/types."""
        data = self._c.get("/v3/reference/tickers/types")
        results = data.get("results", [])
        return [TickerType(**r) for r in results]

    def get_ticker_events(self, ticker_id: str) -> list[TickerEvent]:
        """Fetch ticker events from /vX/reference/tickers/{id}/events.

        EXPERIMENTAL: endpoint may change.
        """
        try:
            data = self._c.get(f"/vX/reference/tickers/{ticker_id}/events")
            events = data.get("results", {}).get("events", [])
            return [TickerEvent(ticker=ticker_id, **e) for e in events]
        except Exception as e:
            log.warning("ticker_events_failed", ticker=ticker_id, error=str(e))
            return []

    def get_ipos(
        self,
        *,
        ticker: str | None = None,
        listing_date_gte: date | None = None,
        listing_date_lte: date | None = None,
        order: str = "desc",
        limit: int = 1000,
    ) -> list[IPORecord]:
        """Fetch IPOs from /vX/reference/ipos.

        EXPERIMENTAL: endpoint may change.
        """
        params: dict[str, Any] = {"order": order, "limit": limit}
        if ticker:
            params["ticker"] = ticker
        if listing_date_gte:
            params["listing_date.gte"] = listing_date_gte.isoformat()
        if listing_date_lte:
            params["listing_date.lte"] = listing_date_lte.isoformat()

        try:
            raw = self._c.get_all_pages("/vX/reference/ipos", params)
            return [IPORecord(**r) for r in raw]
        except Exception as e:
            log.warning("ipos_failed", error=str(e))
            return []

    # ── Market Operations ───────────────────────────────────
    def get_exchanges(self) -> list[Exchange]:
        """Fetch exchanges from /v3/reference/exchanges."""
        data = self._c.get("/v3/reference/exchanges")
        results = data.get("results", [])
        return [Exchange(**r) for r in results]

    def get_conditions(
        self,
        *,
        asset_class: str = "stocks",
        data_type: str | None = None,
    ) -> list[Condition]:
        """Fetch condition codes from /v3/reference/conditions."""
        params: dict[str, Any] = {"asset_class": asset_class}
        if data_type:
            params["data_type"] = data_type
        data = self._c.get("/v3/reference/conditions", params)
        results = data.get("results", [])
        return [Condition(**r) for r in results]

    def get_market_status(self) -> dict[str, Any]:
        """GET /v1/marketstatus/now."""
        return self._c.get("/v1/marketstatus/now")

    def get_market_holidays(self) -> list[dict[str, Any]]:
        """GET /v1/marketstatus/upcoming."""
        data = self._c.get("/v1/marketstatus/upcoming")
        # This endpoint returns a list directly
        if isinstance(data, list):
            return data
        return data.get("results", [])
