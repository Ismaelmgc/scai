"""Trades endpoint: tick-level trade data."""

from __future__ import annotations

from typing import Any

from app.data.massive.client import MassiveClient
from app.data.massive.schemas import TradeRecord
from app.utils import get_logger

log = get_logger(__name__)


class TradesAPI:
    """Wrapper for Massive trades endpoint (/v3/trades/{stockTicker})."""

    def __init__(self, client: MassiveClient) -> None:
        self._c = client

    def get_trades(
        self,
        ticker: str,
        *,
        timestamp: str | None = None,
        timestamp_gte: str | None = None,
        timestamp_lte: str | None = None,
        sort: str = "timestamp",
        order: str = "asc",
        limit: int = 50000,
        max_pages: int = 10,
    ) -> list[TradeRecord]:
        """Fetch trades for a ticker.

        Parameters
        ----------
        ticker : str
        timestamp : str - exact nanosecond timestamp filter
        timestamp_gte, timestamp_lte : str - range filters (ISO or nanosecond)
        sort : str (timestamp)
        order : str (asc/desc)
        limit : int (up to 50000 per page)
        max_pages : int - limit pagination depth (default 10 = 500k trades max)

        Use Cases:
        - trade_count for liquidity features
        - dollar_volume calculation
        - off-exchange/TRF participation ratio
        - intraday volatility
        - anomalous print detection

        WARNING: Do NOT download full tick history via REST for entire universe.
        Use Flat Files for bulk historical tick data.
        """
        params: dict[str, Any] = {
            "sort": sort,
            "order": order,
            "limit": limit,
        }
        if timestamp:
            params["timestamp"] = timestamp
        if timestamp_gte:
            params["timestamp.gte"] = timestamp_gte
        if timestamp_lte:
            params["timestamp.lte"] = timestamp_lte

        path = f"/v3/trades/{ticker}"
        all_results: list[TradeRecord] = []
        data = self._c.get(path, params)

        for r in data.get("results", []):
            all_results.append(TradeRecord(ticker=ticker, **r))

        pages = 1
        while pages < max_pages and data.get("next_url"):
            data = self._c.get(data["next_url"], _is_next_url=True)
            for r in data.get("results", []):
                all_results.append(TradeRecord(ticker=ticker, **r))
            pages += 1

        log.info("get_trades", ticker=ticker, count=len(all_results), pages=pages)
        return all_results

    def get_trade_count(
        self,
        ticker: str,
        *,
        timestamp_gte: str | None = None,
        timestamp_lte: str | None = None,
    ) -> int:
        """Get approximate trade count for a period (single page, no deep pagination)."""
        params: dict[str, Any] = {"limit": 1}
        if timestamp_gte:
            params["timestamp.gte"] = timestamp_gte
        if timestamp_lte:
            params["timestamp.lte"] = timestamp_lte

        data = self._c.get(f"/v3/trades/{ticker}", params)
        # The 'results_count' field gives total matched
        return data.get("results_count", len(data.get("results", [])))
