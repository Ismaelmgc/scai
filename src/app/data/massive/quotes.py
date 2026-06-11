"""Quotes endpoint: tick-level quote (NBBO) data."""

from __future__ import annotations

from typing import Any

from app.data.massive.client import MassiveClient
from app.data.massive.schemas import QuoteRecord
from app.utils import get_logger

log = get_logger(__name__)


class QuotesAPI:
    """Wrapper for Massive quotes endpoint (/v3/quotes/{stockTicker})."""

    def __init__(self, client: MassiveClient) -> None:
        self._c = client

    def get_quotes(
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
    ) -> list[QuoteRecord]:
        """Fetch NBBO quotes for a ticker.

        Parameters
        ----------
        ticker : str
        timestamp : str - exact nanosecond timestamp filter
        timestamp_gte, timestamp_lte : str - range filters
        sort : str (timestamp)
        order : str (asc/desc)
        limit : int (up to 50000 per page)
        max_pages : int - limit pagination (default 10 = 500k quotes max)

        Derived Features:
        - spread = ask - bid
        - relative_spread = spread / mid
        - quoted_depth = bid_size + ask_size
        - imbalance = (bid_size - ask_size) / (bid_size + ask_size)

        WARNING: Do NOT download full quote history via REST for entire universe.
        Use Flat Files for bulk historical quote data.
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

        path = f"/v3/quotes/{ticker}"
        all_results: list[QuoteRecord] = []
        data = self._c.get(path, params)

        for r in data.get("results", []):
            q = self._parse_quote(ticker, r)
            if q:
                all_results.append(q)

        pages = 1
        while pages < max_pages and data.get("next_url"):
            data = self._c.get(data["next_url"], _is_next_url=True)
            for r in data.get("results", []):
                q = self._parse_quote(ticker, r)
                if q:
                    all_results.append(q)
            pages += 1

        log.info("get_quotes", ticker=ticker, count=len(all_results), pages=pages)
        return all_results

    @staticmethod
    def _parse_quote(ticker: str, r: dict[str, Any]) -> QuoteRecord | None:
        """Parse and validate a single quote record.

        Filters out invalid/crossed quotes (ask < bid or prices <= 0).
        """
        bid = r.get("bid_price", 0)
        ask = r.get("ask_price", 0)

        # Skip invalid quotes
        if bid <= 0 or ask <= 0:
            return None
        if ask < bid:
            return None

        return QuoteRecord(
            ticker=ticker,
            bid_price=bid,
            bid_size=r.get("bid_size", 0),
            bid_exchange=r.get("bid_exchange_id"),
            ask_price=ask,
            ask_size=r.get("ask_size", 0),
            ask_exchange=r.get("ask_exchange_id"),
            conditions=r.get("conditions"),
            indicators=r.get("indicators"),
            participant_timestamp=r.get("participant_timestamp"),
            sip_timestamp=r.get("sip_timestamp"),
            trf_timestamp=r.get("trf_timestamp"),
            sequence_number=r.get("sequence_number"),
            tape=r.get("tape"),
        )

    def compute_spread_stats(
        self,
        ticker: str,
        *,
        timestamp_gte: str | None = None,
        timestamp_lte: str | None = None,
        max_pages: int = 5,
    ) -> dict[str, float]:
        """Compute bid-ask spread statistics for a ticker.

        Returns dict with: mean_spread, mean_relative_spread, mean_depth, mean_imbalance.
        """
        quotes = self.get_quotes(
            ticker,
            timestamp_gte=timestamp_gte,
            timestamp_lte=timestamp_lte,
            max_pages=max_pages,
        )
        if not quotes:
            return {
                "mean_spread": 0.0,
                "mean_relative_spread": 0.0,
                "mean_depth": 0.0,
                "mean_imbalance": 0.0,
                "quote_count": 0,
            }

        spreads = [q.spread for q in quotes]
        rel_spreads = [q.relative_spread for q in quotes]
        depths = [q.bid_size + q.ask_size for q in quotes]
        imbalances = [
            (q.bid_size - q.ask_size) / max(q.bid_size + q.ask_size, 1)
            for q in quotes
        ]

        n = len(quotes)
        return {
            "mean_spread": sum(spreads) / n,
            "mean_relative_spread": sum(rel_spreads) / n,
            "mean_depth": sum(depths) / n,
            "mean_imbalance": sum(imbalances) / n,
            "quote_count": n,
        }
