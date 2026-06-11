"""Corporate actions endpoints: splits, dividends, ticker events."""

from __future__ import annotations

from datetime import date
from typing import Any

from app.data.massive.client import MassiveClient
from app.data.massive.schemas import DividendRecord, SplitRecord
from app.utils import get_logger

log = get_logger(__name__)


class CorporateActionsAPI:
    """Wrapper for Massive corporate actions endpoints."""

    def __init__(self, client: MassiveClient) -> None:
        self._c = client

    def get_splits(
        self,
        *,
        ticker: str | None = None,
        execution_date: date | None = None,
        execution_date_gte: date | None = None,
        execution_date_lte: date | None = None,
        order: str = "desc",
        limit: int = 1000,
    ) -> list[SplitRecord]:
        """Fetch stock splits from /stocks/v1/splits.

        Parameters
        ----------
        ticker : filter by ticker
        execution_date : exact date filter
        execution_date_gte, execution_date_lte : range filters
        order : asc/desc
        limit : results per page

        Notes
        -----
        - Use adjusted=True in aggregate bars if you want pre-adjusted data
        - For raw Flat Files, apply splits manually using split_from/split_to
        """
        params: dict[str, Any] = {"order": order, "limit": limit}
        if ticker:
            params["ticker"] = ticker
        if execution_date:
            params["execution_date"] = execution_date.isoformat()
        if execution_date_gte:
            params["execution_date.gte"] = execution_date_gte.isoformat()
        if execution_date_lte:
            params["execution_date.lte"] = execution_date_lte.isoformat()

        raw = self._c.get_all_pages("/stocks/v1/splits", params)

        splits = []
        for r in raw:
            splits.append(SplitRecord(
                ticker=r.get("ticker", ""),
                execution_date=r.get("execution_date", "1900-01-01"),
                split_from=r.get("split_from", 1),
                split_to=r.get("split_to", 1),
                adjustment_type=r.get("adjustment_type"),
                historical_adjustment_factor=r.get("historical_adjustment_factor"),
            ))

        log.info("get_splits", count=len(splits))
        return splits

    def get_dividends(
        self,
        *,
        ticker: str | None = None,
        ex_dividend_date: date | None = None,
        ex_dividend_date_gte: date | None = None,
        ex_dividend_date_lte: date | None = None,
        frequency: int | None = None,
        distribution_type: str | None = None,
        order: str = "desc",
        limit: int = 1000,
    ) -> list[DividendRecord]:
        """Fetch dividends from /stocks/v1/dividends.

        Parameters
        ----------
        ticker : filter by ticker
        ex_dividend_date : exact date filter
        ex_dividend_date_gte, ex_dividend_date_lte : range filters
        frequency : 1=annual, 2=bi-annual, 4=quarterly, 12=monthly, 0=one-time
        distribution_type : recurring, special, long_term, short_term
        order : asc/desc
        limit : results per page
        """
        params: dict[str, Any] = {"order": order, "limit": limit}
        if ticker:
            params["ticker"] = ticker
        if ex_dividend_date:
            params["ex_dividend_date"] = ex_dividend_date.isoformat()
        if ex_dividend_date_gte:
            params["ex_dividend_date.gte"] = ex_dividend_date_gte.isoformat()
        if ex_dividend_date_lte:
            params["ex_dividend_date.lte"] = ex_dividend_date_lte.isoformat()
        if frequency is not None:
            params["frequency"] = frequency
        if distribution_type:
            params["distribution_type"] = distribution_type

        raw = self._c.get_all_pages("/stocks/v1/dividends", params)

        dividends = []
        for r in raw:
            dividends.append(DividendRecord(
                ticker=r.get("ticker", ""),
                declaration_date=r.get("declaration_date"),
                ex_dividend_date=r.get("ex_dividend_date", "1900-01-01"),
                record_date=r.get("record_date"),
                pay_date=r.get("pay_date"),
                cash_amount=r.get("cash_amount", 0),
                frequency=r.get("frequency"),
                distribution_type=r.get("distribution_type"),
                currency=r.get("currency"),
                historical_adjustment_factor=r.get("historical_adjustment_factor"),
                split_adjusted_cash_amount=r.get("split_adjusted_cash_amount"),
            ))

        log.info("get_dividends", count=len(dividends))
        return dividends

    def get_all_corporate_actions(
        self,
        *,
        ticker: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> dict[str, list]:
        """Convenience: fetch both splits and dividends for a ticker/date range.

        Returns dict with 'splits' and 'dividends' keys.
        """
        splits = self.get_splits(
            ticker=ticker,
            execution_date_gte=start_date,
            execution_date_lte=end_date,
        )
        dividends = self.get_dividends(
            ticker=ticker,
            ex_dividend_date_gte=start_date,
            ex_dividend_date_lte=end_date,
        )
        return {"splits": splits, "dividends": dividends}
