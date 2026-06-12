"""Financials endpoint: SEC filings, income statement, balance sheet, cash flow."""

from __future__ import annotations

from datetime import date
from typing import Any

from app.data.massive.client import MassiveClient
from app.data.massive.schemas import FinancialRecord
from app.utils import get_logger

log = get_logger(__name__)


class FinancialsAPI:
    """Wrapper for Massive financials endpoint (/vX/reference/financials).

    NOTE: This is an experimental endpoint. Availability depends on plan.
    If not available, methods return empty results with a logged warning.
    """

    def __init__(self, client: MassiveClient) -> None:
        self._c = client

    def get_financials(
        self,
        *,
        ticker: str | None = None,
        cik: str | None = None,
        period_of_report_date: date | None = None,
        period_of_report_date_gte: date | None = None,
        period_of_report_date_lte: date | None = None,
        filing_date: date | None = None,
        filing_date_gte: date | None = None,
        filing_date_lte: date | None = None,
        timeframe: str | None = None,
        include_sources: bool = False,
        order: str = "desc",
        limit: int = 100,
    ) -> list[FinancialRecord]:
        """Fetch financial statements from /vX/reference/financials.

        Parameters
        ----------
        ticker : filter by ticker
        cik : filter by CIK
        period_of_report_date : exact date (end of fiscal period)
        filing_date : exact SEC filing date
        timeframe : annual, quarterly, ttm
        include_sources : include source_filing_url

        CRITICAL: Never use financials before filing_date for backtesting.
        The filing_date is when the data became publicly available.
        """
        params: dict[str, Any] = {"order": order, "limit": limit}
        if ticker:
            params["ticker"] = ticker
        if cik:
            params["cik"] = cik
        if period_of_report_date:
            params["period_of_report_date"] = period_of_report_date.isoformat()
        if period_of_report_date_gte:
            params["period_of_report_date.gte"] = period_of_report_date_gte.isoformat()
        if period_of_report_date_lte:
            params["period_of_report_date.lte"] = period_of_report_date_lte.isoformat()
        if filing_date:
            params["filing_date"] = filing_date.isoformat()
        if filing_date_gte:
            params["filing_date.gte"] = filing_date_gte.isoformat()
        if filing_date_lte:
            params["filing_date.lte"] = filing_date_lte.isoformat()
        if timeframe:
            params["timeframe"] = timeframe
        if include_sources:
            params["include_sources"] = "true"

        try:
            raw = self._c.get_all_pages("/vX/reference/financials", params)
        except Exception as e:
            log.warning("financials_unavailable", error=str(e))
            return []

        records = []
        for r in raw:
            records.append(FinancialRecord(
                ticker=(r.get("tickers", [None])[0]
                        if isinstance(r.get("tickers"), list) else r.get("ticker", "")),
                cik=r.get("cik"),
                company_name=r.get("company_name"),
                fiscal_period=r.get("fiscal_period"),
                fiscal_year=r.get("fiscal_year"),
                filing_date=r.get("filing_date"),
                period_of_report_date=r.get("period_of_report_date"),
                timeframe=r.get("timeframe"),
                source_filing_url=r.get("source_filing_url"),
                financials=r.get("financials", {}),
            ))

        log.info("get_financials", count=len(records))
        return records

    def get_ticker_financials(
        self,
        ticker: str,
        *,
        timeframe: str = "quarterly",
        limit: int = 20,
    ) -> list[FinancialRecord]:
        """Convenience: get recent financials for a single ticker."""
        return self.get_financials(ticker=ticker, timeframe=timeframe, limit=limit)
