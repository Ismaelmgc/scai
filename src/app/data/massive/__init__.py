"""Massive.com (formerly Polygon.io) comprehensive data integration.

This module provides a production-grade client for the Massive REST API,
covering all major endpoints relevant to US small-cap equity research:

- Reference data (tickers, types, events, exchanges)
- Aggregates (daily, intraday, grouped)
- Snapshots (full-market, single-ticker)
- Trades (tick-level)
- Quotes (tick-level, spread calculation)
- Corporate actions (splits, dividends, IPOs)
- Financials (income, balance sheet, cash flow)
- News (sentiment, event detection)
- Flat Files (bulk historical data)
"""

from app.data.massive.aggregates import AggregatesAPI
from app.data.massive.client import MassiveClient
from app.data.massive.corporate_actions import CorporateActionsAPI
from app.data.massive.financials import FinancialsAPI
from app.data.massive.flat_files import FlatFilesAPI
from app.data.massive.news import NewsAPI
from app.data.massive.quotes import QuotesAPI
from app.data.massive.reference import ReferenceAPI
from app.data.massive.snapshots import SnapshotsAPI
from app.data.massive.trades import TradesAPI

__all__ = [
    "MassiveClient",
    "ReferenceAPI",
    "AggregatesAPI",
    "SnapshotsAPI",
    "TradesAPI",
    "QuotesAPI",
    "CorporateActionsAPI",
    "FinancialsAPI",
    "NewsAPI",
    "FlatFilesAPI",
]
