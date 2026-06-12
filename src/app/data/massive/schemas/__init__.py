"""Pydantic schemas for Massive API responses.

Each schema represents a normalized record ready for storage.
Raw responses are persisted separately for auditability.
"""
from datetime import UTC, date, datetime
from datetime import date as _date  # alias: a field literally named `date` shadows the type
from typing import Any

from pydantic import BaseModel, Field


# ── Metadata mixin ──────────────────────────────────────────
class IngestionMeta(BaseModel):
    """Metadata attached to every stored record."""

    source: str = "massive"
    endpoint: str = ""
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    available_at: datetime | None = None
    request_id: str | None = None


# ── Reference ───────────────────────────────────────────────
class TickerRecord(BaseModel):
    ticker: str
    name: str | None = None
    active: bool = True
    market: str = "stocks"
    locale: str = "us"
    primary_exchange: str | None = None
    type: str | None = None
    currency_name: str | None = None
    cik: str | None = None
    composite_figi: str | None = None
    share_class_figi: str | None = None
    last_updated_utc: datetime | None = None
    delisted_utc: datetime | None = None


class TickerDetail(BaseModel):
    ticker: str
    name: str | None = None
    active: bool = True
    market: str = "stocks"
    locale: str = "us"
    primary_exchange: str | None = None
    type: str | None = None
    currency_name: str | None = None
    cik: str | None = None
    composite_figi: str | None = None
    share_class_figi: str | None = None
    market_cap: float | None = None
    sic_code: str | None = None
    sic_description: str | None = None
    list_date: date | None = None
    delisted_utc: datetime | None = None
    share_class_shares_outstanding: float | None = None
    weighted_shares_outstanding: float | None = None
    total_employees: int | None = None
    homepage_url: str | None = None
    description: str | None = None
    # Point-in-time
    as_of_date: date | None = None


class TickerType(BaseModel):
    code: str
    description: str
    asset_class: str | None = None
    locale: str | None = None


class TickerEvent(BaseModel):
    ticker: str
    event_type: str
    date: _date | None = None
    name: str | None = None
    details: dict[str, Any] | None = None


class IPORecord(BaseModel):
    ticker: str | None = None
    name: str | None = None
    listing_date: date | None = None
    ipo_status: str | None = None
    offer_price: float | None = None
    shares_offered: float | None = None
    primary_exchange: str | None = None


# ── Market Operations ───────────────────────────────────────
class Exchange(BaseModel):
    id: int
    type: str
    asset_class: str | None = None
    locale: str | None = None
    name: str
    mic: str | None = None
    operating_mic: str | None = None
    participant_id: str | None = None
    acronym: str | None = None
    url: str | None = None


class Condition(BaseModel):
    id: int
    type: str
    name: str
    asset_class: str | None = None
    data_types: list[str] | None = None
    sip_mapping: dict[str, str] | None = None
    update_rules: dict[str, Any] | None = None
    description: str | None = None
    exchange: int | None = None
    legacy: bool = False


# ── Aggregates ──────────────────────────────────────────────
class DailyBar(BaseModel):
    ticker: str
    trading_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
    transactions: int | None = None
    timestamp_ms: int | None = None
    adjusted: bool = True


class GroupedDailyBar(BaseModel):
    """One bar from grouped daily endpoint (all tickers for a date)."""

    ticker: str
    trading_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
    transactions: int | None = None


class DailyOpenClose(BaseModel):
    ticker: str
    trading_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    after_hours: float | None = None
    pre_market: float | None = None


# ── Snapshots ───────────────────────────────────────────────
class SnapshotTicker(BaseModel):
    ticker: str
    updated: int | None = None  # nanosecond timestamp

    # Day
    day_open: float | None = None
    day_high: float | None = None
    day_low: float | None = None
    day_close: float | None = None
    day_volume: float | None = None
    day_vwap: float | None = None

    # Previous day
    prev_open: float | None = None
    prev_high: float | None = None
    prev_low: float | None = None
    prev_close: float | None = None
    prev_volume: float | None = None
    prev_vwap: float | None = None

    # Minute (latest)
    min_open: float | None = None
    min_high: float | None = None
    min_low: float | None = None
    min_close: float | None = None
    min_volume: float | None = None
    min_vwap: float | None = None

    # Last trade/quote
    last_trade_price: float | None = None
    last_trade_size: float | None = None
    last_trade_timestamp: int | None = None

    last_quote_bid: float | None = None
    last_quote_ask: float | None = None
    last_quote_bid_size: float | None = None
    last_quote_ask_size: float | None = None
    last_quote_timestamp: int | None = None

    # Derived
    change: float | None = None
    change_percent: float | None = None
    otc: bool = False


# ── Trades ──────────────────────────────────────────────────
class TradeRecord(BaseModel):
    ticker: str
    price: float
    size: float
    exchange: int | None = None
    conditions: list[int] | None = None
    correction: int | None = None
    id: str | None = None
    participant_timestamp: int | None = None
    sip_timestamp: int | None = None
    trf_id: int | None = None
    trf_timestamp: int | None = None
    sequence_number: int | None = None
    tape: int | None = None


# ── Quotes ──────────────────────────────────────────────────
class QuoteRecord(BaseModel):
    ticker: str
    bid_price: float
    bid_size: float
    bid_exchange: int | None = None
    ask_price: float
    ask_size: float
    ask_exchange: int | None = None
    conditions: list[int] | None = None
    indicators: list[int] | None = None
    participant_timestamp: int | None = None
    sip_timestamp: int | None = None
    trf_timestamp: int | None = None
    sequence_number: int | None = None
    tape: int | None = None

    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price

    @property
    def mid_price(self) -> float:
        return (self.bid_price + self.ask_price) / 2

    @property
    def relative_spread(self) -> float:
        mid = self.mid_price
        if mid <= 0:
            return 0.0
        return self.spread / mid


# ── Corporate Actions ───────────────────────────────────────
class SplitRecord(BaseModel):
    ticker: str
    execution_date: date
    split_from: float
    split_to: float
    adjustment_type: str | None = None
    historical_adjustment_factor: float | None = None


class DividendRecord(BaseModel):
    ticker: str
    declaration_date: date | None = None
    ex_dividend_date: date
    record_date: date | None = None
    pay_date: date | None = None
    cash_amount: float
    frequency: int | None = None
    distribution_type: str | None = None
    currency: str | None = None
    historical_adjustment_factor: float | None = None
    split_adjusted_cash_amount: float | None = None


# ── Financials ──────────────────────────────────────────────
class FinancialRecord(BaseModel):
    ticker: str
    cik: str | None = None
    company_name: str | None = None
    fiscal_period: str | None = None
    fiscal_year: str | None = None
    filing_date: date | None = None
    period_of_report_date: date | None = None
    timeframe: str | None = None
    source_filing_url: str | None = None
    financials: dict[str, Any] = Field(default_factory=dict)


# ── News ────────────────────────────────────────────────────
class NewsArticle(BaseModel):
    id: str
    publisher: str | None = None
    title: str
    author: str | None = None
    article_url: str | None = None
    published_utc: datetime | None = None
    tickers: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    description: str | None = None
    insights: list[dict[str, Any]] = Field(default_factory=list)
