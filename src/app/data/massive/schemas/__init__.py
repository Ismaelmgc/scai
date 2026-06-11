"""Pydantic schemas for Massive API responses.

Each schema represents a normalized record ready for storage.
Raw responses are persisted separately for auditability.
"""

from datetime import date, datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Metadata mixin ──────────────────────────────────────────
class IngestionMeta(BaseModel):
    """Metadata attached to every stored record."""

    source: str = "massive"
    endpoint: str = ""
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(__import__("datetime").timezone.utc))
    available_at: Optional[datetime] = None
    request_id: Optional[str] = None


# ── Reference ───────────────────────────────────────────────
class TickerRecord(BaseModel):
    ticker: str
    name: Optional[str] = None
    active: bool = True
    market: str = "stocks"
    locale: str = "us"
    primary_exchange: Optional[str] = None
    type: Optional[str] = None
    currency_name: Optional[str] = None
    cik: Optional[str] = None
    composite_figi: Optional[str] = None
    share_class_figi: Optional[str] = None
    last_updated_utc: Optional[datetime] = None
    delisted_utc: Optional[datetime] = None


class TickerDetail(BaseModel):
    ticker: str
    name: Optional[str] = None
    active: bool = True
    market: str = "stocks"
    locale: str = "us"
    primary_exchange: Optional[str] = None
    type: Optional[str] = None
    currency_name: Optional[str] = None
    cik: Optional[str] = None
    composite_figi: Optional[str] = None
    share_class_figi: Optional[str] = None
    market_cap: Optional[float] = None
    sic_code: Optional[str] = None
    sic_description: Optional[str] = None
    list_date: Optional[date] = None
    delisted_utc: Optional[datetime] = None
    share_class_shares_outstanding: Optional[float] = None
    weighted_shares_outstanding: Optional[float] = None
    total_employees: Optional[int] = None
    homepage_url: Optional[str] = None
    description: Optional[str] = None
    # Point-in-time
    as_of_date: Optional[date] = None


class TickerType(BaseModel):
    code: str
    description: str
    asset_class: Optional[str] = None
    locale: Optional[str] = None


class TickerEvent(BaseModel):
    ticker: str
    event_type: str
    date: Optional[date] = None
    name: Optional[str] = None
    details: Optional[dict[str, Any]] = None


class IPORecord(BaseModel):
    ticker: Optional[str] = None
    name: Optional[str] = None
    listing_date: Optional[date] = None
    ipo_status: Optional[str] = None
    offer_price: Optional[float] = None
    shares_offered: Optional[float] = None
    primary_exchange: Optional[str] = None


# ── Market Operations ───────────────────────────────────────
class Exchange(BaseModel):
    id: int
    type: str
    asset_class: Optional[str] = None
    locale: Optional[str] = None
    name: str
    mic: Optional[str] = None
    operating_mic: Optional[str] = None
    participant_id: Optional[str] = None
    acronym: Optional[str] = None
    url: Optional[str] = None


class Condition(BaseModel):
    id: int
    type: str
    name: str
    asset_class: Optional[str] = None
    data_types: Optional[list[str]] = None
    sip_mapping: dict[str, str] | None = None
    update_rules: Optional[dict[str, Any]] = None
    description: Optional[str] = None
    exchange: Optional[int] = None
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
    vwap: Optional[float] = None
    transactions: Optional[int] = None
    timestamp_ms: Optional[int] = None
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
    vwap: Optional[float] = None
    transactions: Optional[int] = None


class DailyOpenClose(BaseModel):
    ticker: str
    trading_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    after_hours: Optional[float] = None
    pre_market: Optional[float] = None


# ── Snapshots ───────────────────────────────────────────────
class SnapshotTicker(BaseModel):
    ticker: str
    updated: Optional[int] = None  # nanosecond timestamp

    # Day
    day_open: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    day_close: Optional[float] = None
    day_volume: Optional[float] = None
    day_vwap: Optional[float] = None

    # Previous day
    prev_open: Optional[float] = None
    prev_high: Optional[float] = None
    prev_low: Optional[float] = None
    prev_close: Optional[float] = None
    prev_volume: Optional[float] = None
    prev_vwap: Optional[float] = None

    # Minute (latest)
    min_open: Optional[float] = None
    min_high: Optional[float] = None
    min_low: Optional[float] = None
    min_close: Optional[float] = None
    min_volume: Optional[float] = None
    min_vwap: Optional[float] = None

    # Last trade/quote
    last_trade_price: Optional[float] = None
    last_trade_size: Optional[float] = None
    last_trade_timestamp: Optional[int] = None

    last_quote_bid: Optional[float] = None
    last_quote_ask: Optional[float] = None
    last_quote_bid_size: Optional[float] = None
    last_quote_ask_size: Optional[float] = None
    last_quote_timestamp: Optional[int] = None

    # Derived
    change: Optional[float] = None
    change_percent: Optional[float] = None
    otc: bool = False


# ── Trades ──────────────────────────────────────────────────
class TradeRecord(BaseModel):
    ticker: str
    price: float
    size: float
    exchange: Optional[int] = None
    conditions: Optional[list[int]] = None
    correction: Optional[int] = None
    id: Optional[str] = None
    participant_timestamp: Optional[int] = None
    sip_timestamp: Optional[int] = None
    trf_id: Optional[int] = None
    trf_timestamp: Optional[int] = None
    sequence_number: Optional[int] = None
    tape: Optional[int] = None


# ── Quotes ──────────────────────────────────────────────────
class QuoteRecord(BaseModel):
    ticker: str
    bid_price: float
    bid_size: float
    bid_exchange: Optional[int] = None
    ask_price: float
    ask_size: float
    ask_exchange: Optional[int] = None
    conditions: Optional[list[int]] = None
    indicators: Optional[list[int]] = None
    participant_timestamp: Optional[int] = None
    sip_timestamp: Optional[int] = None
    trf_timestamp: Optional[int] = None
    sequence_number: Optional[int] = None
    tape: Optional[int] = None

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
    adjustment_type: Optional[str] = None
    historical_adjustment_factor: Optional[float] = None


class DividendRecord(BaseModel):
    ticker: str
    declaration_date: Optional[date] = None
    ex_dividend_date: date
    record_date: Optional[date] = None
    pay_date: Optional[date] = None
    cash_amount: float
    frequency: Optional[int] = None
    distribution_type: Optional[str] = None
    currency: Optional[str] = None
    historical_adjustment_factor: Optional[float] = None
    split_adjusted_cash_amount: Optional[float] = None


# ── Financials ──────────────────────────────────────────────
class FinancialRecord(BaseModel):
    ticker: str
    cik: Optional[str] = None
    company_name: Optional[str] = None
    fiscal_period: Optional[str] = None
    fiscal_year: Optional[str] = None
    filing_date: Optional[date] = None
    period_of_report_date: Optional[date] = None
    timeframe: Optional[str] = None
    source_filing_url: Optional[str] = None
    financials: dict[str, Any] = Field(default_factory=dict)


# ── News ────────────────────────────────────────────────────
class NewsArticle(BaseModel):
    id: str
    publisher: Optional[str] = None
    title: str
    author: Optional[str] = None
    article_url: Optional[str] = None
    published_utc: Optional[datetime] = None
    tickers: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    description: Optional[str] = None
    insights: list[dict[str, Any]] = Field(default_factory=list)
