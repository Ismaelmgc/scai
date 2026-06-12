"""Aggregates endpoints: custom bars, grouped daily, open-close, previous close."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pytz

from app.data.massive.client import MassiveClient
from app.data.massive.schemas import DailyBar, DailyOpenClose, GroupedDailyBar
from app.utils import get_logger

log = get_logger(__name__)

_ET = pytz.timezone("America/New_York")


def _ts_to_trading_date(timestamp_ms: int) -> date:
    """Convert Unix timestamp (ms) to Eastern Time trading date."""
    dt_utc = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
    dt_et = dt_utc.astimezone(_ET)
    return dt_et.date()


class AggregatesAPI:
    """Wrapper for Massive aggregates (OHLCV) endpoints."""

    def __init__(self, client: MassiveClient) -> None:
        self._c = client

    def get_custom_bars(
        self,
        ticker: str,
        *,
        multiplier: int = 1,
        timespan: str = "day",
        from_date: date | str | None = None,
        to_date: date | str | None = None,
        adjusted: bool = True,
        sort: str = "asc",
        limit: int = 50000,
    ) -> list[DailyBar]:
        """Fetch aggregate bars from /v2/aggs/ticker/{ticker}/range/...

        Parameters
        ----------
        ticker : str
        multiplier : int (1)
        timespan : str (day, minute, hour, week, month, quarter, year)
        from_date, to_date : date or str (YYYY-MM-DD)
        adjusted : bool (True)
        sort : str (asc/desc)
        limit : int (max 50000)

        Returns list of DailyBar (also used for intraday, trading_date derived from timestamp).
        """
        from_str = (from_date.isoformat() if isinstance(from_date, date)
                    else (from_date or "2010-01-01"))
        to_str = (to_date.isoformat() if isinstance(to_date, date)
                  else (to_date or date.today().isoformat()))

        path = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_str}/{to_str}"
        params: dict[str, Any] = {
            "adjusted": str(adjusted).lower(),
            "sort": sort,
            "limit": limit,
        }

        # This endpoint uses next_url for pagination
        all_bars: list[DailyBar] = []
        data = self._c.get(path, params)
        results = data.get("results", [])

        for r in results:
            ts = r.get("t", 0)
            bar = DailyBar(
                ticker=ticker,
                trading_date=_ts_to_trading_date(ts) if ts else date.today(),
                open=r.get("o", 0),
                high=r.get("h", 0),
                low=r.get("l", 0),
                close=r.get("c", 0),
                volume=r.get("v", 0),
                vwap=r.get("vw"),
                transactions=r.get("n"),
                timestamp_ms=ts,
                adjusted=adjusted,
            )
            all_bars.append(bar)

        # Follow pagination
        while data.get("next_url"):
            data = self._c.get(data["next_url"], _is_next_url=True)
            for r in data.get("results", []):
                ts = r.get("t", 0)
                bar = DailyBar(
                    ticker=ticker,
                    trading_date=_ts_to_trading_date(ts) if ts else date.today(),
                    open=r.get("o", 0),
                    high=r.get("h", 0),
                    low=r.get("l", 0),
                    close=r.get("c", 0),
                    volume=r.get("v", 0),
                    vwap=r.get("vw"),
                    transactions=r.get("n"),
                    timestamp_ms=ts,
                    adjusted=adjusted,
                )
                all_bars.append(bar)

        log.info("get_custom_bars", ticker=ticker, bars=len(all_bars), timespan=timespan)
        return all_bars

    def get_grouped_daily(
        self,
        date_: date | str,
        *,
        adjusted: bool = True,
        include_otc: bool = False,
    ) -> list[GroupedDailyBar]:
        """Fetch all tickers for a single date from /v2/aggs/grouped/locale/us/market/stocks/{date}.

        Useful for bulk backfill — one call gives all tickers for that day.
        """
        date_str = date_.isoformat() if isinstance(date_, date) else date_
        path = f"/v2/aggs/grouped/locale/us/market/stocks/{date_str}"
        params: dict[str, Any] = {
            "adjusted": str(adjusted).lower(),
            "include_otc": str(include_otc).lower(),
        }

        data = self._c.get(path, params)
        results = data.get("results", [])

        bars = []
        for r in results:
            bars.append(GroupedDailyBar(
                ticker=r.get("T", ""),
                trading_date=date_ if isinstance(date_, date) else date.fromisoformat(date_str),
                open=r.get("o", 0),
                high=r.get("h", 0),
                low=r.get("l", 0),
                close=r.get("c", 0),
                volume=r.get("v", 0),
                vwap=r.get("vw"),
                transactions=r.get("n"),
            ))

        log.info("get_grouped_daily", date=date_str, tickers=len(bars))
        return bars

    def get_daily_open_close(
        self,
        ticker: str,
        date_: date | str,
        *,
        adjusted: bool = True,
    ) -> DailyOpenClose | None:
        """Fetch daily open-close from /v1/open-close/{ticker}/{date}.

        Includes pre-market and after-hours prices.
        """
        date_str = date_.isoformat() if isinstance(date_, date) else date_
        path = f"/v1/open-close/{ticker}/{date_str}"
        params: dict[str, Any] = {"adjusted": str(adjusted).lower()}

        data = self._c.get(path, params)
        if data.get("status") == "NOT_FOUND":
            return None

        return DailyOpenClose(
            ticker=ticker,
            trading_date=date_ if isinstance(date_, date) else date.fromisoformat(date_str),
            open=data.get("open", 0),
            high=data.get("high", 0),
            low=data.get("low", 0),
            close=data.get("close", 0),
            volume=data.get("volume", 0),
            after_hours=data.get("afterHours"),
            pre_market=data.get("preMarket"),
        )

    def get_previous_close(
        self,
        ticker: str,
        *,
        adjusted: bool = True,
    ) -> DailyBar | None:
        """Fetch previous day close from /v2/aggs/ticker/{ticker}/prev.

        Returns single bar for the most recent trading day.
        """
        path = f"/v2/aggs/ticker/{ticker}/prev"
        params: dict[str, Any] = {"adjusted": str(adjusted).lower()}

        data = self._c.get(path, params)
        results = data.get("results", [])
        if not results:
            return None

        r = results[0]
        ts = r.get("t", 0)
        return DailyBar(
            ticker=ticker,
            trading_date=_ts_to_trading_date(ts) if ts else date.today(),
            open=r.get("o", 0),
            high=r.get("h", 0),
            low=r.get("l", 0),
            close=r.get("c", 0),
            volume=r.get("v", 0),
            vwap=r.get("vw"),
            transactions=r.get("n"),
            timestamp_ms=ts,
            adjusted=adjusted,
        )
