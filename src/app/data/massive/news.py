"""News endpoint: market news and ticker-specific news."""

from __future__ import annotations

from datetime import date
from typing import Any

from app.data.massive.client import MassiveClient
from app.data.massive.schemas import NewsArticle
from app.utils import get_logger

log = get_logger(__name__)


class NewsAPI:
    """Wrapper for Massive news endpoint (/v2/reference/news).

    NOTE: Availability depends on plan. If not available,
    methods return empty results with a logged warning.
    """

    def __init__(self, client: MassiveClient) -> None:
        self._c = client

    def get_news(
        self,
        *,
        ticker: str | None = None,
        published_utc_gte: str | None = None,
        published_utc_lte: str | None = None,
        order: str = "desc",
        limit: int = 1000,
        max_pages: int = 10,
    ) -> list[NewsArticle]:
        """Fetch news articles from /v2/reference/news.

        Parameters
        ----------
        ticker : filter by ticker mentioned
        published_utc_gte, published_utc_lte : ISO datetime range filters
        order : desc (newest first) or asc
        limit : results per page
        max_pages : pagination depth limit

        CRITICAL: Do NOT use news with published_utc > decision timestamp.
        This would introduce look-ahead bias.
        """
        params: dict[str, Any] = {"order": order, "limit": limit}
        if ticker:
            params["ticker"] = ticker
        if published_utc_gte:
            params["published_utc.gte"] = published_utc_gte
        if published_utc_lte:
            params["published_utc.lte"] = published_utc_lte

        try:
            raw = self._c.get_all_pages("/v2/reference/news", params, max_pages=max_pages)
        except Exception as e:
            log.warning("news_unavailable", error=str(e))
            return []

        articles = []
        for r in raw:
            articles.append(NewsArticle(
                id=r.get("id", ""),
                publisher=r.get("publisher", {}).get("name") if isinstance(r.get("publisher"), dict) else r.get("publisher"),
                title=r.get("title", ""),
                author=r.get("author"),
                article_url=r.get("article_url"),
                published_utc=r.get("published_utc"),
                tickers=r.get("tickers", []),
                keywords=r.get("keywords", []),
                description=r.get("description"),
                insights=r.get("insights", []),
            ))

        log.info("get_news", count=len(articles), ticker=ticker)
        return articles

    def get_ticker_news_count(
        self,
        ticker: str,
        *,
        published_utc_gte: str | None = None,
        published_utc_lte: str | None = None,
    ) -> int:
        """Get news count for a ticker in a time window (single page)."""
        articles = self.get_news(
            ticker=ticker,
            published_utc_gte=published_utc_gte,
            published_utc_lte=published_utc_lte,
            limit=1,
            max_pages=1,
        )
        return len(articles)

    def compute_news_features(
        self,
        ticker: str,
        *,
        as_of: str,
        lookback_days: int = 7,
    ) -> dict[str, Any]:
        """Compute news-based features for a ticker as of a given datetime.

        Returns dict with: news_count, positive_count, negative_count, sentiment_score.

        IMPORTANT: Only uses news published BEFORE as_of to prevent look-ahead bias.
        """
        from datetime import datetime, timedelta, timezone as tz

        end_dt = datetime.fromisoformat(as_of.replace("Z", "+00:00")) if isinstance(as_of, str) else as_of
        start_dt = end_dt - timedelta(days=lookback_days)

        articles = self.get_news(
            ticker=ticker,
            published_utc_gte=start_dt.isoformat(),
            published_utc_lte=end_dt.isoformat(),
            order="desc",
            max_pages=3,
        )

        positive = 0
        negative = 0
        for a in articles:
            for insight in a.insights:
                sentiment = insight.get("sentiment")
                if sentiment == "positive":
                    positive += 1
                elif sentiment == "negative":
                    negative += 1

        total = len(articles)
        sentiment_score = (positive - negative) / max(total, 1)

        return {
            "news_count": total,
            "positive_count": positive,
            "negative_count": negative,
            "sentiment_score": sentiment_score,
        }
