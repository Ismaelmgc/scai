"""Sentiment features from Polygon.io news API.

Two modes of operation:
  1. BULK download (no ticker filter) → aggregate + per-ticker features
  2. Per-ticker features for daily live pipeline

Features produced:
  - Per-ticker: news_count_7d, sentiment_score_7d, sentiment_pos_7d,
    sentiment_neg_7d, news_momentum_7d (change vs prior 7d), has_news
  - Market-wide: mkt_sentiment_score, mkt_news_volume, mkt_sentiment_breadth
    (% tickers with positive sentiment)

IMPORTANT: Always use published_utc <= decision_date to avoid look-ahead bias.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from app.utils import get_logger

log = get_logger(__name__)


# ── Bulk news download ──────────────────────────────────────

def download_news_bulk(
    news_api,
    from_date: str,
    to_date: str,
    max_pages_per_chunk: int = 10,
    chunk_days: int = 7,
) -> pd.DataFrame:
    """Download all news articles in date range without ticker filter.

    Downloads in weekly chunks to stay within Polygon pagination limits.
    Returns DataFrame with columns: published_date, ticker, sentiment, title.

    Rate limit: 5 calls/min on free plan → ~13s between pages.
    """
    start = pd.Timestamp(from_date)
    end = pd.Timestamp(to_date)
    all_rows: list[dict] = []

    current = start
    chunk_num = 0
    while current < end:
        chunk_end = min(current + pd.Timedelta(days=chunk_days), end)
        chunk_num += 1

        # Use date-only strings (YYYY-MM-DD) — Polygon rejects datetime with T
        articles = news_api.get_news(
            published_utc_gte=current.strftime("%Y-%m-%d"),
            published_utc_lte=chunk_end.strftime("%Y-%m-%d"),
            order="asc",
            limit=1000,
            max_pages=max_pages_per_chunk,
        )

        for a in articles:
            pub_date = a.published_utc
            if pub_date is None:
                continue
            if isinstance(pub_date, str):
                pub_date = pd.Timestamp(pub_date)
            pub_date_str = pub_date.date().isoformat()

            # Extract sentiment from insights (per-ticker)
            if a.insights:
                for ins in a.insights:
                    ticker = ins.get("ticker", "")
                    sentiment = ins.get("sentiment", "neutral")
                    if ticker:
                        all_rows.append({
                            "published_date": pub_date_str,
                            "ticker": ticker,
                            "sentiment": sentiment,
                            "title": a.title[:200] if a.title else "",
                        })
            else:
                # No insights → still record article for count/attention features
                for ticker in a.tickers[:10]:  # cap per-article tickers
                    all_rows.append({
                        "published_date": pub_date_str,
                        "ticker": ticker,
                        "sentiment": "unknown",
                        "title": a.title[:200] if a.title else "",
                    })

        log.info("news_chunk", chunk=chunk_num,
                 period=f"{current.date()} -> {chunk_end.date()}",
                 articles=len(articles), rows=len(all_rows))
        current = chunk_end

    if not all_rows:
        return pd.DataFrame(columns=["published_date", "ticker", "sentiment", "title"])

    df = pd.DataFrame(all_rows)
    df["published_date"] = pd.to_datetime(df["published_date"])
    return df


# ── Feature construction ────────────────────────────────────

def build_sentiment_features(
    news_df: pd.DataFrame,
    ohlcv: pd.DataFrame,
    lookback_days: int = 7,
) -> pd.DataFrame:
    """Build per-ticker sentiment features from news data (vectorized).

    For each (ticker, date) in ohlcv, looks back `lookback_days` into news
    to compute features. Uses published_date <= date (anti look-ahead).

    Returns DataFrame with columns: ticker, date, + sentiment features.
    """
    if news_df.empty:
        log.warning("no_news_data_for_features")
        return pd.DataFrame(columns=["ticker", "date"])

    news = news_df.copy()
    news["published_date"] = pd.to_datetime(news["published_date"])

    ohlcv = ohlcv.copy()
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    # Get unique (ticker, date) combinations from ohlcv
    ticker_dates = ohlcv[["ticker", "date"]].drop_duplicates().copy()

    # Pre-aggregate news by (ticker, published_date)
    news["is_pos"] = (news["sentiment"] == "positive").astype(int)
    news["is_neg"] = (news["sentiment"] == "negative").astype(int)

    daily_news = news.groupby(["ticker", "published_date"]).agg(
        count=("sentiment", "size"),
        pos=("is_pos", "sum"),
        neg=("is_neg", "sum"),
    ).reset_index()

    # Vectorized approach: for each ticker, reindex to full date range,
    # then use rolling sum (7-day window)
    tickers_with_news = set(daily_news["ticker"].unique())
    ohlcv_tickers = set(ticker_dates["ticker"].unique())

    all_results = []

    # Process tickers with news using rolling sums
    for ticker in ohlcv_tickers:
        t_dates = ticker_dates[ticker_dates["ticker"] == ticker][["date"]].copy()

        if ticker not in tickers_with_news:
            t_dates["ticker"] = ticker
            t_dates["news_count_7d"] = 0
            t_dates["sentiment_score_7d"] = 0.0
            t_dates["sentiment_pos_ratio_7d"] = 0.0
            t_dates["sentiment_neg_ratio_7d"] = 0.0
            t_dates["has_news_7d"] = 0
            t_dates["news_momentum_7d"] = 0.0
            all_results.append(t_dates)
            continue

        # Get this ticker's daily news, reindex to calendar dates
        t_news = daily_news[daily_news["ticker"] == ticker].set_index("published_date")
        t_news = t_news[["count", "pos", "neg"]]

        # Create a full daily index covering the OHLCV date range for this ticker
        date_min = t_dates["date"].min() - pd.Timedelta(days=lookback_days * 2)
        date_max = t_dates["date"].max()
        full_idx = pd.date_range(date_min, date_max, freq="D")
        t_news = t_news.reindex(full_idx, fill_value=0)

        # Rolling sum over lookback window
        window = f"{lookback_days}D"
        roll_count = t_news["count"].rolling(window, min_periods=1).sum()
        roll_pos = t_news["pos"].rolling(window, min_periods=1).sum()
        roll_neg = t_news["neg"].rolling(window, min_periods=1).sum()

        # Prior window (for momentum): shift by lookback_days
        prior_count = roll_count.shift(lookback_days)

        # Build a lookup DataFrame indexed by date
        lookup = pd.DataFrame({
            "count": roll_count,
            "pos": roll_pos,
            "neg": roll_neg,
            "prior_count": prior_count,
        }, index=full_idx)

        # Join with ohlcv dates for this ticker
        t_dates = t_dates.set_index("date")
        merged = t_dates.join(lookup, how="left").fillna(0)
        merged = merged.reset_index().rename(columns={"index": "date"})

        # Compute features
        merged["ticker"] = ticker
        total_sent = merged["pos"] + merged["neg"]
        merged["news_count_7d"] = merged["count"].astype(int)
        merged["sentiment_score_7d"] = ((merged["pos"] - merged["neg"]) / total_sent.clip(lower=1)).round(4)
        merged["sentiment_pos_ratio_7d"] = (merged["pos"] / merged["count"].clip(lower=1)).round(4)
        merged["sentiment_neg_ratio_7d"] = (merged["neg"] / merged["count"].clip(lower=1)).round(4)
        merged["has_news_7d"] = (merged["count"] > 0).astype(int)
        merged["news_momentum_7d"] = (
            (merged["count"] - merged["prior_count"]) / merged["prior_count"].clip(lower=1)
        ).round(4)

        all_results.append(merged[["ticker", "date", "news_count_7d", "sentiment_score_7d",
                                    "sentiment_pos_ratio_7d", "sentiment_neg_ratio_7d",
                                    "has_news_7d", "news_momentum_7d"]])

    result_df = pd.concat(all_results, ignore_index=True)
    result_df["date"] = pd.to_datetime(result_df["date"])

    n_with_news = (result_df["has_news_7d"] > 0).sum()
    log.info("ticker_sentiment_features",
             rows=len(result_df), with_news=int(n_with_news),
             coverage_pct=round(n_with_news / max(len(result_df), 1) * 100, 1))

    return result_df


def build_market_sentiment_features(
    news_df: pd.DataFrame,
    ohlcv: pd.DataFrame,
    lookback_days: int = 7,
) -> pd.DataFrame:
    """Build market-wide sentiment features (vectorized, not ticker-specific).

    These capture macro sentiment shifts that affect all small-caps.
    One row per date in ohlcv.

    Features:
      - mkt_sentiment_score: (positive - negative) / total across ALL news
      - mkt_news_volume: total articles per day (normalized)
      - mkt_sentiment_breadth: % of tickers with net positive sentiment
      - mkt_sentiment_momentum: change in mkt_sentiment vs prior week
    """
    all_dates = sorted(ohlcv["date"].unique())

    if news_df.empty:
        result_df = pd.DataFrame({"date": pd.to_datetime(all_dates)})
        for col in ["mkt_sentiment_score", "mkt_news_volume",
                     "mkt_sentiment_breadth", "mkt_sentiment_momentum"]:
            result_df[col] = 0.0
        return result_df

    news = news_df.copy()
    news["published_date"] = pd.to_datetime(news["published_date"])
    news["is_pos"] = (news["sentiment"] == "positive").astype(int)
    news["is_neg"] = (news["sentiment"] == "negative").astype(int)

    # Daily market-wide aggregates
    daily_mkt = news.groupby("published_date").agg(
        total=("sentiment", "size"),
        pos=("is_pos", "sum"),
        neg=("is_neg", "sum"),
    ).reset_index().set_index("published_date").sort_index()

    # Breadth: per-ticker daily net sentiment → % positive
    ticker_daily = news.groupby(["published_date", "ticker"]).agg(
        pos=("is_pos", "sum"),
        neg=("is_neg", "sum"),
    ).reset_index()
    ticker_daily["net_positive"] = (ticker_daily["pos"] > ticker_daily["neg"]).astype(int)
    breadth_daily = ticker_daily.groupby("published_date").agg(
        n_positive=("net_positive", "sum"),
        n_total=("ticker", "nunique"),
    ).reset_index().set_index("published_date").sort_index()
    breadth_daily["breadth"] = breadth_daily["n_positive"] / breadth_daily["n_total"].clip(lower=1)

    # Reindex to full calendar range for rolling
    date_min = pd.Timestamp(all_dates[0]) - pd.Timedelta(days=lookback_days * 2)
    date_max = pd.Timestamp(all_dates[-1])
    full_idx = pd.date_range(date_min, date_max, freq="D")

    daily_mkt = daily_mkt.reindex(full_idx, fill_value=0)
    breadth_daily = breadth_daily.reindex(full_idx)
    breadth_daily["breadth"] = breadth_daily["breadth"].fillna(0.5)

    # Rolling sums
    window = f"{lookback_days}D"
    roll_total = daily_mkt["total"].rolling(window, min_periods=1).sum()
    roll_pos = daily_mkt["pos"].rolling(window, min_periods=1).sum()
    roll_neg = daily_mkt["neg"].rolling(window, min_periods=1).sum()
    roll_breadth = breadth_daily["breadth"].rolling(window, min_periods=1).mean()

    # Score and volume
    total_sent = roll_pos + roll_neg
    score = (roll_pos - roll_neg) / total_sent.clip(lower=1)
    volume = roll_total / lookback_days

    # Momentum: score - prior_score
    prior_score = score.shift(lookback_days)
    momentum = score - prior_score.fillna(0)

    # Build lookup
    lookup = pd.DataFrame({
        "mkt_sentiment_score": score.round(4),
        "mkt_news_volume": volume.round(2),
        "mkt_sentiment_breadth": roll_breadth.round(4),
        "mkt_sentiment_momentum": momentum.round(4),
    }, index=full_idx)

    # Select only ohlcv dates
    ohlcv_dates_ts = pd.DatetimeIndex(pd.to_datetime(all_dates))
    result_df = lookup.loc[lookup.index.isin(ohlcv_dates_ts)].reset_index()
    result_df = result_df.rename(columns={"index": "date"})
    result_df["date"] = pd.to_datetime(result_df["date"])

    log.info("market_sentiment_features", rows=len(result_df))
    return result_df
