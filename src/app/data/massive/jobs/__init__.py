"""Jobs for Massive data ingestion: backfill, daily update, validation.

Each job orchestrates downloads, normalization, and storage via ParquetStore.
All jobs respect rate limits and persist metadata for auditability.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd

from app.config import get_settings
from app.data.massive.aggregates import AggregatesAPI
from app.data.massive.client import MassiveClient
from app.data.massive.corporate_actions import CorporateActionsAPI
from app.data.massive.reference import ReferenceAPI
from app.data.massive.snapshots import SnapshotsAPI
from app.data.store.parquet_store import ParquetStore
from app.utils import get_logger

log = get_logger(__name__)


def _get_client() -> MassiveClient:
    """Create client from settings."""
    settings = get_settings()
    return MassiveClient(api_key=settings.polygon_api_key)


def _get_store() -> ParquetStore:
    """Get the default parquet store."""
    return ParquetStore()


# ── Backfill Reference ──────────────────────────────────────
def backfill_reference(
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[str, int]:
    """Backfill reference data: tickers, types, exchanges, conditions.

    Downloads the full ticker universe and supporting dictionaries.
    For point-in-time, use date filtering in list_tickers.
    """
    client = _get_client()
    ref = ReferenceAPI(client)
    store = _get_store()

    counts: dict[str, int] = {}

    # 1. Ticker types
    types = ref.get_ticker_types()
    if types:
        df = pd.DataFrame([t.model_dump() for t in types])
        store.write("massive_ticker_types", df)
        counts["ticker_types"] = len(types)

    # 2. Exchanges
    exchanges = ref.get_exchanges()
    if exchanges:
        df = pd.DataFrame([e.model_dump() for e in exchanges])
        store.write("massive_exchanges", df)
        counts["exchanges"] = len(exchanges)

    # 3. Conditions
    conditions = ref.get_conditions()
    if conditions:
        df = pd.DataFrame([c.model_dump() for c in conditions])
        store.write("massive_conditions", df)
        counts["conditions"] = len(conditions)

    # 4. Full ticker list (active + inactive for survivorship)
    all_tickers = []
    for active in [True, False]:
        tickers = ref.list_tickers(active=active, limit=1000)
        all_tickers.extend(tickers)

    if all_tickers:
        df = pd.DataFrame([t.model_dump() for t in all_tickers])
        df["ingested_at"] = datetime.now(UTC)
        store.upsert("massive_tickers", df, key_cols=["ticker"])
        counts["tickers"] = len(all_tickers)

    client.close()
    log.info("backfill_reference_complete", counts=counts)
    return counts


# ── Backfill Daily Bars ─────────────────────────────────────
def backfill_daily_bars(
    *,
    tickers: list[str] | None = None,
    start_date: date = date(2020, 1, 1),
    end_date: date | None = None,
    adjusted: bool = True,
    use_grouped: bool = False,
) -> dict[str, int]:
    """Backfill daily OHLCV bars.

    Strategy:
    - If use_grouped=True: use grouped daily endpoint (1 call per date, all tickers)
    - If use_grouped=False: use custom bars per ticker (better for targeted backfill)

    For free plan (5 calls/min), grouped daily is more efficient for full-market backfills.
    """
    client = _get_client()
    aggs = AggregatesAPI(client)
    store = _get_store()
    end = end_date or date.today()

    total_bars = 0

    if use_grouped:
        # One call per trading day — efficient for full market
        current = start_date
        while current <= end:
            if current.weekday() < 5:  # Skip weekends
                bars = aggs.get_grouped_daily(current, adjusted=adjusted)
                if bars:
                    df = pd.DataFrame([b.model_dump() for b in bars])
                    df["ingested_at"] = datetime.now(UTC)
                    df["adjusted"] = adjusted
                    store.upsert("massive_daily_bars", df, key_cols=["ticker", "trading_date"])
                    total_bars += len(bars)
            current += timedelta(days=1)
    else:
        # Per-ticker — better for targeted backfill
        if not tickers:
            # Load from stored universe
            try:
                uni_df = store.read("massive_tickers")
                tickers = uni_df["ticker"].tolist()[:50]  # Limit for free plan
            except Exception:
                log.warning("no_tickers_for_backfill")
                client.close()
                return {"bars": 0}

        for ticker in tickers:
            bars = aggs.get_custom_bars(
                ticker,
                from_date=start_date,
                to_date=end,
                adjusted=adjusted,
            )
            if bars:
                df = pd.DataFrame([b.model_dump() for b in bars])
                df["ingested_at"] = datetime.now(UTC)
                store.upsert("massive_daily_bars", df, key_cols=["ticker", "trading_date"])
                total_bars += len(bars)

    client.close()
    log.info("backfill_daily_bars_complete", bars=total_bars)
    return {"bars": total_bars}


# ── Backfill Corporate Actions ──────────────────────────────
def backfill_corporate_actions(
    *,
    start_date: date = date(2010, 1, 1),
    end_date: date | None = None,
) -> dict[str, int]:
    """Backfill splits and dividends."""
    client = _get_client()
    ca = CorporateActionsAPI(client)
    store = _get_store()

    end = end_date or date.today()

    # Splits
    splits = ca.get_splits(execution_date_gte=start_date, execution_date_lte=end)
    if splits:
        df = pd.DataFrame([s.model_dump() for s in splits])
        df["ingested_at"] = datetime.now(UTC)
        store.upsert("massive_splits", df, key_cols=["ticker", "execution_date"])

    # Dividends
    dividends = ca.get_dividends(ex_dividend_date_gte=start_date, ex_dividend_date_lte=end)
    if dividends:
        df = pd.DataFrame([d.model_dump() for d in dividends])
        df["ingested_at"] = datetime.now(UTC)
        store.upsert("massive_dividends", df, key_cols=["ticker", "ex_dividend_date"])

    client.close()
    log.info("backfill_corporate_actions_complete", splits=len(splits), dividends=len(dividends))
    return {"splits": len(splits), "dividends": len(dividends)}


# ── Daily Update ────────────────────────────────────────────
def update_daily() -> dict[str, Any]:
    """Run daily market data update.

    Fetches:
    1. Previous trading day's grouped bars (all tickers)
    2. Latest snapshots
    3. Recent corporate actions (last 7 days)
    """
    client = _get_client()
    aggs = AggregatesAPI(client)
    snaps = SnapshotsAPI(client)
    ca = CorporateActionsAPI(client)
    store = _get_store()

    results: dict[str, Any] = {}

    # 1. Previous day grouped bars
    yesterday = date.today() - timedelta(days=1)
    # Find last trading day (skip weekends)
    while yesterday.weekday() >= 5:
        yesterday -= timedelta(days=1)

    bars = aggs.get_grouped_daily(yesterday, adjusted=True)
    if bars:
        df = pd.DataFrame([b.model_dump() for b in bars])
        df["ingested_at"] = datetime.now(UTC)
        df["adjusted"] = True
        store.upsert("massive_daily_bars", df, key_cols=["ticker", "trading_date"])
        results["daily_bars"] = len(bars)

    # 2. Snapshots
    snapshots = snaps.get_full_market_snapshot()
    if snapshots:
        df = pd.DataFrame([s.model_dump() for s in snapshots])
        df["ingested_at"] = datetime.now(UTC)
        df["snapshot_date"] = date.today()
        store.write("massive_snapshots", df)
        results["snapshots"] = len(snapshots)

    # 3. Recent corporate actions
    week_ago = date.today() - timedelta(days=7)
    splits = ca.get_splits(execution_date_gte=week_ago)
    if splits:
        df = pd.DataFrame([s.model_dump() for s in splits])
        df["ingested_at"] = datetime.now(UTC)
        store.upsert("massive_splits", df, key_cols=["ticker", "execution_date"])
        results["new_splits"] = len(splits)

    dividends = ca.get_dividends(ex_dividend_date_gte=week_ago)
    if dividends:
        df = pd.DataFrame([d.model_dump() for d in dividends])
        df["ingested_at"] = datetime.now(UTC)
        store.upsert("massive_dividends", df, key_cols=["ticker", "ex_dividend_date"])
        results["new_dividends"] = len(dividends)

    client.close()
    log.info("update_daily_complete", results=results)
    return results


# ── Update Snapshots ────────────────────────────────────────
def update_snapshots() -> int:
    """Fetch and store current market snapshots."""
    client = _get_client()
    snaps = SnapshotsAPI(client)
    store = _get_store()

    snapshots = snaps.get_full_market_snapshot()
    if snapshots:
        df = pd.DataFrame([s.model_dump() for s in snapshots])
        df["ingested_at"] = datetime.now(UTC)
        df["snapshot_date"] = date.today()
        store.write("massive_snapshots", df)

    client.close()
    return len(snapshots)


# ── Validate ────────────────────────────────────────────────
def validate_data() -> dict[str, list[str]]:
    """Run validation checks on stored Massive data.

    Returns dict of {check_name: [issues]} — empty lists mean pass.
    """
    store = _get_store()
    issues: dict[str, list[str]] = {}

    # 1. Daily bars validation
    bar_issues = []
    try:
        bars_df = store.read("massive_daily_bars")
        if bars_df is not None and not bars_df.empty:
            # No high < low
            bad_hl = bars_df[bars_df["high"] < bars_df["low"]]
            if not bad_hl.empty:
                bar_issues.append(f"{len(bad_hl)} bars with high < low")

            # No negative prices
            for col in ["open", "high", "low", "close"]:
                if col in bars_df.columns:
                    neg = bars_df[bars_df[col] < 0]
                    if not neg.empty:
                        bar_issues.append(f"{len(neg)} bars with negative {col}")

            # No zero volume (suspicious)
            zero_vol = bars_df[bars_df["volume"] == 0]
            if len(zero_vol) > len(bars_df) * 0.1:
                bar_issues.append(f"{len(zero_vol)} bars with zero volume (>10%)")

            # Duplicate check
            if "ticker" in bars_df.columns and "trading_date" in bars_df.columns:
                dupes = bars_df.duplicated(subset=["ticker", "trading_date"])
                if dupes.any():
                    bar_issues.append(f"{dupes.sum()} duplicate ticker+date rows")
    except Exception:
        bar_issues.append("Cannot read massive_daily_bars")
    issues["daily_bars"] = bar_issues

    # 2. Splits validation
    split_issues = []
    try:
        splits_df = store.read("massive_splits")
        if splits_df is not None and not splits_df.empty:
            dupes = splits_df.duplicated(subset=["ticker", "execution_date"])
            if dupes.any():
                split_issues.append(f"{dupes.sum()} duplicate splits")
    except Exception:
        split_issues.append("Cannot read massive_splits (may not exist yet)")
    issues["splits"] = split_issues

    # 3. Tickers validation
    ticker_issues = []
    try:
        tickers_df = store.read("massive_tickers")
        if tickers_df is not None and not tickers_df.empty:
            dupes = tickers_df.duplicated(subset=["ticker"])
            if dupes.any():
                ticker_issues.append(f"{dupes.sum()} duplicate tickers")
    except Exception:
        ticker_issues.append("Cannot read massive_tickers (may not exist yet)")
    issues["tickers"] = ticker_issues

    # Summary
    total_issues = sum(len(v) for v in issues.values())
    log.info("validate_complete", total_issues=total_issues)
    return issues


# ── Audit ───────────────────────────────────────────────────
def audit() -> dict[str, Any]:
    """Audit the current state of Massive data layer.

    Returns summary of what's stored, what's missing, coverage stats.
    """
    store = _get_store()
    summary: dict[str, Any] = {}

    datasets = [
        "massive_tickers",
        "massive_ticker_types",
        "massive_exchanges",
        "massive_conditions",
        "massive_daily_bars",
        "massive_snapshots",
        "massive_splits",
        "massive_dividends",
    ]

    for ds in datasets:
        try:
            df = store.read(ds)
            if df is not None and not df.empty:
                info: dict[str, Any] = {"rows": len(df), "columns": list(df.columns)}
                if "trading_date" in df.columns:
                    info["date_range"] = f"{df['trading_date'].min()} → {df['trading_date'].max()}"
                if "ticker" in df.columns:
                    info["unique_tickers"] = df["ticker"].nunique()
                if "ingested_at" in df.columns:
                    info["last_ingested"] = str(df["ingested_at"].max())
                summary[ds] = info
            else:
                summary[ds] = {"status": "empty"}
        except Exception:
            summary[ds] = {"status": "not_found"}

    return summary
