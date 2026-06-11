"""Shared test fixtures for the SCAI test suite."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate a small synthetic OHLCV panel (3 tickers × 120 days)."""
    rng = np.random.default_rng(42)
    tickers = ["AAA", "BBB", "CCC"]
    n_days = 120
    base_date = date(2023, 1, 3)  # First trading day

    rows = []
    for ticker in tickers:
        price = 10.0 + rng.uniform(-2, 2)
        for d in range(n_days):
            dt = base_date + timedelta(days=d)
            if dt.weekday() >= 5:  # skip weekends
                continue
            ret = rng.normal(0.0005, 0.02)
            price *= 1 + ret
            volume = int(rng.uniform(50_000, 500_000))
            rows.append({
                "date": dt,
                "ticker": ticker,
                "open": round(price * (1 + rng.uniform(-0.005, 0.005)), 2),
                "high": round(price * (1 + abs(rng.normal(0, 0.01))), 2),
                "low": round(price * (1 - abs(rng.normal(0, 0.01))), 2),
                "close": round(price, 2),
                "volume": volume,
            })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture
def sample_fundamentals() -> pd.DataFrame:
    """Minimal pivoted fundamentals fixture."""
    return pd.DataFrame({
        "ticker": ["AAA", "AAA", "BBB", "BBB"],
        "date": pd.to_datetime(["2023-02-15", "2023-05-15", "2023-02-20", "2023-05-20"]),
        "revenue": [100_000_000, 110_000_000, 50_000_000, 55_000_000],
        "gross_profit": [40_000_000, 45_000_000, 20_000_000, 22_000_000],
        "net_income": [10_000_000, 12_000_000, 5_000_000, 6_000_000],
        "total_assets": [200_000_000, 210_000_000, 100_000_000, 105_000_000],
        "equity": [80_000_000, 85_000_000, 40_000_000, 42_000_000],
        "total_liabilities": [120_000_000, 125_000_000, 60_000_000, 63_000_000],
        "current_assets": [50_000_000, 55_000_000, 25_000_000, 27_000_000],
        "current_liabilities": [30_000_000, 32_000_000, 15_000_000, 16_000_000],
        "cash": [15_000_000, 18_000_000, 8_000_000, 9_000_000],
    })


@pytest.fixture
def sample_universe() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["AAA", "BBB", "CCC", "DDD"],
        "name": ["Alpha Co", "Beta Inc", "Charlie LLC", "Delta OTC"],
        "exchange": ["XNAS", "XNYS", "XNAS", "OTC"],
        "market_cap": [500_000_000, 800_000_000, 100_000_000, 300_000_000],
        "sector": ["Technology", "Healthcare", "Finance", "Energy"],
        "is_active": [True, True, True, True],
    })
