"""Tests for point-in-time utilities."""

import pandas as pd

from app.utils.point_in_time import as_of, lag_safe_merge, latest_as_of


def test_as_of_filters_correctly():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"]),
        "value": [1, 2, 3],
    })
    result = as_of(df, "2023-02-15")
    assert len(result) == 2
    assert result["value"].tolist() == [1, 2]


def test_as_of_empty_result():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2023-06-01"]),
        "value": [1],
    })
    result = as_of(df, "2023-01-01")
    assert len(result) == 0


def test_latest_as_of():
    df = pd.DataFrame({
        "ticker": ["A", "A", "B", "B"],
        "date": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-01-15", "2023-02-15"]),
        "value": [1, 2, 10, 20],
    })
    result = latest_as_of(df, "2023-02-10")
    assert len(result) == 2
    assert result[result["ticker"] == "A"]["value"].iloc[0] == 2
    assert result[result["ticker"] == "B"]["value"].iloc[0] == 10


def test_lag_safe_merge():
    left = pd.DataFrame({
        "ticker": ["A", "A"],
        "date": pd.to_datetime(["2023-01-05", "2023-01-10"]),
        "price": [10, 11],
    })
    right = pd.DataFrame({
        "ticker": ["A", "A"],
        "date": pd.to_datetime(["2023-01-04", "2023-01-09"]),
        "revenue": [100, 200],
    })
    result = lag_safe_merge(left, right, on=["ticker"], date_col="date", lag_days=1)
    # Right date shifted by 1 day: 2023-01-05 and 2023-01-10
    assert "revenue" in result.columns
