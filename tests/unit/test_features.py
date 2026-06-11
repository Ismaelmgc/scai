"""Tests for price-action features."""

import numpy as np
import pandas as pd

from app.features.price_action import compute_price_action_features


def test_price_action_returns(sample_ohlcv):
    result = compute_price_action_features(sample_ohlcv)
    # Should have return columns
    assert "ret_1d" in result.columns
    assert "ret_5d" in result.columns
    assert "ret_20d" in result.columns
    # First row per ticker should be NaN for ret_1d
    for ticker in result["ticker"].unique():
        first = result[result["ticker"] == ticker].iloc[0]
        assert pd.isna(first["ret_1d"])


def test_price_action_overnight(sample_ohlcv):
    result = compute_price_action_features(sample_ohlcv)
    assert "overnight_ret" in result.columns
    assert "intraday_ret" in result.columns


def test_price_action_gaps(sample_ohlcv):
    result = compute_price_action_features(sample_ohlcv)
    assert "gap_up" in result.columns
    assert "gap_down" in result.columns
    assert set(result["gap_up"].dropna().unique()).issubset({0, 1})


def test_price_action_reversals(sample_ohlcv):
    result = compute_price_action_features(sample_ohlcv)
    assert "reversal_1v5" in result.columns
    assert "reversal_5v20" in result.columns


def test_no_future_leakage(sample_ohlcv):
    """Returns at time t should only use data up to t."""
    result = compute_price_action_features(sample_ohlcv)
    ticker_df = result[result["ticker"] == "AAA"].reset_index(drop=True)
    # Manually check ret_1d for consistency
    for i in range(1, min(5, len(ticker_df))):
        expected = ticker_df["close"].iloc[i] / ticker_df["close"].iloc[i - 1] - 1
        actual = ticker_df["ret_1d"].iloc[i]
        np.testing.assert_almost_equal(actual, expected, decimal=6)
