"""Tests for daily_pipeline catch-up day selection.

Regression guard for the 2026-06-12 bug: a freshly reset portfolio (empty
``last_update``) must NOT replay the full OHLCV history (which begins in 2021)
into the live paper-trading account. Doing so fabricated 586 backtest trades
dated 2021+ and a fake +13,300% return after the V4 reset.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from daily_pipeline import _get_missed_trading_days  # noqa: E402


def _ohlcv(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(dates), "ticker": ["X"] * len(dates)})


# Five years of (sparse) trading days from 2021 to today.
OHLCV = _ohlcv([
    "2021-06-09", "2021-06-23", "2024-01-02",
    "2026-06-08", "2026-06-09", "2026-06-10", "2026-06-11", "2026-06-12",
])


class TestFreshPortfolio:
    def test_empty_last_update_starts_today(self):
        # The bug: empty last_update fell back to ohlcv_dates[0] (2021) and
        # replayed 5 years. Must now return only today.
        missed = _get_missed_trading_days(OHLCV, "", "2026-06-12")
        assert missed == ["2026-06-12"]

    def test_empty_last_update_never_returns_old_dates(self):
        missed = _get_missed_trading_days(OHLCV, "", "2026-06-12")
        assert not any(d.startswith("2021") or d.startswith("2024") for d in missed)


class TestExistingPortfolioCatchUp:
    def test_replays_real_gap(self):
        # A portfolio last updated 06-09 that missed 06-10..06-12 must catch up
        # those days (this behaviour is intentionally preserved).
        missed = _get_missed_trading_days(OHLCV, "2026-06-09", "2026-06-12")
        assert missed == ["2026-06-10", "2026-06-11", "2026-06-12"]

    def test_no_missed_days_when_up_to_date(self):
        missed = _get_missed_trading_days(OHLCV, "2026-06-12", "2026-06-12")
        assert missed == []
