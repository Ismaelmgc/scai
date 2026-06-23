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

from daily_pipeline import _get_missed_trading_days, conviction_sizes  # noqa: E402


def _signals(scores: list[float], tradable: list[bool] | None = None) -> pd.DataFrame:
    n = len(scores)
    df = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n)],
        "v2_score": scores,
        "_tradable": [True] * n if tradable is None else tradable,
    })
    return df.sort_values("v2_score", ascending=False)


class TestConvictionSizes:
    """Conviction-linear sizing: tilt capital toward higher-conviction names,
    weights over the top-K sum to 1, untradable names get 0."""

    def test_weights_sum_to_one_over_top_k(self):
        df = _signals([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
        sizes = conviction_sizes(df, top_k=8)
        assert abs(sizes.sum() - 1.0) < 1e-9
        assert (sizes > 0).sum() == 8  # exactly the top-8 funded

    def test_higher_score_gets_more_weight(self):
        df = _signals([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0])
        sizes = conviction_sizes(df, top_k=8)
        ordered = sizes.loc[df.index]  # df is score-desc
        # monotonically non-increasing weight as score falls
        assert all(ordered.iloc[i] >= ordered.iloc[i + 1] - 1e-12
                   for i in range(len(ordered) - 1))
        # and the top name beats flat 1/8
        assert ordered.iloc[0] > 1.0 / 8

    def test_untradable_get_zero(self):
        df = _signals([5.0, 4.0, 3.0, 2.0],
                      tradable=[True, False, True, True])
        sizes = conviction_sizes(df, top_k=8)
        untradable_idx = df[~df["_tradable"]].index
        assert (sizes.loc[untradable_idx] == 0).all()
        assert abs(sizes.sum() - 1.0) < 1e-9  # tradable still fully invested

    def test_degenerate_flat_scores_fall_back_to_equal(self):
        df = _signals([2.0, 2.0, 2.0, 2.0])
        sizes = conviction_sizes(df, top_k=8)
        funded = sizes[sizes > 0]
        assert len(funded) == 4
        assert (abs(funded - 0.25) < 1e-9).all()

    def test_no_tradable_names_returns_all_zero(self):
        df = _signals([5.0, 4.0], tradable=[False, False])
        sizes = conviction_sizes(df, top_k=8)
        assert (sizes == 0).all()


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
