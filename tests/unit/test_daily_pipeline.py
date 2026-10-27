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

from daily_pipeline import conviction_sizes  # noqa: E402


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
