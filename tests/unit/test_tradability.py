"""Tests for the point-in-time tradability gate."""
import pandas as pd
import pytest

from app.features.tradability import tradable_mask, is_stale, MIN_PRICE, MIN_ADV_USD


def _df(**cols):
    return pd.DataFrame(cols)


class TestTradableMask:
    def test_passes_liquid_stock(self):
        df = _df(close=[5.0], adv_usd_20d=[2_000_000.0])
        assert tradable_mask(df).tolist() == [True]

    def test_rejects_subpenny_zombie(self):
        # SRNE-style: $0.0006 close, $153/day dollar volume
        df = _df(close=[0.0006], adv_usd_20d=[153.0])
        assert tradable_mask(df).tolist() == [False]

    def test_rejects_cheap_but_liquid(self):
        df = _df(close=[0.80], adv_usd_20d=[5_000_000.0])
        assert tradable_mask(df).tolist() == [False]

    def test_rejects_pricey_but_illiquid(self):
        df = _df(close=[25.0], adv_usd_20d=[50_000.0])
        assert tradable_mask(df).tolist() == [False]

    def test_nan_counts_as_untradable(self):
        df = _df(close=[float("nan"), 5.0], adv_usd_20d=[1e6, float("nan")])
        assert tradable_mask(df).tolist() == [False, False]

    def test_boundary_values_inclusive(self):
        df = _df(close=[MIN_PRICE], adv_usd_20d=[float(MIN_ADV_USD)])
        assert tradable_mask(df).tolist() == [True]

    def test_custom_thresholds(self):
        df = _df(close=[2.0, 2.0], adv_usd_20d=[400_000.0, 600_000.0])
        assert tradable_mask(df, min_price=2.0, min_adv_usd=500_000).tolist() == [False, True]

    def test_missing_column_raises(self):
        with pytest.raises(KeyError):
            tradable_mask(_df(close=[5.0]))


class TestIsStale:
    def test_fresh_same_day(self):
        d = pd.Timestamp("2026-06-11")
        assert not is_stale(d, d)

    def test_weekend_gap_ok(self):
        # Friday features on Monday run = 3 calendar days -> fresh
        assert not is_stale(pd.Timestamp("2026-06-05"), pd.Timestamp("2026-06-08"))

    def test_stale_after_threshold(self):
        assert is_stale(pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-11"))
