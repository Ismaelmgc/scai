"""Tests for PaperTrader exit logic (V4 profit target + label precedence)."""
import pandas as pd

from app.paper_trading import PaperTrader, PortfolioState


def _position(entry_price: float, high_price: float | None = None) -> dict:
    return {
        "ticker": "TEST", "side": "long", "shares": 10,
        "entry_price": entry_price, "entry_date": "2026-06-01",
        "entry_day_idx": 0, "trailing_stop_pct": 0.16,
        "high_price": high_price or entry_price, "low_price": entry_price,
        "holding_period_days": 20,
    }


def _trader(pos: dict, tmp_path, **kwargs) -> PaperTrader:
    state = PortfolioState(positions=[pos], current_day_idx=1)
    return PaperTrader(state, tmp_path / "p.json", **kwargs)


def _ohlcv(price: float) -> pd.DataFrame:
    return pd.DataFrame({"ticker": ["TEST"], "date": ["2026-06-02"], "close": [price]})


class TestProfitTarget:
    def test_exits_at_target(self, tmp_path):
        pt = _trader(_position(10.0), tmp_path, profit_target=0.40)
        closed = pt.update_positions(_ohlcv(14.10), "2026-06-02")
        assert len(closed) == 1
        assert closed[0].exit_reason == "profit_target"

    def test_no_exit_below_target(self, tmp_path):
        pt = _trader(_position(10.0), tmp_path, profit_target=0.40)
        closed = pt.update_positions(_ohlcv(13.50), "2026-06-02")
        assert closed == []
        assert len(pt.state.positions) == 1

    def test_disabled_by_default(self, tmp_path):
        pt = _trader(_position(10.0), tmp_path)
        closed = pt.update_positions(_ohlcv(15.0), "2026-06-02")
        assert closed == []  # +50% but no target configured, no trail hit

    def test_no_cooldown_after_profit_target(self, tmp_path):
        # Cooldown must apply only to trailing-stop exits
        pt = _trader(_position(10.0), tmp_path, profit_target=0.40)
        pt.update_positions(_ohlcv(14.10), "2026-06-02")
        assert "TEST" not in pt.state.cooldown_until


class TestTrailingStopStillWorks:
    def test_trailing_stop_exit_sets_cooldown(self, tmp_path):
        pos = _position(10.0, high_price=12.0)  # trail trigger = 12*(1-.16) = 10.08
        pt = _trader(pos, tmp_path, profit_target=0.40)
        closed = pt.update_positions(_ohlcv(10.0), "2026-06-02")
        assert len(closed) == 1
        assert closed[0].exit_reason == "trailing_stop"
        assert pt.state.cooldown_until.get("TEST", 0) > 0


class TestSplitReconciliation:
    """Retroactive split adjustment (Polygon adjusted=True) must not read as a
    loss. Regression for WLFC's real 3:1 split (2026-07-21) that fired a fake
    −71% trailing stop in both small-cap books."""

    def test_three_to_one_split_no_fake_stop(self, tmp_path):
        # Entered pre-split @217.74; the split rescales the panel to 1/3, so the
        # entry bar's adjusted open ≈72.58 and today's close ≈63. Uncorrected
        # this reads as −71% and fires a trailing stop.
        pos = {
            "ticker": "WLFC", "side": "LONG", "shares": 0.574,
            "entry_price": 217.74, "entry_date": "2026-07-15",
            "entry_day_idx": 0, "trailing_stop_pct": 0.16,
            "high_price": 217.74, "low_price": 217.74,
            "stop_loss_price": round(217.74 * 0.84, 4), "holding_period_days": 20,
        }
        pt = PaperTrader(PortfolioState(positions=[pos], current_day_idx=3),
                         tmp_path / "p.json")
        ohlcv = pd.DataFrame({
            "ticker": ["WLFC", "WLFC"], "date": ["2026-07-15", "2026-07-20"],
            "open": [72.58, 63.0], "close": [72.58, 63.03], "volume": [1e6, 1e6],
        })
        closed = pt.update_positions(ohlcv, "2026-07-20")
        assert closed == []                        # no fake trailing stop
        assert len(pt.state.positions) == 1
        p = pt.state.positions[0]
        assert 70.0 < p["entry_price"] < 75.0      # rescaled ≈ /3
        assert 1.70 < p["shares"] < 1.75           # rescaled ≈ *3
        assert abs(p["shares"] * p["entry_price"] - 0.574 * 217.74) < 1.0  # value kept

    def test_no_split_leaves_position_and_exits_normally(self, tmp_path):
        # Entry bar unchanged -> factor ≈1 -> no rescale; a real drop below the
        # trail still exits.
        pos = {
            "ticker": "TEST", "side": "LONG", "shares": 10.0,
            "entry_price": 10.0, "entry_date": "2026-06-01",
            "entry_day_idx": 0, "trailing_stop_pct": 0.16,
            "high_price": 12.0, "low_price": 10.0,
            "stop_loss_price": 8.4, "holding_period_days": 20,
        }
        pt = PaperTrader(PortfolioState(positions=[pos], current_day_idx=3),
                         tmp_path / "p.json")
        ohlcv = pd.DataFrame({
            "ticker": ["TEST", "TEST"], "date": ["2026-06-01", "2026-06-02"],
            "open": [10.0, 10.0], "close": [10.0, 10.0], "volume": [1e6, 1e6],
        })
        closed = pt.update_positions(ohlcv, "2026-06-02")
        assert len(closed) == 1
        assert closed[0].exit_reason == "trailing_stop"
        assert abs(closed[0].entry_price - 10.0) < 0.01  # entry NOT rescaled
