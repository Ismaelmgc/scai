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
