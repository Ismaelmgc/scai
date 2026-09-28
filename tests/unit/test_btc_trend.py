"""BTC trend paper book: rule hysteresis, fills with costs, idempotent catch-up."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import btc_trend_daily as bt  # noqa: E402


def _fresh() -> dict:
    return {"initial_capital": 1000.0, "cash": 1000.0, "positions": [], "closed_trades": [],
            "current_day_idx": 0, "last_update": "2025-12-31", "pending_signals": [],
            "max_positions": 1}


def test_regime_band_hysteresis():
    flat = [100.0] * bt.SMA_N
    # +2% stays out (inside the band), +5% enters, a dip to -2% holds, -6% exits
    close = pd.Series(flat + [102.0, 105.0, 98.5, 94.0])
    target, _ = bt.regime(close)
    assert target.tolist()[-4:] == [0, 1, 1, 0]


def test_regime_no_signal_before_sma_warmup():
    target, _ = bt.regime(pd.Series(np.linspace(1, 1000, bt.SMA_N - 1)))
    assert target.sum() == 0


def test_buy_then_sell_applies_costs_and_records_trade():
    st = _fresh()
    r = bt.process_day(st, pd.Timestamp("2026-01-01"), 100.0, 1, 90.0)
    assert r["action"] == "BUY" and st["cash"] == 0.0
    p = st["positions"][0]
    assert p["entry_price"] == pytest.approx(100 * (1 + bt.COST), abs=0.01)
    assert p["ticker"] == bt.TICKER
    # trail_trigger (high*(1-pct)) must show the exit level SMA*(1-BAND)
    exit_level = 90 * (1 - bt.BAND)
    assert p["high_price"] * (1 - p["trailing_stop_pct"]) == pytest.approx(exit_level, rel=1e-3)

    bt.process_day(st, pd.Timestamp("2026-01-02"), 120.0, 1, 95.0)
    assert st["positions"][0]["high_price"] == 120.0

    r = bt.process_day(st, pd.Timestamp("2026-01-03"), 110.0, 0, 115.0)
    assert r["action"] == "SELL" and not st["positions"]
    t = st["closed_trades"][0]
    assert t["exit_price"] == pytest.approx(110 * (1 - bt.COST), abs=0.01)
    assert t["days_held"] == 2 and t["exit_reason"] == "tendencia_rota"
    assert st["cash"] == pytest.approx(1000 / (100 * (1 + bt.COST)) * 110 * (1 - bt.COST))
    assert np.isfinite(st["cash"])


def test_hold_and_cash_days_do_nothing():
    st = _fresh()
    assert bt.process_day(st, pd.Timestamp("2026-01-01"), 100.0, 0, 110.0)["action"] is None
    assert st["cash"] == 1000.0 and st["last_update"] == "2026-01-01"


def test_catch_up_is_ordered_and_idempotent(monkeypatch):
    idx = pd.date_range("2025-06-01", periods=bt.SMA_N + 30, freq="D")
    close = pd.Series(np.r_[np.full(bt.SMA_N, 100.0), np.linspace(101, 130, 30)], index=idx)
    data = pd.DataFrame({"open": close, "high": close, "low": close, "close": close})
    target, sma = bt.regime(data["close"])
    st = _fresh() | {"last_update": str(idx[-5].date())}
    todo = [d for d in data.index if d > pd.Timestamp(st["last_update"])]
    for d in todo:
        bt.process_day(st, d, float(data.at[d, "close"]), int(target[d]), float(sma[d]))
    assert st["last_update"] == str(idx[-1].date()) and st["current_day_idx"] == 4
    # a second run over the same candles has nothing to process
    assert not [d for d in data.index if d > pd.Timestamp(st["last_update"])]
