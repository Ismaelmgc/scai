"""BTC trend <-> managed-futures book: rule hysteresis, fills with costs, pending
rotation orders, idempotent catch-up."""
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
            "max_positions": 1, "pending_orders": []}


def _tickers(st: dict) -> list[str]:
    return [p["ticker"] for p in st["positions"]]


def test_regime_band_hysteresis():
    flat = [100.0] * bt.SMA_N
    # +2% stays out (inside the band), +5% enters, a dip to -2% holds, -6% exits
    close = pd.Series(flat + [102.0, 105.0, 98.5, 94.0])
    target, _ = bt.regime(close)
    assert target.tolist()[-4:] == [0, 1, 1, 0]


def test_regime_no_signal_before_sma_warmup():
    target, _ = bt.regime(pd.Series(np.linspace(1, 1000, bt.SMA_N - 1)))
    assert target.sum() == 0


def test_entry_from_cash_buys_btc_at_the_close():
    st = _fresh()
    r = bt.process_day(st, pd.Timestamp("2026-01-01"), 100.0, 1, 90.0)
    assert r["action"] == "BUY" and st["cash"] == 0.0 and _tickers(st) == [bt.TICKER]
    p = st["positions"][0]
    assert p["entry_price"] == pytest.approx(100 * (1 + bt.COST))
    # trail_trigger (high*(1-pct)) must show the exit level SMA*(1-BAND)
    exit_level = 90 * (1 - bt.BAND)
    assert p["high_price"] * (1 - p["trailing_stop_pct"]) == pytest.approx(exit_level, rel=1e-3)


def test_exit_sells_btc_now_and_buys_cta_at_next_open():
    st = _fresh()
    bt.process_day(st, pd.Timestamp("2026-01-01"), 100.0, 1, 90.0)
    r = bt.process_day(st, pd.Timestamp("2026-01-02"), 110.0, 0, 115.0)
    assert r["action"] == "SELL" and not st["positions"]
    assert st["closed_trades"][0]["exit_price"] == pytest.approx(110 * (1 - bt.COST))
    assert st["pending_orders"] == [{"action": "buy_cta", "after": "2026-01-03"}]
    cash = st["cash"]
    # not due yet on the signal day's own session
    assert bt.execute_pending(st, pd.Timestamp("2026-01-02"), 50.0, 111.0) == []
    bt.execute_pending(st, pd.Timestamp("2026-01-05"), 50.0, 111.0)
    assert _tickers(st) == [bt.CTA] and st["cash"] == 0.0 and not st["pending_orders"]
    assert st["positions"][0]["shares"] == pytest.approx(cash / (50 * (1 + bt.ETF_COST)))
    assert np.isfinite(st["positions"][0]["shares"])


def test_entry_from_cta_rotates_at_next_open():
    st = _fresh()
    bt._open(st, bt.CTA, 50.0, "2026-01-01")
    r = bt.process_day(st, pd.Timestamp("2026-01-09"), 120.0, 1, 100.0)
    assert r["action"] == "SEÑAL" and _tickers(st) == [bt.CTA]
    # a second "in" candle must not queue a duplicate order
    bt.process_day(st, pd.Timestamp("2026-01-10"), 121.0, 1, 100.0)
    assert len(st["pending_orders"]) == 1
    bt.execute_pending(st, pd.Timestamp("2026-01-12"), 55.0, 125.0)
    assert _tickers(st) == [bt.TICKER] and not st["pending_orders"]
    t = st["closed_trades"][0]
    assert t["ticker"] == bt.CTA and t["exit_reason"] == "rotacion_a_btc"
    assert t["exit_price"] == pytest.approx(55 * (1 - bt.ETF_COST))
    assert st["positions"][0]["entry_price"] == pytest.approx(125 * (1 + bt.COST))


def test_signal_flip_back_cancels_unfilled_rotation():
    st = _fresh()
    bt._open(st, bt.CTA, 50.0, "2026-01-01")
    bt.process_day(st, pd.Timestamp("2026-01-09"), 120.0, 1, 100.0)
    bt.process_day(st, pd.Timestamp("2026-01-10"), 95.0, 0, 100.0)
    assert not st["pending_orders"] and _tickers(st) == [bt.CTA]


def test_reentry_from_cash_cancels_pending_cta_buy():
    st = _fresh()
    bt.process_day(st, pd.Timestamp("2026-01-01"), 100.0, 1, 90.0)
    bt.process_day(st, pd.Timestamp("2026-01-02"), 80.0, 0, 90.0)      # out -> buy_cta queued
    bt.process_day(st, pd.Timestamp("2026-01-03"), 99.0, 1, 90.0)      # back in, ETF not opened yet
    assert _tickers(st) == [bt.TICKER] and not st["pending_orders"]


def test_paris_open_is_dst_aware():
    assert bt.paris_open_utc(pd.Timestamp("2026-07-01")) == pd.Timestamp("2026-07-01 07:00")
    assert bt.paris_open_utc(pd.Timestamp("2026-01-15")) == pd.Timestamp("2026-01-15 08:00")


def test_run_days_orders_fills_before_the_candle_and_is_idempotent():
    idx = pd.date_range("2026-01-01", periods=6, freq="D")
    data = pd.DataFrame({"open": [100, 101, 102, 103, 104, 105.0],
                         "close": [101, 102, 103, 104, 105, 106.0]}, index=idx)
    target = pd.Series([0, 0, 1, 1, 1, 1], index=idx)
    sma = pd.Series(100.0, index=idx)
    cta = pd.DataFrame({"open": [50.0] * 6, "close": [50.0] * 6}, index=idx)
    st = _fresh() | {"last_update": "2025-12-31"}
    bt._open(st, bt.CTA, 50.0, "2025-12-01")
    # no hourly data -> the fill falls back to that day's daily-candle open
    results, fills = bt.run_days(st, data, target, sma, cta, pd.Series(dtype=float))
    assert fills == ["CTA→BTC 2026-01-04"]
    assert st["positions"][0]["entry_price"] == pytest.approx(103 * (1 + bt.COST))
    assert st["last_update"] == "2026-01-06" and len(results) == 6
    assert bt.run_days(st, data, target, sma, cta, pd.Series(dtype=float)) == ([], [])
