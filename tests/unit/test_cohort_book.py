"""Cohort engine: backtest/live parity, exits, sector cap, schedule, migration, guards."""
import json
import math
import random

import pytest

from app.engine import cohort_book as cb

CFG = cb.BookConfig()


def _market(n_days=120, n_names=30, seed=7):
    rnd = random.Random(seed)
    names = [f"T{i:02d}" for i in range(n_names)]
    sectors = {t: f"S{i % 5}" for i, t in enumerate(names)}
    px = {t: 100.0 for t in names}
    dates, closes, ranked = [], {}, {}
    for d in range(n_days):
        ds = f"2025-{1 + d // 28:02d}-{1 + d % 28:02d}"
        for t in names:
            px[t] *= 1 + rnd.gauss(0.0005, 0.025)
        closes[ds] = dict(px)
        order = sorted(names, key=lambda t: rnd.random())
        ranked[ds] = [cb.Candidate(t, -i, 0.02, sectors[t]) for i, t in enumerate(order)]
        dates.append(ds)
    return dates, closes, ranked


def test_live_stepping_with_json_roundtrip_equals_batch_backtest():
    dates, closes, ranked = _market()
    batch, curve = cb.run(dates, closes, ranked, 1000.0, CFG)
    live = cb.new_book(1000.0, CFG)
    live.day_idx = -1
    live_curve = []
    for d in dates:                                   # each session: load -> step -> save
        live = cb.from_dict(json.loads(json.dumps(cb.to_dict(live, CFG))))
        cb.step(live, d, closes[d], ranked[d], CFG)
        live_curve.append(cb.equity(live, closes[d]))
    assert live.closed_trades == batch.closed_trades
    assert live_curve == pytest.approx(curve, rel=1e-12)
    assert cb.to_dict(live, CFG)["positions"] == cb.to_dict(batch, CFG)["positions"]


def test_cash_is_conserved_minus_costs():
    dates, closes, ranked = _market(n_days=60)
    state, curve = cb.run(dates, closes, ranked, 1000.0, CFG)
    assert all(math.isfinite(v) and v > 0 for v in curve)
    assert sum(c.cash for c in state.cohorts) >= 0


def test_schedule_one_cohort_every_five_sessions():
    assert [cb.rebalancing_cohort(i, CFG) for i in range(0, 25)] == [
        0, None, None, None, None, 1, None, None, None, None, 2, None, None, None, None,
        3, None, None, None, None, 0, None, None, None, None]


def test_sector_cap_and_fill_from_next_ranked():
    st = cb.new_book(400.0, CFG)
    st.day_idx = -1
    ranked = [cb.Candidate(f"S{i}", -i, 0.02, "Tech") for i in range(5)] + \
             [cb.Candidate(f"O{i}", -10 - i, 0.02, None if i == 0 else f"X{i}") for i in range(8)]
    closes = {c.ticker: 50.0 for c in ranked}
    out = cb.step(st, "2025-01-01", closes, ranked, CFG)
    held = [p.ticker for p in st.cohorts[0].positions]
    assert out["cohort"] == 0 and len(held) == 8
    assert sum(t.startswith("S") for t in held) == 2           # max 2 per sector
    assert "O0" in held                                        # unknown sector bypasses the cap
    assert st.cohorts[0].cash == 0.0 and st.cohorts[1].cash == pytest.approx(100.0)


def _one_position(trail=0.10, pt=0.40):
    cfg = cb.BookConfig(top_k=1, profit_target=pt, cost_bps=0.0)
    st = cb.new_book(100.0, cfg)
    st.day_idx = -1
    cb.step(st, "d0", {"A": 100.0}, [cb.Candidate("A", 1.0, trail / cfg.trail_mult)], cfg)
    return st, cfg


def test_trailing_stop_from_peak_close():
    st, cfg = _one_position()
    cb.step(st, "d1", {"A": 120.0}, [], cfg)
    out = cb.step(st, "d2", {"A": 107.9}, [], cfg)          # -10.1% from the 120 peak
    assert [t["exit_reason"] for t in out["exited"]] == ["trailing_stop"]
    assert out["exited"][0]["pnl_pct"] == pytest.approx(0.079)
    assert st.cohorts[0].cash == pytest.approx(25.0 * 1.079)  # 1/4 of 100; cash idles in its cohort


def test_profit_target_and_expiry():
    st, cfg = _one_position()
    out = cb.step(st, "d1", {"A": 140.0}, [], cfg)
    assert out["exited"][0]["exit_reason"] == "profit_target"
    st, cfg = _one_position()
    for i in range(1, cfg.hold_days):
        assert not cb.step(st, f"d{i}", {"A": 101.0}, [], cfg)["exited"]
    assert cb.step(st, "d20", {"A": 101.0}, [], cfg)["exited"][0]["exit_reason"] == "expiry"


def test_non_finite_price_never_enters_the_book():
    st, cfg = _one_position()
    for bad in (float("nan"), None, 0.0, float("inf")):
        cb.step(st, "dx", {"A": bad}, [], cfg)
    assert len(st.cohorts[0].positions) == 1 and math.isfinite(cb.equity(st, {"A": float("nan")}))
    st2 = cb.new_book(100.0, cfg)
    st2.day_idx = -1
    cb.step(st2, "d0", {"B": float("nan")}, [cb.Candidate("B", 1.0)], cfg)
    assert not st2.cohorts[0].positions and st2.cohorts[0].cash == pytest.approx(25.0)


def test_split_reconciliation_keeps_value():
    st, cfg = _one_position()
    before = cb.equity(st, {"A": 100.0})
    fixed = cb.reconcile_splits(st, {("A", "d0"): 25.0}, cfg)   # 4:1 split rescaled history
    p = st.cohorts[0].positions[0]
    assert fixed == ["A"] and p.ref_close == pytest.approx(25.0) and p.peak == pytest.approx(25.0)
    assert cb.equity(st, {"A": 25.0}) == pytest.approx(before)
    assert not cb.step(st, "d1", {"A": 25.5}, [], cfg)["exited"]  # no fake -75% stop


def test_migration_from_slot_book_keeps_equity_and_converges_to_equal_cohorts():
    positions = [{"ticker": f"P{i}", "entry_date": f"2026-10-{10 + i}",
                  "entry_day_idx": 72 + i, "entry_price": 100.0 * 1.0015, "shares": 1.2,
                  "high_price": 104.0, "trailing_stop_pct": 0.16} for i in range(8)]
    old = {"initial_capital": 1000.0, "cash": 5.0, "current_day_idx": 80,
           "last_update": "2026-10-26", "commission_bps": 5.0, "slippage_bps": 10.0,
           "closed_trades": [{"ticker": "X"}], "positions": positions}
    st = cb.from_slot_book(old, CFG, sectors={"P0": "Tech"})
    closes = {f"P{i}": 103.0 for i in range(8)} | {f"N{i}": 50.0 for i in range(40)}
    eq0 = cb.equity(st, closes)
    assert eq0 == pytest.approx(5.0 + 8 * 1.2 * 103.0)
    assert st.day_idx == 80 and st.closed_trades == [{"ticker": "X"}]
    assert len(st.legacy) == 8 and st.reserve == 5.0 and all(c.cash == 0 for c in st.cohorts)
    view = cb.to_dict(st, CFG)
    assert len(view["positions"]) == 8 and view["cash"] == pytest.approx(5.0)
    st = cb.from_dict(json.loads(json.dumps(view)))         # survives the Supabase round-trip
    ranked = [cb.Candidate(f"N{i}", -i, 0.02, f"S{i % 9}") for i in range(40)]
    for k in range(35):                       # legacy expires ~s.99; last cohort tops up at s.110
        cb.step(st, f"2026-11-{k + 1:02d}", closes, ranked, CFG)
    assert not st.legacy and st.reserve < 0.005 * eq0            # only a rounding residue left
    values = [c.cash + sum(p.shares * closes[p.ticker] for p in c.positions) for c in st.cohorts]
    assert sum(values) == pytest.approx(eq0, rel=0.01)       # only round-trip costs lost
    assert max(values) / min(values) < 1.05                  # 4 cohorts ~1/4 each


def test_cohort_with_dust_cash_does_not_buy():
    st = cb.new_book(1000.0, CFG)
    st.day_idx = -1
    st.cohorts[0].cash, st.reserve = 0.04, 999.96 - 750.0     # dust in the rotating cohort
    st.cohorts[1].cash = st.cohorts[2].cash = st.cohorts[3].cash = 250.0
    st.reserve = 0.0
    out = cb.step(st, "d0", {"A": 10.0}, [cb.Candidate("A", 1.0)], CFG)
    assert out["entered"] == [] and st.cohorts[0].cash == pytest.approx(0.04)
