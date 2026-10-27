"""Cohort portfolio engine — the ONE implementation of the liquidcap strategy,
shared by the backtest and the live paper book.

Why one engine: the research harness and the live PaperTrader were two separate
implementations of the same strategy and they drifted apart (open- vs close-entry,
split handling, and finally the portfolio construction itself: the live book ran
8 refill-on-exit slots while the harness validated 4 overlapping cohorts, ~1.3x
the volatility). Here the live job is literally "run the backtest one more day".

Construction (what the purged walk-forward validated):
  * capital split into N_COHORTS (4) cohorts; every REBALANCE_EVERY (5) sessions
    one cohort sells whatever it still holds and buys the day's top-k (8) names
    with ALL its cash, equal-weight -> each cohort turns over every 20 sessions;
  * at most MAX_PER_SECTOR (2) names per GICS sector inside a cohort (names with an
    unknown sector bypass the cap); the shortfall is filled by the next-ranked name;
  * exits, evaluated on each session's close in this order: profit target (+40%
    vs the entry close), trailing stop from the peak close (ATR*5.3 clipped to
    [10%, 16%]), expiry after HOLD_DAYS (20) sessions;
  * an exited position's cash IDLES in its cohort until that cohort's next
    rebalance — it is never re-deployed into "the next name on the list";
  * fills at the session close +/- COST_BPS.
A non-finite price never enters the book: the position is simply held/skipped.

Migration from the old 8-slot book: its open positions become LEGACY positions
(no cohort) that run to their normal exits; their proceeds go to a shared
RESERVE, and each cohort, when it rotates, tops itself up from the reserve to
1/N of the book. In a backtest the reserve is always 0, so nothing changes.

Pure Python + dataclasses; the state round-trips through JSON (Supabase).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class BookConfig:
    n_cohorts: int = 4
    rebalance_every: int = 5
    top_k: int = 8
    hold_days: int = 20
    trail_mult: float = 5.3
    trail_min: float = 0.10
    trail_max: float = 0.16
    profit_target: float | None = 0.40
    max_per_sector: int | None = 2
    cost_bps: float = 15.0
    # A stored entry close differing from today's (split-adjusted) history by more
    # than this ratio is treated as a corporate action and rescaled.
    split_tol: float = 0.25
    # A cohort holding less than this fraction of its 1/N share doesn't buy (no dust
    # positions while a migrated book is still converging; never binds in a backtest).
    min_cohort_fraction: float = 0.01

    @property
    def cost(self) -> float:
        return self.cost_bps / 10_000


@dataclass
class Candidate:
    ticker: str
    score: float
    atr_pct: float = float("nan")
    sector: str | None = None


@dataclass
class Position:
    ticker: str
    entry_date: str
    entry_idx: int
    ref_close: float          # the entry session's close (exits are measured vs this)
    shares: float
    peak: float               # highest close since entry
    trail: float              # trailing-stop distance (fraction)
    sector: str | None = None


@dataclass
class Cohort:
    cash: float
    positions: list[Position] = field(default_factory=list)


@dataclass
class BookState:
    initial_capital: float
    day_idx: int = 0
    last_update: str = ""
    cohorts: list[Cohort] = field(default_factory=list)
    closed_trades: list[dict] = field(default_factory=list)
    reserve: float = 0.0                       # unassigned cash (migration only)
    legacy: list[Position] = field(default_factory=list)


def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x) and x > 0


def new_book(capital: float, cfg: BookConfig = BookConfig(), last_update: str = "") -> BookState:
    share = capital / cfg.n_cohorts
    return BookState(initial_capital=capital, last_update=last_update,
                     cohorts=[Cohort(cash=share) for _ in range(cfg.n_cohorts)])


def trail_for(atr_pct: float, cfg: BookConfig) -> float:
    atr = atr_pct if _finite(atr_pct) else 0.03
    return float(min(max(atr * cfg.trail_mult, cfg.trail_min), cfg.trail_max))


def rebalancing_cohort(day_idx: int, cfg: BookConfig) -> int | None:
    """Which cohort rebalances on session `day_idx` (None on the 4 sessions in between)."""
    if day_idx % cfg.rebalance_every:
        return None
    return (day_idx // cfg.rebalance_every) % cfg.n_cohorts


def _value(positions: list[Position], closes: dict[str, float]) -> float:
    return sum(p.shares * (closes[p.ticker] if _finite(closes.get(p.ticker)) else p.ref_close)
               for p in positions)


def equity(state: BookState, closes: dict[str, float]) -> float:
    return (state.reserve + _value(state.legacy, closes)
            + sum(c.cash + _value(c.positions, closes) for c in state.cohorts))


def _close(state: BookState, cohort: Cohort | None, p: Position, px: float, date: str,
           reason: str, cfg: BookConfig) -> dict:
    """Sell `p`; proceeds go to its cohort, or to the reserve for a legacy position."""
    fill = px * (1 - cfg.cost)
    if cohort is None:
        state.legacy.remove(p)
        state.reserve += p.shares * fill
    else:
        cohort.positions.remove(p)
        cohort.cash += p.shares * fill
    entry_fill = p.ref_close * (1 + cfg.cost)
    trade = {"ticker": p.ticker, "entry_date": p.entry_date, "exit_date": date,
             "entry_price": round(entry_fill, 4), "exit_price": round(fill, 4),
             "shares": p.shares, "pnl_pct": round(fill / entry_fill - 1, 6),
             "pnl_usd": round(p.shares * (fill - entry_fill), 4), "exit_reason": reason,
             "days_held": state.day_idx - p.entry_idx,
             "cohort": None if cohort is None else state.cohorts.index(cohort)}
    state.closed_trades.append(trade)
    return trade


def step(state: BookState, date: str, closes: dict[str, float],
         ranked: list[Candidate], cfg: BookConfig = BookConfig()) -> dict:
    """Advance the book by one session: exits on today's close, then (every
    `rebalance_every` sessions) one cohort rotates into today's top-k.

    `ranked` must be sorted best-first and already restricted to tradable names.
    Returns {"exited": [trade dicts], "entered": [tickers], "cohort": int|None}.
    """
    state.day_idx += 1
    exited: list[dict] = []
    holders = [(None, state.legacy)] + [(c, c.positions) for c in state.cohorts]
    for cohort, positions in holders:
        for p in list(positions):
            px = closes.get(p.ticker)
            if not _finite(px):
                continue                                  # no usable price: hold
            p.peak = max(p.peak, px)
            if cfg.profit_target and px >= p.ref_close * (1 + cfg.profit_target):
                exited.append(_close(state, cohort, p, px, date, "profit_target", cfg))
            elif (px - p.peak) / p.peak <= -p.trail:
                exited.append(_close(state, cohort, p, px, date, "trailing_stop", cfg))
            elif state.day_idx - p.entry_idx >= cfg.hold_days:
                exited.append(_close(state, cohort, p, px, date, "expiry", cfg))

    entered: list[str] = []
    ci = rebalancing_cohort(state.day_idx, cfg)
    if ci is not None:
        cohort = state.cohorts[ci]
        for p in list(cohort.positions):                  # normally none left (all expired)
            px = closes.get(p.ticker)
            exited.append(_close(state, cohort, p, px if _finite(px) else p.ref_close,
                                 date, "rebalance", cfg))
        if state.reserve > 0:                              # migration: refill toward 1/N
            target = equity(state, closes) / cfg.n_cohorts
            top_up = min(state.reserve, max(0.0, target - cohort.cash))
            cohort.cash += top_up
            state.reserve -= top_up
        picks: list[Candidate] = []
        per_sector: dict[str, int] = {}
        for cand in ranked:
            if len(picks) >= cfg.top_k:
                break
            if not _finite(closes.get(cand.ticker)) or any(x.ticker == cand.ticker for x in picks):
                continue
            sec = cand.sector
            if cfg.max_per_sector and sec and per_sector.get(sec, 0) >= cfg.max_per_sector:
                continue
            picks.append(cand)
            if sec:
                per_sector[sec] = per_sector.get(sec, 0) + 1
        floor = cfg.min_cohort_fraction * equity(state, closes) / cfg.n_cohorts
        if picks and cohort.cash > floor:
            alloc = cohort.cash / len(picks)
            for cand in picks:
                px = closes[cand.ticker]
                cohort.positions.append(Position(
                    ticker=cand.ticker, entry_date=date, entry_idx=state.day_idx, ref_close=px,
                    shares=alloc / (px * (1 + cfg.cost)), peak=px,
                    trail=trail_for(cand.atr_pct, cfg), sector=cand.sector))
                entered.append(cand.ticker)
            cohort.cash = 0.0
    state.last_update = date
    return {"exited": exited, "entered": entered, "cohort": ci}


def reconcile_splits(state: BookState, entry_closes: dict[tuple[str, str], float],
                     cfg: BookConfig = BookConfig()) -> list[str]:
    """Rescale positions whose stored entry close no longer matches the (split-
    adjusted) history: `entry_closes[(ticker, entry_date)]` is today's view of that
    session's close. A real split rescales history by the ratio, so the position
    is rescaled by the same factor (value unchanged). Returns the rescaled tickers."""
    fixed = []
    for c in state.cohorts:
        for p in c.positions:
            hist = entry_closes.get((p.ticker, p.entry_date))
            if not _finite(hist):
                continue
            f = hist / p.ref_close
            if abs(f - 1) > cfg.split_tol:
                p.ref_close *= f
                p.peak *= f
                p.shares /= f
                fixed.append(p.ticker)
    return fixed


# ---------------------------------------------------------------- (de)serialisation
def to_dict(state: BookState, cfg: BookConfig = BookConfig(),
            closes: dict[str, float] | None = None) -> dict:
    """JSON-ready state. Besides the cohorts (source of truth) it carries the flat
    fields the dashboard / live-prices worker read: cash, positions, etc."""
    d = asdict(state)
    d["engine"] = "cohorts_v1"
    d["cash"] = state.reserve + sum(c.cash for c in state.cohorts)
    d["current_day_idx"] = state.day_idx
    d["max_positions"] = cfg.n_cohorts * cfg.top_k
    d["pending_signals"] = []
    holders = [(i, c.positions) for i, c in enumerate(state.cohorts)] + [(None, state.legacy)]
    d["positions"] = [{
        "ticker": p.ticker, "entry_date": p.entry_date,
        "entry_price": round(p.ref_close * (1 + cfg.cost), 4), "shares": p.shares,
        "high_price": p.peak, "trailing_stop_pct": p.trail, "entry_day_idx": p.entry_idx,
        "cohort": i} for i, positions in holders for p in positions]
    if closes is not None:
        d["equity"] = round(equity(state, closes), 4)
    return d


def from_dict(d: dict) -> BookState:
    return BookState(
        initial_capital=float(d["initial_capital"]), day_idx=int(d["day_idx"]),
        last_update=d.get("last_update", ""),
        cohorts=[Cohort(cash=float(c["cash"]), positions=[Position(**p) for p in c["positions"]])
                 for c in d["cohorts"]],
        closed_trades=list(d.get("closed_trades", [])),
        reserve=float(d.get("reserve", 0.0)),
        legacy=[Position(**p) for p in d.get("legacy", [])])


def from_slot_book(old: dict, cfg: BookConfig = BookConfig(),
                   sectors: dict[str, str] | None = None) -> BookState:
    """One-off migration of a PaperTrader (8-slot) state into the cohort engine
    without selling anything: open positions become LEGACY positions that run to
    their normal exits, cash goes to the reserve, and cohorts fill up from the
    reserve as they rotate (converged after one full 20-session cycle). Equity,
    day index and trade history are carried over unchanged."""
    cost = (old.get("commission_bps", 5.0) + old.get("slippage_bps", 10.0)) / 10_000
    legacy = []
    for p in old.get("positions", []):
        ref = float(p["entry_price"]) / (1 + cost)        # PaperTrader stored close*(1+cost)
        legacy.append(Position(
            ticker=p["ticker"], entry_date=p["entry_date"], entry_idx=int(p["entry_day_idx"]),
            ref_close=ref, shares=float(p["shares"]),
            peak=max(float(p.get("high_price", ref)), ref),
            trail=float(p.get("trailing_stop_pct", cfg.trail_max)),
            sector=(sectors or {}).get(p["ticker"])))
    return BookState(initial_capital=float(old["initial_capital"]),
                     day_idx=int(old["current_day_idx"]), last_update=old.get("last_update", ""),
                     cohorts=[Cohort(cash=0.0) for _ in range(cfg.n_cohorts)],
                     closed_trades=list(old.get("closed_trades", [])),
                     reserve=float(old.get("cash", 0.0)), legacy=legacy)

def run(dates: list[str], closes_by_date: dict[str, dict[str, float]],
        ranked_by_date: dict[str, list[Candidate]], capital: float = 1.0,
        cfg: BookConfig = BookConfig()) -> tuple[BookState, list[float]]:
    """Backtest = the live `step` applied to every session in order. The first
    session is a rebalance day (cohort 0), like the harness. Returns the final
    state and the daily equity curve."""
    state = new_book(capital, cfg)
    state.day_idx = -1
    curve = []
    for d in dates:
        closes = closes_by_date.get(d, {})
        step(state, d, closes, ranked_by_date.get(d, []), cfg)
        curve.append(equity(state, closes))
    return state, curve
