"""Paper trading engine — simulated execution with persistent portfolio state.

Tracks a virtual portfolio in a JSON file, processes BUY signals by entering
at next-day open, monitors trailing stops and holding-period expiry daily,
and logs all trades to a parquet audit trail.

Usage from daily_pipeline.py:
    pt = PaperTrader.load_or_create("data/paper_trading/portfolio.json")
    pt.process_signals(today_signals, ohlcv_today)
    pt.update_positions(ohlcv_today)
    pt.save()
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from app.utils import get_logger

log = get_logger(__name__)


@dataclass
class PaperPosition:
    """A single open position in the paper portfolio."""
    ticker: str
    side: str                     # "LONG" or "SHORT"
    shares: int
    entry_price: float
    entry_date: str               # ISO date
    entry_day_idx: int            # trading-day counter at entry
    trailing_stop_pct: float
    holding_period_days: int      # max holding period
    high_price: float             # watermark for trailing stop
    low_price: float              # watermark for short trailing stop
    stop_loss_price: float
    position_size_pct: float


@dataclass
class PaperTrade:
    """A completed (closed) trade."""
    ticker: str
    side: str
    shares: int
    entry_price: float
    entry_date: str
    exit_price: float
    exit_date: str
    exit_reason: str              # trailing_stop | expiry | manual
    pnl_pct: float
    pnl_usd: float
    days_held: int


@dataclass
class PortfolioState:
    """Serializable portfolio state."""
    initial_capital: float = 1000.0
    cash: float = 1000.0
    positions: list[dict] = field(default_factory=list)
    closed_trades: list[dict] = field(default_factory=list)
    current_day_idx: int = 0
    last_update: str = ""
    # Pending signals: entered at next open
    pending_signals: list[dict] = field(default_factory=list)
    # Cooldown: ticker → day_idx when re-entry is allowed
    cooldown_until: dict[str, int] = field(default_factory=dict)
    # Config
    max_positions: int = 8
    holding_period_days: int = 20
    commission_bps: float = 5.0
    slippage_bps: float = 10.0
    cooldown_days: int = 5


class PaperTrader:
    """Paper trading engine with persistent JSON state."""

    def __init__(self, state: PortfolioState, state_path: Path, *,
                 adaptive_stop: bool = False,
                 profit_target: float | None = None) -> None:
        self.state = state
        self.state_path = state_path
        self._adaptive_stop = adaptive_stop
        self._profit_target = profit_target

    # ── Persistence ─────────────────────────────────────────

    @classmethod
    def load_or_create(
        cls,
        path: str | Path,
        initial_capital: float = 1000.0,
        max_positions: int = 8,
        holding_period_days: int = 20,
        adaptive_stop: bool = False,
        profit_target: float | None = None,
    ) -> PaperTrader:
        """Load existing portfolio or create a new one."""
        p = Path(path)
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            state = PortfolioState(**data)
            log.info("portfolio_loaded", cash=state.cash, positions=len(state.positions),
                     trades=len(state.closed_trades))
        else:
            state = PortfolioState(
                initial_capital=initial_capital,
                cash=initial_capital,
                max_positions=max_positions,
                holding_period_days=holding_period_days,
            )
            log.info("portfolio_created", capital=initial_capital)
        return cls(state, p, adaptive_stop=adaptive_stop, profit_target=profit_target)

    def save(self) -> None:
        """Persist portfolio state to JSON."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(asdict(self.state), f, indent=2, default=str)
        log.info("portfolio_saved", path=str(self.state_path))

    # ── Signal Processing ──────────────────────────────────

    def process_signals(self, signals: pd.DataFrame, today: str) -> tuple[set[str], dict[str, str]]:
        """Queue BUY signals for execution at next-day open.

        Parameters
        ----------
        signals : DataFrame with columns: ticker, recommendation, position_size_pct,
                  trailing_stop_pct, stop_loss_pct
        today : ISO date string (signal generation date)

        Returns
        -------
        (traded_tickers, skip_reasons) — which BUY signals were queued and why others were skipped.
        """
        traded_tickers: set[str] = set()
        skip_reasons: dict[str, str] = {}

        buys = signals[signals["recommendation"] == "BUY"].copy()
        if buys.empty:
            log.info("no_buy_signals", date=today)
            return traded_tickers, skip_reasons

        current_tickers = {p["ticker"] for p in self.state.positions}
        pending_tickers = {p["ticker"] for p in self.state.pending_signals}
        available_slots = self.state.max_positions - len(current_tickers) - len(pending_tickers)

        if available_slots <= 0:
            for _, sig in buys.iterrows():
                skip_reasons[sig["ticker"]] = "slots_full"
            log.info("no_slots_available", date=today, positions=len(current_tickers))
            return traded_tickers, skip_reasons

        # Queue top signals (already ranked by probability)
        queued = 0
        for _, sig in buys.iterrows():
            ticker = sig["ticker"]
            if ticker in current_tickers:
                skip_reasons[ticker] = "already_held"
                continue
            if ticker in pending_tickers:
                skip_reasons[ticker] = "already_pending"
                continue
            # Cooldown: block re-entry after trailing stop
            cd = self.state.cooldown_until.get(ticker, 0)
            if cd > self.state.current_day_idx:
                skip_reasons[ticker] = "cooldown"
                log.info("signal_blocked_cooldown", ticker=ticker,
                         days_left=cd - self.state.current_day_idx)
                continue
            if queued >= available_slots:
                skip_reasons[ticker] = "slots_full"
                continue
            self.state.pending_signals.append({
                "ticker": ticker,
                "signal_date": today,
                "position_size_pct": float(sig.get("position_size_pct", 0.25)),
                "trailing_stop_pct": float(sig.get("trailing_stop_pct", 0.16)),
                "stop_loss_pct": float(sig.get("stop_loss_pct", 0.16)),
            })
            traded_tickers.add(ticker)
            queued += 1
            log.info("signal_queued", ticker=ticker, size=sig.get("position_size_pct"))

        log.info("signals_processed", date=today, queued=queued,
                 pending=len(self.state.pending_signals))
        return traded_tickers, skip_reasons

    def execute_pending(self, ohlcv_today: pd.DataFrame, today: str) -> list[str]:
        """Execute pending signals at today's open prices.

        Call this AFTER market open data is available (t+1 from signal date).

        Returns list of tickers entered.
        """
        if not self.state.pending_signals:
            return []

        ohlcv_today = ohlcv_today.copy()
        ohlcv_today["date"] = pd.to_datetime(ohlcv_today["date"])
        today_ts = pd.Timestamp(today)
        prices = ohlcv_today[ohlcv_today["date"] == today_ts].set_index("ticker")

        cost_pct = (self.state.commission_bps + self.state.slippage_bps) / 10_000
        entered = []
        remaining = []

        for sig in self.state.pending_signals:
            ticker = sig["ticker"]
            # Check cooldown (signal may have been queued before stop triggered)
            cd = self.state.cooldown_until.get(ticker, 0)
            if cd > self.state.current_day_idx:
                log.info("pending_blocked_cooldown", ticker=ticker,
                         days_left=cd - self.state.current_day_idx)
                continue
            if ticker not in prices.index:
                # No price data → keep pending for next day (max 3 days)
                sig.setdefault("_retry", 0)
                sig["_retry"] += 1
                if sig["_retry"] <= 3:
                    remaining.append(sig)
                else:
                    log.warning("signal_expired", ticker=ticker)
                continue

            open_price = float(prices.loc[ticker, "open"])
            entry_price = open_price * (1 + cost_pct)  # slippage + commission

            alloc = sig["position_size_pct"] * self._portfolio_value(prices)
            shares = int(alloc / entry_price)
            if shares <= 0:
                continue

            # Participation check (max 5% of volume)
            volume = float(prices.loc[ticker].get("volume", 1e9))
            max_shares = int(volume * 0.05)
            shares = min(shares, max_shares)
            if shares <= 0:
                continue

            cost = shares * entry_price
            if cost > self.state.cash:
                shares = int(self.state.cash / entry_price)
                cost = shares * entry_price
            if shares <= 0:
                continue

            self.state.cash -= cost
            stop_price = entry_price * (1 - sig["trailing_stop_pct"])

            position = {
                "ticker": ticker,
                "side": "LONG",
                "shares": shares,
                "entry_price": round(entry_price, 4),
                "entry_date": today,
                "entry_day_idx": self.state.current_day_idx,
                "trailing_stop_pct": sig["trailing_stop_pct"],
                "holding_period_days": self.state.holding_period_days,
                "high_price": round(entry_price, 4),
                "low_price": round(entry_price, 4),
                "stop_loss_price": round(stop_price, 4),
                "position_size_pct": sig["position_size_pct"],
            }
            self.state.positions.append(position)
            entered.append(ticker)
            log.info("position_opened", ticker=ticker, shares=shares,
                     price=round(entry_price, 2), cost=round(cost, 2))

        self.state.pending_signals = remaining
        return entered

    def update_positions(self, ohlcv_today: pd.DataFrame, today: str) -> list[PaperTrade]:
        """Update all positions with today's prices. Check exits.

        Call this after market close data is available.

        Returns list of closed trades.
        """
        ohlcv_today = ohlcv_today.copy()
        ohlcv_today["date"] = pd.to_datetime(ohlcv_today["date"])
        today_ts = pd.Timestamp(today)
        prices = ohlcv_today[ohlcv_today["date"] == today_ts].set_index("ticker")

        self.state.current_day_idx += 1
        self.state.last_update = today
        cost_pct = (self.state.commission_bps + self.state.slippage_bps) / 10_000

        closed_trades: list[PaperTrade] = []
        remaining_positions: list[dict] = []

        for pos in self.state.positions:
            ticker = pos["ticker"]
            if ticker not in prices.index:
                remaining_positions.append(pos)
                continue

            current_price = float(prices.loc[ticker, "close"])
            exit_reason = None

            # Update watermarks
            if current_price > pos["high_price"]:
                pos["high_price"] = round(current_price, 4)

            days_held = self.state.current_day_idx - pos["entry_day_idx"]

            # Profit target: lock in runaway winners before they round-trip
            # (V4 exit sweep 2026-06-11: pt40 -> Sharpe 2.52->2.79, ret intact)
            if (self._profit_target
                    and current_price >= pos["entry_price"] * (1 + self._profit_target)):
                exit_reason = "profit_target"

            # Check trailing stop
            trail_pct = pos["trailing_stop_pct"]
            # Adaptive stop: tighten to 6% after day 5 if profitable
            if (self._adaptive_stop and trail_pct > 0 and days_held > 5
                    and current_price > pos["entry_price"]):
                trail_pct = min(trail_pct, 0.06)
            if exit_reason is None and trail_pct > 0:
                trail_trigger = pos["high_price"] * (1 - trail_pct)
                if current_price <= trail_trigger:
                    exit_reason = "trailing_stop"

            # Check holding period expiry
            if exit_reason is None and days_held >= pos["holding_period_days"]:
                exit_reason = "expiry"

            if exit_reason:
                exit_price = current_price * (1 - cost_pct)
                proceeds = pos["shares"] * exit_price
                self.state.cash += proceeds

                pnl_pct = (exit_price / pos["entry_price"]) - 1
                pnl_usd = proceeds - (pos["shares"] * pos["entry_price"])

                trade = PaperTrade(
                    ticker=ticker,
                    side=pos["side"],
                    shares=pos["shares"],
                    entry_price=pos["entry_price"],
                    entry_date=pos["entry_date"],
                    exit_price=round(exit_price, 4),
                    exit_date=today,
                    exit_reason=exit_reason,
                    pnl_pct=round(pnl_pct, 4),
                    pnl_usd=round(pnl_usd, 2),
                    days_held=days_held,
                )
                self.state.closed_trades.append(asdict(trade))
                closed_trades.append(trade)
                # Set cooldown after trailing stop exit
                if exit_reason == "trailing_stop" and self.state.cooldown_days > 0:
                    self.state.cooldown_until[ticker] = (
                        self.state.current_day_idx + self.state.cooldown_days
                    )
                log.info("position_closed", ticker=ticker, reason=exit_reason,
                         pnl_pct=f"{pnl_pct:+.2%}", pnl_usd=f"{pnl_usd:+.2f}")
            else:
                remaining_positions.append(pos)

        self.state.positions = remaining_positions
        return closed_trades

    # ── Portfolio Metrics ──────────────────────────────────

    def _portfolio_value(self, prices_index: pd.DataFrame | None = None) -> float:
        """Calculate total portfolio value (cash + positions)."""
        value = self.state.cash
        for pos in self.state.positions:
            if prices_index is not None and pos["ticker"] in prices_index.index:
                price = float(prices_index.loc[pos["ticker"], "close"])
            else:
                price = pos["entry_price"]
            value += pos["shares"] * price
        return value

    def summary(self, ohlcv: pd.DataFrame | None = None) -> dict:
        """Return portfolio summary dict."""
        prices_idx = None
        if ohlcv is not None and not ohlcv.empty:
            ohlcv = ohlcv.copy()
            ohlcv["date"] = pd.to_datetime(ohlcv["date"])
            latest = ohlcv.loc[ohlcv.groupby("ticker")["date"].idxmax()]
            prices_idx = latest.set_index("ticker")

        total_value = self._portfolio_value(prices_idx)
        total_return = (total_value / self.state.initial_capital) - 1

        # Win rate from closed trades
        closed = self.state.closed_trades
        n_trades = len(closed)
        n_wins = sum(1 for t in closed if t["pnl_pct"] > 0)
        win_rate = n_wins / n_trades if n_trades > 0 else 0

        avg_pnl = np.mean([t["pnl_pct"] for t in closed]) if closed else 0
        total_pnl_usd = sum(t["pnl_usd"] for t in closed)

        open_positions = []
        for pos in self.state.positions:
            current = pos["entry_price"]
            if prices_idx is not None and pos["ticker"] in prices_idx.index:
                current = float(prices_idx.loc[pos["ticker"], "close"])
            unrealized = (current / pos["entry_price"] - 1)
            # Compute effective trail (adaptive tightens to 6% after day 5 if profitable)
            trail_pct = pos["trailing_stop_pct"]
            days_held = self.state.current_day_idx - pos.get("entry_day_idx", 0)
            if (self._adaptive_stop and trail_pct > 0 and days_held > 5
                    and current > pos["entry_price"]):
                trail_pct = min(trail_pct, 0.06)
            open_positions.append({
                "ticker": pos["ticker"],
                "entry_date": pos["entry_date"],
                "entry_price": pos["entry_price"],
                "current_price": round(current, 2),
                "pnl_pct": f"{unrealized:+.2%}",
                "trail_trigger": round(pos["high_price"] * (1 - trail_pct), 2),
                "trail_pct": round(trail_pct * 100, 0),
            })

        return {
            "total_value": round(total_value, 2),
            "cash": round(self.state.cash, 2),
            "total_return": f"{total_return:+.2%}",
            "n_open_positions": len(self.state.positions),
            "n_closed_trades": n_trades,
            "win_rate": f"{win_rate:.0%}",
            "avg_pnl_pct": f"{avg_pnl:+.2%}",
            "total_pnl_usd": round(total_pnl_usd, 2),
            "last_update": self.state.last_update,
            "open_positions": open_positions,
            "pending_signals": len(self.state.pending_signals),
        }

    def trades_to_dataframe(self) -> pd.DataFrame:
        """Export closed trades as a DataFrame."""
        if not self.state.closed_trades:
            return pd.DataFrame()
        return pd.DataFrame(self.state.closed_trades)
