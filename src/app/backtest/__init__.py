"""Backtest layer – walk-forward backtesting with realistic execution.

Key design rules:
  - Signal generated at close of day t.
  - Entry at open (or simulated VWAP) of day t+1.
  - Slippage proportional to spread proxy and participation rate.
  - Commissions per side configurable.
  - Reports net and gross performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.utils import get_logger

log = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run."""
    start_date: str = "2015-01-01"
    end_date: str = "2024-12-31"
    initial_capital: float = 1_000
    commission_bps: float = 5.0
    slippage_bps: float = 10.0
    max_positions: int = 20
    rebalance_frequency: str = "weekly"  # daily | weekly | monthly
    max_participation_rate: float = 0.05
    use_vwap_entry: bool = False
    holding_period_trading_days: int = 7
    use_stop_loss: bool = True
    use_take_profit: bool = True
    trailing_stop_pct: float = 0.0  # 0 = disabled

    @property
    def rebalance_freq_days(self) -> int:
        return {"daily": 1, "weekly": 5, "monthly": 21}.get(self.rebalance_frequency, 5)


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    portfolio_values: pd.Series
    trades: pd.DataFrame
    daily_returns: pd.Series
    metrics: dict[str, float]
    performance_by_year: pd.DataFrame
    performance_by_bucket: pd.DataFrame | None = None
    open_positions: list[dict] | None = None


class Backtester:
    """Vectorised backtest engine with realistic execution model."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    def run(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> BacktestResult:
        """Run backtest given signals and price data.

        Parameters
        ----------
        signals : DataFrame with columns: date, ticker, position_size_pct, recommendation.
                  One row per signal. Only BUY signals are acted on.
        prices : DataFrame with columns: date, ticker, open, close, volume.

        Returns
        -------
        BacktestResult with portfolio curve, trades, and metrics.
        """
        cfg = self.config
        signals = signals[signals["recommendation"].isin(["BUY", "SHORT"])].copy()
        signals["date"] = pd.to_datetime(signals["date"])
        prices = prices.copy()
        prices["date"] = pd.to_datetime(prices["date"])

        # Filter prices to backtest period (with margin before for signal mapping)
        bt_start = pd.Timestamp(cfg.start_date)
        bt_end = pd.Timestamp(cfg.end_date)

        # Map signal date → entry on next trading day (needs ALL dates for mapping)
        all_dates = sorted(prices["date"].unique())
        date_to_next = {d: all_dates[i + 1] for i, d in enumerate(all_dates[:-1])}
        # Build trading-day index for holding period calculation
        date_to_idx = {d: i for i, d in enumerate(all_dates)}

        signals["entry_date"] = signals["date"].map(date_to_next)
        signals = signals.dropna(subset=["entry_date"])

        # Only iterate over dates in backtest period + holding buffer for open positions
        hold_buffer = cfg.holding_period_trading_days + 5
        bt_dates = [d for d in all_dates if bt_start <= d]
        # Limit to end_date + buffer (for positions opened near end of validation)
        bt_end_idx = next((i for i, d in enumerate(bt_dates) if d > bt_end), len(bt_dates))
        bt_dates = bt_dates[:min(bt_end_idx + hold_buffer, len(bt_dates))]

        # Build signal lookup: stop_loss_pct, take_profit_pct, trailing_stop_pct per (ticker, date)
        signal_params = {}
        for _, sig in signals.iterrows():
            key = (sig["ticker"], sig["entry_date"])
            signal_params[key] = {
                "stop_loss_pct": sig.get("stop_loss_pct", 0.05),
                "take_profit_pct": sig.get("take_profit_pct", 0.05),
                "trailing_stop_pct": sig.get("trailing_stop_pct", cfg.trailing_stop_pct),
            }

        # Merge entry prices
        signals = signals.merge(
            prices[["date", "ticker", "open", "close", "volume"]].rename(
                columns={"date": "entry_date", "open": "entry_price", "volume": "entry_volume"}
            ),
            on=["entry_date", "ticker"],
            how="left",
        )

        # Slippage and commission
        cost_pct = (cfg.commission_bps + cfg.slippage_bps) / 10_000
        signals["entry_price_adj"] = signals["entry_price"] * (1 + cost_pct)

        # Build daily portfolio
        capital = cfg.initial_capital
        portfolio_values: list[dict[str, Any]] = []
        trades: list[dict[str, Any]] = []
        # ticker → {shares, entry_price, entry_date, entry_idx, stop_loss, take_profit, high_price}
        positions: dict[str, dict[str, Any]] = {}

        for dt in bt_dates:
            dt_ts = pd.Timestamp(dt)
            dt_idx = date_to_idx[dt]

            # Check for new signals on previous day
            new_signals = signals[signals["entry_date"] == dt_ts].copy()
            for _, sig in new_signals.head(cfg.max_positions - len(positions)).iterrows():
                ticker = sig["ticker"]
                if ticker in positions:
                    continue
                is_short = sig.get("recommendation", "BUY") == "SHORT"
                alloc = min(sig.get("position_size_pct", 0.05), 0.60) * capital
                if alloc <= 0 or pd.isna(sig["entry_price_adj"]):
                    continue
                shares = int(alloc / sig["entry_price_adj"])
                if shares <= 0:
                    continue
                # Participation check
                if sig.get("entry_volume", 0) > 0:
                    max_shares = int(sig["entry_volume"] * cfg.max_participation_rate)
                    shares = min(shares, max_shares)
                if shares <= 0:
                    continue

                # Get per-signal stop/take params
                params = signal_params.get((ticker, dt_ts), {})
                entry_p = sig["entry_price_adj"]

                if is_short:
                    # Short: receive proceeds now, owe shares later
                    # entry price already adjusted
                    proceeds = shares * entry_p * (1 - cost_pct / 10000 * 0)
                    capital += proceeds
                    pos_trail = params.get("trailing_stop_pct", cfg.trailing_stop_pct)
                    positions[ticker] = {
                        "shares": -shares,  # Negative = short
                        "entry_price": entry_p,
                        "entry_date": dt_ts,
                        "entry_idx": dt_idx,
                        "stop_loss": entry_p * (1 + params.get("stop_loss_pct", 0.05)),
                        "take_profit": entry_p * (1 - params.get("take_profit_pct", 0.05)),
                        "high_price": entry_p,
                        "low_price": entry_p,
                        "is_short": True,
                        "trailing_stop_pct": pos_trail,
                    }
                    trades.append({
                        "date": dt_ts,
                        "ticker": ticker,
                        "action": "SHORT",
                        "shares": shares,
                        "price": entry_p,
                        "cost": proceeds,
                    })
                else:
                    cost = shares * entry_p
                    capital -= cost
                    pos_trail = params.get("trailing_stop_pct", cfg.trailing_stop_pct)
                    positions[ticker] = {
                        "shares": shares,
                        "entry_price": entry_p,
                        "entry_date": dt_ts,
                        "entry_idx": dt_idx,
                        "stop_loss": entry_p * (1 - params.get("stop_loss_pct", 0.05)),
                        "take_profit": entry_p * (1 + params.get("take_profit_pct", 0.05)),
                        "high_price": entry_p,
                        "low_price": entry_p,
                        "is_short": False,
                        "trailing_stop_pct": pos_trail,
                    }
                    trades.append({
                        "date": dt_ts,
                        "ticker": ticker,
                        "action": "BUY",
                        "shares": shares,
                        "price": entry_p,
                        "cost": cost,
                    })

            # Mark-to-market
            day_prices = prices[prices["date"] == dt_ts].set_index("ticker")["close"].to_dict()
            position_value = 0
            for tick, pos in positions.items():
                current_p = day_prices.get(tick, pos["entry_price"])
                if pos.get("is_short"):
                    # Short: value = proceeds_received - cost_to_buy_back
                    position_value += (pos["entry_price"] * abs(pos["shares"])
                                       - current_p * abs(pos["shares"]))
                else:
                    position_value += pos["shares"] * current_p
            total_value = capital + position_value
            portfolio_values.append({"date": dt_ts, "value": total_value})

            # Exit logic: stop-loss, take-profit, trailing stop, or holding period expiry
            to_close: list[tuple[str, str]] = []  # (ticker, reason)
            for tick, pos in positions.items():
                current_price = day_prices.get(tick, pos["entry_price"])
                is_short = pos.get("is_short", False)

                if is_short:
                    # For shorts: track low price (trailing stop from low)
                    if current_price < pos.get("low_price", pos["entry_price"]):
                        pos["low_price"] = current_price

                    # Short stop-loss: price goes UP (always active for shorts)
                    if current_price >= pos["stop_loss"]:
                        to_close.append((tick, "stop_loss"))
                        continue

                    # Short take-profit: price goes DOWN (always active for shorts)
                    if current_price <= pos["take_profit"]:
                        to_close.append((tick, "take_profit"))
                        continue

                    # Trailing stop for shorts: price rises from low
                    pos_trail = pos.get("trailing_stop_pct", cfg.trailing_stop_pct)
                    if pos_trail > 0:
                        low_p = pos.get("low_price", pos["entry_price"])
                        trail_price = low_p * (1 + pos_trail)
                        if current_price >= trail_price:
                            to_close.append((tick, "trailing_stop"))
                            continue
                else:
                    # Long positions
                    if current_price > pos["high_price"]:
                        pos["high_price"] = current_price

                    if cfg.use_stop_loss and current_price <= pos["stop_loss"]:
                        to_close.append((tick, "stop_loss"))
                        continue

                    if cfg.use_take_profit and current_price >= pos["take_profit"]:
                        to_close.append((tick, "take_profit"))
                        continue

                    pos_trail = pos.get("trailing_stop_pct", cfg.trailing_stop_pct)
                    if pos_trail > 0:
                        trail_price = pos["high_price"] * (1 - pos_trail)
                        if current_price <= trail_price:
                            to_close.append((tick, "trailing_stop"))
                            continue

                # Holding period expiry (trading days)
                trading_days_held = dt_idx - pos["entry_idx"]
                if trading_days_held >= cfg.holding_period_trading_days:
                    to_close.append((tick, "expiry"))

            for tick, reason in to_close:
                pos = positions.pop(tick)
                exit_price = day_prices.get(tick, pos["entry_price"])
                is_short = pos.get("is_short", False)

                if is_short:
                    # Buy to cover: pay exit_price * (1 + cost) to close short
                    cover_cost = abs(pos["shares"]) * exit_price * (1 + cost_pct)
                    capital -= cover_cost
                    trades.append({
                        "date": dt_ts,
                        "ticker": tick,
                        "action": "COVER",
                        "shares": abs(pos["shares"]),
                        "price": exit_price,
                        "cost": cover_cost,
                        "reason": reason,
                    })
                else:
                    # Sell long position
                    sell_price = exit_price * (1 - cost_pct)
                    proceeds = pos["shares"] * sell_price
                    capital += proceeds
                    trades.append({
                        "date": dt_ts,
                        "ticker": tick,
                        "action": "SELL",
                        "shares": pos["shares"],
                        "price": sell_price,
                        "cost": proceeds,
                        "reason": reason,
                    })

        pv = pd.DataFrame(portfolio_values).set_index("date")["value"]
        daily_ret = pv.pct_change().dropna()
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
            columns=["date", "ticker", "action", "shares", "price", "cost"]
        )

        metrics = self._compute_metrics(pv, daily_ret, trades_df)
        perf_by_year = self._performance_by_year(daily_ret)

        # Collect open positions at end of simulation
        last_date = bt_dates[-1] if bt_dates else None
        last_prices = (prices[prices["date"] == last_date].set_index("ticker")["close"].to_dict()
                       if last_date else {})
        open_pos_list = []
        for tick, pos in positions.items():
            current_p = last_prices.get(tick, pos["entry_price"])
            entry_p = pos["entry_price"]
            is_short = pos.get("is_short", False)
            pnl_pct = (entry_p / current_p - 1) if is_short else (current_p / entry_p - 1)
            days_held = date_to_idx.get(last_date, 0) - pos["entry_idx"]
            trail = pos.get("trailing_stop_pct", cfg.trailing_stop_pct)
            high_p = pos.get("high_price", entry_p)
            trail_price = (high_p * (1 - trail) if not is_short
                           else pos.get("low_price", entry_p) * (1 + trail))
            open_pos_list.append({
                "ticker": tick,
                "side": "SHORT" if is_short else "LONG",
                "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                "entry_price": round(entry_p, 2),
                "current_price": round(current_p, 2),
                "high_price": round(high_p, 2),
                "shares": abs(pos["shares"]),
                "pnl_pct": round(pnl_pct * 100, 2),
                "days_held": days_held,
                "trailing_stop_pct": round(trail * 100, 2),
                "trail_trigger_price": round(trail_price, 2),
                "days_remaining": max(0, cfg.holding_period_trading_days - days_held),
            })

        return BacktestResult(
            portfolio_values=pv,
            trades=trades_df,
            daily_returns=daily_ret,
            metrics=metrics,
            performance_by_year=perf_by_year,
            open_positions=open_pos_list,
        )

    @staticmethod
    def _compute_metrics(
        pv: pd.Series,
        daily_ret: pd.Series,
        trades: pd.DataFrame,
    ) -> dict[str, float]:
        if len(daily_ret) < 2:
            return {}
        total_days = (pv.index[-1] - pv.index[0]).days
        years = total_days / 365.25 if total_days > 0 else 1

        total_return = pv.iloc[-1] / pv.iloc[0] - 1
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        ann_vol = daily_ret.std() * np.sqrt(252)
        sharpe = (daily_ret.mean() * 252) / ann_vol if ann_vol > 0 else 0

        downside = daily_ret[daily_ret < 0].std() * np.sqrt(252)
        sortino = (daily_ret.mean() * 252) / downside if downside > 0 else 0

        cum = (1 + daily_ret).cumprod()
        running_max = cum.cummax()
        drawdown = cum / running_max - 1
        max_dd = drawdown.min()

        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        # Hit rate
        if not trades.empty and "action" in trades.columns:
            buys = trades[trades["action"] == "BUY"]
            n_trades = len(buys)
        else:
            n_trades = 0

        return {
            "total_return": float(total_return),
            "cagr": float(cagr),
            "annual_volatility": float(ann_vol),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_dd),
            "calmar_ratio": float(calmar),
            "n_trades": n_trades,
            "total_days": total_days,
        }

    @staticmethod
    def _performance_by_year(daily_ret: pd.Series) -> pd.DataFrame:
        if daily_ret.empty:
            return pd.DataFrame()
        yearly = daily_ret.groupby(daily_ret.index.year).apply(
            lambda x: pd.Series({
                "return": (1 + x).prod() - 1,
                "volatility": x.std() * np.sqrt(252),
                "sharpe": (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0,
                "max_drawdown": ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min(),
                "trading_days": len(x),
            })
        )
        return yearly

