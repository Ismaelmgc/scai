"""Signal Tracker — records every generated signal and backfills actual outcomes.

Maintains a single consolidated parquet (signal_history.parquet) that logs:
- Every BUY signal the model generates each day
- Whether it was actually traded (or skipped due to full slots, etc.)
- The actual forward return after N days (backfilled as data becomes available)

This enables comparing model predictions vs real outcomes for all signals,
not just the ones that were traded — critical for validating before real money.

Usage from daily_pipeline.py:
    tracker = SignalTracker("data/paper_trading/signal_history.parquet")
    tracker.record_signals(signals_df, traded_tickers, today)
    tracker.backfill_outcomes(ohlcv, horizon_days=20)
    tracker.save()
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.utils import get_logger

log = get_logger(__name__)

SIGNAL_HISTORY_PATH = "data/paper_trading/signal_history.parquet"


class SignalTracker:
    """Tracks all generated signals and backfills actual returns."""

    def __init__(self, path: str | Path = SIGNAL_HISTORY_PATH) -> None:
        self.path = Path(path)
        self.history = self._load()

    def _load(self) -> pd.DataFrame:
        if self.path.exists():
            df = pd.read_parquet(self.path)
            log.info("signal_history_loaded", rows=len(df))
            return df
        return pd.DataFrame({
            "signal_date": pd.Series(dtype="str"),
            "ticker": pd.Series(dtype="str"),
            "v2_score": pd.Series(dtype="float64"),
            "rank": pd.Series(dtype="int64"),
            "was_traded": pd.Series(dtype="bool"),
            "skip_reason": pd.Series(dtype="str"),
            "entry_date": pd.Series(dtype="str"),
            "entry_price": pd.Series(dtype="float64"),
            "exit_date": pd.Series(dtype="str"),
            "exit_price": pd.Series(dtype="float64"),
            "exit_reason": pd.Series(dtype="str"),
            "pnl_pct": pd.Series(dtype="float64"),
            "actual_close_at_signal": pd.Series(dtype="float64"),
            "actual_close_20d": pd.Series(dtype="float64"),
            "actual_ret_20d": pd.Series(dtype="float64"),
            "outcome_filled": pd.Series(dtype="bool"),
        })

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.history.to_parquet(self.path, index=False)
        log.info("signal_history_saved", rows=len(self.history))

    def record_signals(
        self,
        signals: pd.DataFrame,
        traded_tickers: set[str],
        skip_reasons: dict[str, str],
        today: str,
    ) -> None:
        """Record today's BUY signals with trade/skip status.

        Parameters
        ----------
        signals : DataFrame — all signals (must have ticker, ensemble_score, recommendation)
        traded_tickers : set of tickers that were actually queued/traded today
        skip_reasons : dict {ticker: reason} for signals not traded
        today : ISO date string
        """
        buys = signals[signals["recommendation"] == "BUY"].copy()
        if buys.empty:
            return

        # Avoid duplicate recording for the same date
        if not self.history.empty:
            already = self.history[self.history["signal_date"] == today]
            if not already.empty:
                log.info("signals_already_recorded", date=today)
                return

        records = []
        for rank, (_, row) in enumerate(buys.iterrows()):
            ticker = row["ticker"]
            was_traded = ticker in traded_tickers
            skip_reason = skip_reasons.get(ticker, "") if not was_traded else ""

            records.append({
                "signal_date": today,
                "ticker": ticker,
                "v2_score": float(row.get("ensemble_score", 0)),
                "rank": rank,
                "was_traded": was_traded,
                "skip_reason": skip_reason,
                "entry_date": None,
                "entry_price": None,
                "exit_date": None,
                "exit_price": None,
                "exit_reason": None,
                "pnl_pct": None,
                "actual_close_at_signal": None,
                "actual_close_20d": None,
                "actual_ret_20d": None,
                "outcome_filled": False,
            })

        new_df = pd.DataFrame(records)
        self.history = pd.concat([self.history, new_df], ignore_index=True)
        log.info("signals_recorded", date=today, count=len(records),
                 traded=len(traded_tickers))

    def update_trade_outcomes(self, closed_trades: list[dict]) -> None:
        """Fill in entry/exit data from actual paper trades.

        Call after paper trading closes positions. Matches by ticker + entry_date.
        """
        if not closed_trades or self.history.empty:
            return

        for trade in closed_trades:
            ticker = trade["ticker"]
            entry_date = trade["entry_date"]

            # Match signal: signal_date should be 1 day before entry_date (next-day open)
            mask = (
                (self.history["ticker"] == ticker)
                & (self.history["was_traded"])
                & (self.history["entry_date"].isna())
            )
            candidates = self.history[mask]
            if candidates.empty:
                continue

            # Take the most recent unmatched signal for this ticker
            idx = candidates.index[-1]
            self.history.loc[idx, "entry_date"] = entry_date
            self.history.loc[idx, "entry_price"] = trade["entry_price"]
            self.history.loc[idx, "exit_date"] = trade["exit_date"]
            self.history.loc[idx, "exit_price"] = trade["exit_price"]
            self.history.loc[idx, "exit_reason"] = trade["exit_reason"]
            self.history.loc[idx, "pnl_pct"] = trade["pnl_pct"]

    def backfill_outcomes(self, ohlcv: pd.DataFrame, horizon_days: int = 20) -> int:
        """Backfill actual_ret_20d for signals old enough to have outcome data.

        Looks up the close price on signal_date and signal_date + horizon trading days.
        Returns number of signals updated.
        """
        if self.history.empty:
            return 0

        ohlcv = ohlcv.copy()
        ohlcv["date"] = pd.to_datetime(ohlcv["date"])

        # Only process signals not yet filled
        unfilled = self.history[self.history["outcome_filled"] == False].copy()  # noqa: E712
        if unfilled.empty:
            return 0

        updated = 0
        for idx, row in unfilled.iterrows():
            ticker = row["ticker"]
            signal_date = pd.Timestamp(row["signal_date"])

            ticker_data = ohlcv[ohlcv["ticker"] == ticker].sort_values("date")
            if ticker_data.empty:
                continue

            # Find close on signal date
            on_signal = ticker_data[ticker_data["date"] == signal_date]
            if on_signal.empty:
                # Try closest date before
                before = ticker_data[ticker_data["date"] <= signal_date]
                if before.empty:
                    continue
                on_signal = before.iloc[[-1]]

            close_at_signal = float(on_signal.iloc[0]["close"])

            # Find close N trading days after signal
            future = ticker_data[ticker_data["date"] > signal_date]
            if len(future) < horizon_days:
                continue  # Not enough data yet

            close_20d = float(future.iloc[horizon_days - 1]["close"])
            actual_ret = (close_20d / close_at_signal) - 1

            self.history.loc[idx, "actual_close_at_signal"] = close_at_signal
            self.history.loc[idx, "actual_close_20d"] = close_20d
            self.history.loc[idx, "actual_ret_20d"] = round(actual_ret, 6)
            self.history.loc[idx, "outcome_filled"] = True
            updated += 1

        if updated:
            log.info("outcomes_backfilled", count=updated)
        return updated

    def summary_stats(self) -> dict:
        """Compute summary statistics for model validation."""
        if self.history.empty:
            return {}

        filled = self.history[self.history["outcome_filled"]]
        if filled.empty:
            return {"total_signals": len(self.history), "outcomes_available": 0}

        traded = filled[filled["was_traded"]]
        not_traded = filled[~filled["was_traded"]]

        stats = {
            "total_signals": len(self.history),
            "outcomes_available": len(filled),
            "traded_count": len(traded),
            "not_traded_count": len(not_traded),
        }

        if not traded.empty:
            stats["traded_avg_ret_20d"] = round(float(traded["actual_ret_20d"].mean()), 4)
            stats["traded_hit_rate"] = round(
                float((traded["actual_ret_20d"] > 0).mean()), 4
            )

        if not not_traded.empty:
            stats["missed_avg_ret_20d"] = round(float(not_traded["actual_ret_20d"].mean()), 4)
            stats["missed_hit_rate"] = round(
                float((not_traded["actual_ret_20d"] > 0).mean()), 4
            )

        # Score-return correlation
        if len(filled) >= 10:
            corr = filled[["v2_score", "actual_ret_20d"]].corr().iloc[0, 1]
            stats["score_return_correlation"] = round(float(corr), 4)

        return stats
