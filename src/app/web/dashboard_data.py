"""Dashboard data layer — builds the render-ready view from Supabase + OHLCV.

Pure data (no FastAPI) so both the web server and the daily pipeline can import
it. The pipeline computes the view and writes it to Supabase (`dashboard_view`);
the logged-in client reads that view and paints the dashboard. Keeping this out
of `server.py` avoids dragging FastAPI into the pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.data import supabase_store

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "processed"
PAPER_TRADING_DIR = ROOT / "data" / "paper_trading"
PAPER_TRADING_ADAPTIVE_DIR = ROOT / "data" / "paper_trading" / "adaptive"


def _load_ohlcv() -> pd.DataFrame:
    ohlcv = pd.read_parquet(DATA_DIR / "ohlcv_smallcap.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    return ohlcv


_spy_cache: pd.DataFrame | None = None


def _load_spy() -> pd.DataFrame | None:
    """SPY daily closes (benchmark for the equity chart / alpha). Cached."""
    global _spy_cache
    if _spy_cache is not None:
        return _spy_cache
    path = DATA_DIR / "smallcap_spy.parquet"
    if not path.exists():
        return None
    spy = pd.read_parquet(path)
    spy["date"] = pd.to_datetime(spy["date"])
    _spy_cache = spy.sort_values("date")[["date", "close"]].reset_index(drop=True)
    return _spy_cache


def _spy_aligned(chart_dates: list[str], initial_capital: float) -> list[float]:
    """SPY equity normalised to `initial_capital` at the first chart date, sampled
    on-or-before each chart date (so a buy-and-hold SPY of the same € can overlay
    the portfolio line). [] if SPY data is unavailable."""
    spy = _load_spy()
    if spy is None or not chart_dates:
        return []
    target = pd.DataFrame({"date": pd.to_datetime(chart_dates)})
    merged = pd.merge_asof(target, spy, on="date", direction="backward")
    closes = merged["close"].ffill().bfill()
    if closes.isna().all() or float(closes.iloc[0]) == 0:
        return []
    base = float(closes.iloc[0])
    return [round(float(initial_capital * c / base), 2) for c in closes]


def _compute_stats(values: list[float], spy_values: list[float]) -> dict | None:
    """Sharpe (annualised), max drawdown and alpha vs SPY from the NAV series.
    None when there is too little history (<10 NAV points) for the figures to mean
    anything — the paper-trading was reset 2026-06-11, so early days are noisy."""
    if not values or len(values) < 10:
        return None
    arr = np.asarray(values, dtype=float)
    rets = arr[1:] / arr[:-1] - 1
    running_max = np.maximum.accumulate(arr)
    max_dd = float((arr / running_max - 1).min()) * 100
    std = float(rets.std())
    sharpe = float(rets.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    total_ret = (arr[-1] / arr[0] - 1) * 100
    alpha = None
    if spy_values and len(spy_values) == len(values) and spy_values[0]:
        spy_ret = (spy_values[-1] / spy_values[0] - 1) * 100
        alpha = round(total_ret - spy_ret, 1)
    return {"sharpe": round(sharpe, 2), "max_dd": round(max_dd, 1), "alpha": alpha}


def load_paper_trading(ohlcv: pd.DataFrame,
                       pt_dir: Path | None = None,
                       adaptive_stop: bool = False) -> dict | None:
    pt_dir = pt_dir or PAPER_TRADING_DIR
    strategy = "adaptive" if pt_dir == PAPER_TRADING_ADAPTIVE_DIR else "baseline"

    # Source of truth is Supabase; fall back to the local JSON (offline/dev).
    state = supabase_store.read_state(strategy)
    if state is None:
        portfolio_path = pt_dir / "portfolio.json"
        if not portfolio_path.exists():
            return None
        with open(portfolio_path) as f:
            state = json.load(f)

    positions = []
    for pos in state.get("positions", []):
        ticker = pos["ticker"]
        ticker_data = ohlcv[ohlcv["ticker"] == ticker].sort_values("date")
        current_price = (float(ticker_data.iloc[-1]["close"])
                         if not ticker_data.empty else pos["entry_price"])
        pnl_pct = (current_price / pos["entry_price"] - 1) * 100
        invested = pos["shares"] * pos["entry_price"]
        current_value = pos["shares"] * current_price
        profit = current_value - invested
        # Compute effective trail pct (adaptive tightens to 6% after day 5 if profitable)
        effective_trail_pct = pos["trailing_stop_pct"]
        days_held = state.get("current_day_idx", 0) - pos.get("entry_day_idx", 0)
        if (adaptive_stop and effective_trail_pct > 0 and days_held > 5
                and current_price > pos["entry_price"]):
            effective_trail_pct = min(effective_trail_pct, 0.06)
        trail_trigger = pos["high_price"] * (1 - effective_trail_pct)
        positions.append({
            "ticker": ticker,
            "entry_date": pos["entry_date"],
            "entry_price": round(pos["entry_price"], 4),
            "current_price": round(current_price, 2),
            "shares": pos["shares"],
            "invested": round(invested, 2),
            "current_value": round(current_value, 2),
            "profit": round(profit, 2),
            "pnl_pct": round(pnl_pct, 1),
            "trailing_stop_pct": round(effective_trail_pct * 100, 0),
            "trail_trigger": round(trail_trigger, 2),
            "high_price": round(pos["high_price"], 2),
        })

    closed_trades = []
    for t in state.get("closed_trades", []):
        closed_trades.append({
            "ticker": t["ticker"],
            "entry_date": t["entry_date"],
            "exit_date": t["exit_date"],
            "entry_price": round(t["entry_price"], 4),
            "exit_price": round(t["exit_price"], 2),
            "shares": t["shares"],
            "pnl_pct": round(t["pnl_pct"] * 100, 1),
            "pnl_usd": round(t["pnl_usd"], 2),
            "exit_reason": t["exit_reason"],
            "days_held": t["days_held"],
        })

    pending = state.get("pending_signals", [])

    total_value = state["cash"]
    for pos in positions:
        total_value += pos["current_value"]
    total_return = (total_value / state["initial_capital"] - 1) * 100

    n_closed = len(closed_trades)
    n_wins = sum(1 for t in closed_trades if t["pnl_pct"] > 0)
    win_rate = round(n_wins / n_closed * 100, 0) if n_closed > 0 else 0
    # float() casts: np.float64 isn't JSON-serializable for the Supabase view.
    avg_win = (float(round(np.mean([t["pnl_pct"] for t in closed_trades if t["pnl_pct"] > 0]), 1))
               if n_wins > 0 else 0)
    avg_loss = (float(round(np.mean([t["pnl_pct"] for t in closed_trades if t["pnl_pct"] <= 0]), 1))
                if (n_closed - n_wins) > 0 else 0)
    total_profit = round(sum(t["pnl_usd"] for t in closed_trades), 2)

    # Portfolio value history: Supabase nav_history, falling back to daily_log.
    chart_dates = []
    chart_values = []
    nav = supabase_store.read_nav(strategy)
    if nav:
        for e in nav:
            chart_dates.append(str(e["date"])[:10])
            chart_values.append(round(float(e["portfolio_value"]), 2))
    else:
        daily_log_path = pt_dir / "daily_log.jsonl"
        if daily_log_path.exists():
            with open(daily_log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        chart_dates.append(entry["date"])
                        chart_values.append(round(entry["portfolio_value"], 2))
                    except (json.JSONDecodeError, KeyError):
                        continue

    spy_values = _spy_aligned(chart_dates, state["initial_capital"])
    stats = _compute_stats(chart_values, spy_values)

    return {
        "positions": positions,
        "closed_trades": closed_trades,
        "pending": pending,
        "cash": round(state["cash"], 2),
        "total_value": round(total_value, 2),
        "initial_capital": state["initial_capital"],
        "total_return": round(total_return, 2),
        "n_open": len(positions),
        "max_positions": state.get("max_positions", 8),
        "n_closed": n_closed,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "total_profit": total_profit,
        "last_update": state.get("last_update", ""),
        "chart_dates": chart_dates,
        "chart_values": chart_values,
        "spy_values": spy_values,
        "stats": stats,
    }


def load_signal_history(pt_dir: Path | None = None) -> list[dict]:
    pt_dir = pt_dir or PAPER_TRADING_DIR
    strategy = "adaptive" if pt_dir == PAPER_TRADING_ADAPTIVE_DIR else "baseline"

    # Source of truth is Supabase; fall back to the local parquet (offline/dev).
    rows = supabase_store.read_signals(strategy, limit=50)
    if rows:
        return [{
            "ticker": r.get("ticker", ""),
            "date": str(r.get("signal_date", ""))[:10],
            "score": round(float(r.get("score") or 0), 4),
            "was_traded": bool(r.get("was_traded", False)),
            "skip_reason": r.get("skip_reason") or "",
            "actual_ret": (round(float(r["actual_ret_20d"]) * 100, 1)
                           if r.get("actual_ret_20d") is not None else None),
        } for r in rows]

    path = pt_dir / "signal_history.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    # Column names from SignalTracker: signal_date, v2_score, recommendation (may not exist)
    date_col = "signal_date" if "signal_date" in df.columns else "date"
    score_col = "v2_score" if "v2_score" in df.columns else "ensemble_score"
    df = df.sort_values(date_col, ascending=False)
    result = []
    for _, r in df.head(50).iterrows():
        result.append({
            "ticker": r.get("ticker", ""),
            "date": str(r.get(date_col, ""))[:10],
            "score": round(float(r.get(score_col, 0)), 4),
            "was_traded": bool(r.get("was_traded", False)),
            "skip_reason": r.get("skip_reason", "") or "",
            "actual_ret": (round(float(r["actual_ret_20d"]) * 100, 1)
                           if pd.notna(r.get("actual_ret_20d")) else None),
        })
    return result


def _get_data_freshness(ohlcv: pd.DataFrame) -> dict:
    return {
        "latest_date": ohlcv["date"].max().date().isoformat(),
        "earliest_date": ohlcv["date"].min().date().isoformat(),
        "n_tickers": int(ohlcv["ticker"].nunique()),
        "n_rows": len(ohlcv),
    }


def build_view(ohlcv: pd.DataFrame, pt_dir: Path, adaptive_stop: bool) -> dict | None:
    """Render-ready view for one strategy: paper + signals + data freshness.

    This is exactly what the client needs to paint the dashboard, so it can be
    stored in Supabase (`dashboard_view`) and fetched in one read after login.
    """
    paper = load_paper_trading(ohlcv, pt_dir, adaptive_stop=adaptive_stop)
    if paper is None:
        return None
    signals = load_signal_history(pt_dir)
    return {
        "paper": paper,
        "signals": signals,
        "data_info": _get_data_freshness(ohlcv),
    }
