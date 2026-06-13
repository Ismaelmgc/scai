"""SCAI Web Dashboard — Paper Trading UI with pipeline trigger."""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from app.data import supabase_store  # noqa: E402
from app.data.free_sources import finnhub  # noqa: E402

app = FastAPI(title="SCAI Dashboard")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

PAPER_TRADING_DIR = ROOT / "data" / "paper_trading"
PAPER_TRADING_ADAPTIVE_DIR = ROOT / "data" / "paper_trading" / "adaptive"
DATA_DIR = ROOT / "data" / "processed"
_STATIC = Path(__file__).parent / "static"
LOGO_PATH = _STATIC / "scai_bull.png"        # header mark: thin gold line-art bull
FAVICON_PATH = _STATIC / "scai_favicon.png"  # browser tab: gold tile (legible at 16px)

_pipeline_proc: subprocess.Popen | None = None


def _asset_data_uri(path: Path) -> str:
    """Base64 PNG data URI for a brand asset, embedded inline so the static
    Pages snapshot stays a single self-contained file. "" if the asset is missing.
    """
    if not path.exists():
        return ""
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}"


def _load_ohlcv() -> pd.DataFrame:
    ohlcv = pd.read_parquet(DATA_DIR / "ohlcv_smallcap.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    return ohlcv


def _load_paper_trading(ohlcv: pd.DataFrame,
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
    avg_win = (round(np.mean([t["pnl_pct"] for t in closed_trades if t["pnl_pct"] > 0]), 1)
               if n_wins > 0 else 0)
    avg_loss = (round(np.mean([t["pnl_pct"] for t in closed_trades if t["pnl_pct"] <= 0]), 1)
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
    }


def _load_signal_history(pt_dir: Path | None = None) -> list[dict]:
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


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    ohlcv = _load_ohlcv()
    paper = _load_paper_trading(ohlcv, PAPER_TRADING_DIR)
    paper_adaptive = _load_paper_trading(ohlcv, PAPER_TRADING_ADAPTIVE_DIR, adaptive_stop=True)
    signals = _load_signal_history(PAPER_TRADING_DIR)
    signals_adaptive = _load_signal_history(PAPER_TRADING_ADAPTIVE_DIR)
    data_info = _get_data_freshness(ohlcv)

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "paper": paper,
            "paper_adaptive": paper_adaptive,
            "signals": signals,
            "signals_adaptive": signals_adaptive,
            "data_info": data_info,
            "now": datetime.now().strftime("%Y-%m-%d %H:%M"),
            # Live prices stream client-side via the Finnhub WebSocket (works on
            # the static Pages snapshot too, since WebSockets bypass CORS).
            "finnhub_token": finnhub.public_token(),
            "logo": _asset_data_uri(LOGO_PATH),
            "favicon": _asset_data_uri(FAVICON_PATH),
        },
    )


@app.post("/api/run-pipeline", response_class=JSONResponse)
async def run_pipeline():
    global _pipeline_proc

    if _pipeline_proc is not None and _pipeline_proc.poll() is None:
        return JSONResponse(
            {"status": "running", "message": "Pipeline ya en ejecución"},
            status_code=409,
        )

    script = str(ROOT / "scripts" / "daily_pipeline.py")
    python = str(ROOT / ".venv" / "bin" / "python")
    log_path = PAPER_TRADING_DIR / "logs"
    log_path.mkdir(parents=True, exist_ok=True)

    env = {
        "PYTHONPATH": str(ROOT / "src"),
        "DYLD_LIBRARY_PATH": str(ROOT / ".local" / "lib"),
        "PATH": f"{ROOT / '.venv' / 'bin'}:/usr/local/bin:/usr/bin:/bin",
        "HOME": os.environ.get("HOME", ""),
    }
    # Load .env file for API keys
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                env[key.strip()] = val.strip()

    # Kept open intentionally: handed to Popen for the subprocess lifetime.
    stdout_log = open(log_path / "web_pipeline_stdout.log", "w")  # noqa: SIM115
    stderr_log = open(log_path / "web_pipeline_stderr.log", "w")  # noqa: SIM115

    _pipeline_proc = subprocess.Popen(
        [python, script],
        cwd=str(ROOT),
        env=env,
        stdout=stdout_log,
        stderr=stderr_log,
    )

    return {"status": "started", "pid": _pipeline_proc.pid}


@app.get("/api/pipeline-status", response_class=JSONResponse)
async def pipeline_status():
    global _pipeline_proc

    if _pipeline_proc is None:
        return {"status": "idle"}

    rc = _pipeline_proc.poll()
    if rc is None:
        # Read log tail for progress
        log_file = PAPER_TRADING_DIR / "logs" / "web_pipeline_stdout.log"
        tail = ""
        if log_file.exists():
            lines = log_file.read_text().strip().splitlines()
            tail = "\n".join(lines[-10:])
        return {"status": "running", "pid": _pipeline_proc.pid, "log_tail": tail}

    log_file = PAPER_TRADING_DIR / "logs" / "web_pipeline_stdout.log"
    tail = ""
    if log_file.exists():
        lines = log_file.read_text().strip().splitlines()
        tail = "\n".join(lines[-25:])

    _pipeline_proc = None
    return {"status": "finished", "exit_code": rc, "log_tail": tail}
