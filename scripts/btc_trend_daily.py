"""BTC TREND paper book — daily job, tracked on the web dashboard like the other
strategies (Supabase state + view + NAV). Replaces the retired small-cap baseline.

Frozen rule (research 2026-09-28, survivorship-free Binance data 2018-2026):
  signal     daily close (00:00 UTC candle) vs its 100-day SMA, with a 3% band:
             ENTER when close > SMA*1.03, EXIT to cash when close < SMA*0.97
             (hysteresis -> ~2.4 round trips/yr instead of ~6 without the band)
  sizing     100% of equity in BTC when in, 0% (cash) when out. No leverage, no short.
  fills      that candle's official CLOSE +/- 12bps (10bps taker + 2bps slippage)
Backtest 2018-26: CAGR ~42% vs ~23% buy&hold, maxDD ~-65% vs -81%, Sharpe ~1.1;
the edge is crash avoidance (2018, 2022) and it lags B&H in strong bull years.
~2021+ Sharpe ~0.8 (marginal) -> LIVE PAPER IS THE ARBITER.

Data: Binance public market-data mirror (data-api.binance.vision — the main
api.binance.com is geo-blocked from US GitHub runners), Coinbase BTC-USD as
fallback. Only COMPLETED daily candles are used. Fills are the official candle
close, so a late run gives the same result as an on-time one (deterministic).

    PYTHONPATH=src python scripts/btc_trend_daily.py            # update + publish
    PYTHONPATH=src python scripts/btc_trend_daily.py --dry-run  # compute only, no writes
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.data import supabase_store  # noqa: E402
from app.utils import notify_telegram  # noqa: E402

STRATEGY = "btc_trend"
TICKER = "BINANCE:BTCUSDT"      # Finnhub crypto symbol -> the live-prices worker quotes real BTC
CAPITAL = 1000.0
SMA_N = 100
BAND = 0.03
COST = 0.0012
MAX_STALE_DAYS = 3
PT_DIR = ROOT / "data/paper_trading/btc_trend"


# ---------------------------------------------------------------- data
def _binance() -> pd.DataFrame:
    r = httpx.get("https://data-api.binance.vision/api/v3/klines",
                  params={"symbol": "BTCUSDT", "interval": "1d", "limit": 400}, timeout=30)
    r.raise_for_status()
    now_ms = datetime.now(UTC).timestamp() * 1000
    rows = [(k[0], float(k[1]), float(k[2]), float(k[3]), float(k[4]))
            for k in r.json() if k[6] < now_ms]                      # close_time passed
    return pd.DataFrame(rows, columns=["t", "open", "high", "low", "close"])


def _coinbase() -> pd.DataFrame:
    r = httpx.get("https://api.exchange.coinbase.com/products/BTC-USD/candles",
                  params={"granularity": 86400}, headers={"User-Agent": "scai"}, timeout=30)
    r.raise_for_status()
    today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    rows = [(c[0] * 1000, float(c[3]), float(c[2]), float(c[1]), float(c[4]))
            for c in r.json() if c[0] < today]                        # drop today's open candle
    return pd.DataFrame(rows, columns=["t", "open", "high", "low", "close"])


def fetch_daily() -> tuple[pd.DataFrame, str]:
    """Completed daily candles, oldest first, indexed by UTC day. Raises if both
    sources fail or the data is stale/non-finite — a loud failure beats writing a
    NaN into the book (a null in portfolio_state bricks the job)."""
    errors = []
    for name, fn in (("binance", _binance), ("coinbase", _coinbase)):
        try:
            d = fn()
        except httpx.HTTPError as e:
            errors.append(f"{name}: {e}")
            continue
        d["date"] = pd.to_datetime(d["t"], unit="ms").dt.normalize()
        d = d.drop(columns="t").drop_duplicates("date").sort_values("date").set_index("date")
        if len(d) < SMA_N + 20 or not np.isfinite(d[["open", "high", "low", "close"]].to_numpy()).all():
            errors.append(f"{name}: {len(d)} candles / non-finite values")
            continue
        age = (pd.Timestamp(datetime.now(UTC).date()) - d.index[-1]).days
        if age > MAX_STALE_DAYS:
            errors.append(f"{name}: last candle {d.index[-1].date()} is {age}d old")
            continue
        return d, name
    raise RuntimeError("no usable BTC daily data — " + "; ".join(errors))


def regime(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Target position (1 = in BTC, 0 = cash) at each close, with the band's hysteresis."""
    sma = close.rolling(SMA_N).mean()
    pos, p = [], 0
    for px, m in zip(close, sma):
        if not np.isnan(m):
            if p == 0 and px > m * (1 + BAND):
                p = 1
            elif p == 1 and px < m * (1 - BAND):
                p = 0
        pos.append(p)
    return pd.Series(pos, close.index), sma


# ---------------------------------------------------------------- engine
def _mirror(pos: dict, exit_level: float) -> None:
    """Keep the build_view field in sync: trail_trigger (= high*(1-pct)) shows the
    exit level SMA*(1-BAND), i.e. where the rule would move the book to cash."""
    pos["trailing_stop_pct"] = round(max(0.0, 1 - exit_level / pos["high_price"]), 4)


def process_day(st: dict, d: pd.Timestamp, close: float, target: int, sma: float) -> dict:
    ds = str(d.date())
    st["current_day_idx"] += 1
    idx = st["current_day_idx"]
    action = None
    if st["positions"] and target == 0:
        p = st["positions"].pop()
        px = close * (1 - COST)
        st["cash"] += p["shares"] * px
        st["closed_trades"].append({
            "ticker": TICKER, "side": "LONG", "shares": round(p["shares"], 8),
            "entry_price": p["entry_price"], "entry_date": p["entry_date"],
            "exit_price": round(px, 2), "exit_date": ds, "exit_reason": "tendencia_rota",
            "pnl_pct": round(px / p["entry_price"] - 1, 4),
            "pnl_usd": round(p["shares"] * (px - p["entry_price"]), 2),
            "days_held": idx - p["entry_day_idx"]})
        action = "SELL"
    elif not st["positions"] and target == 1:
        px = close * (1 + COST)
        st["positions"].append({"ticker": TICKER, "entry_date": ds, "entry_price": round(px, 2),
                                "shares": st["cash"] / px, "high_price": close, "entry_day_idx": idx})
        st["cash"] = 0.0
        action = "BUY"
    for p in st["positions"]:
        p["high_price"] = max(p["high_price"], close)
        _mirror(p, sma * (1 - BAND))
    st["last_update"] = ds
    return {"date": ds, "action": action, "close": close, "sma": sma, "target": target}


def _signal_row(r: dict) -> dict:
    gap = r["close"] / r["sma"] - 1
    if r["target"]:
        reason = "comprado: sobre SMA100" if r["action"] != "BUY" else "ENTRADA: cierre > SMA100 +3%"
    else:
        reason = "liquidez: bajo tendencia" if r["action"] != "SELL" else "SALIDA: cierre < SMA100 −3%"
    return {"signal_date": r["date"], "ticker": TICKER, "score": round(gap, 4),
            "recommendation": "BUY" if r["target"] else "CASH",
            "was_traded": r["action"] is not None, "skip_reason": reason, "actual_ret_20d": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    data, source = fetch_daily()
    target, sma = regime(data["close"])
    last_day = data.index[-1]
    print(f"BTC {source}: {len(data)} velas, última {last_day.date()} cierre {data['close'].iloc[-1]:,.2f} "
          f"· SMA{SMA_N} {sma.iloc[-1]:,.2f} · objetivo {'COMPRADO' if target.iloc[-1] else 'LIQUIDEZ'}", flush=True)

    st = None if args.dry_run else supabase_store.read_state(STRATEGY)
    if st is None:
        # Fresh book: start on the latest completed candle (no backfill of history).
        st = {"initial_capital": CAPITAL, "cash": CAPITAL, "positions": [], "closed_trades": [],
              "current_day_idx": 0, "last_update": str(data.index[-2].date()),
              "pending_signals": [], "max_positions": 1}

    todo = [d for d in data.index if d > pd.Timestamp(st["last_update"])]
    results = [process_day(st, d, float(data.at[d, "close"]), int(target[d]), float(sma[d])) for d in todo]
    # Idempotent: a re-run on the same candle processes nothing but still republishes.
    already = not results
    actions = [f"{r['action']} {r['date']}" for r in results if r["action"]]
    print(f"  procesadas {len(results)} velas · acciones: {actions or '—'}")

    if args.dry_run:
        print(f"  DRY RUN — cash €{st['cash']:.2f}, posiciones {len(st['positions'])}")
        return

    supabase_store.write_state(STRATEGY, st)
    if results:
        supabase_store.upsert_signals(STRATEGY, [_signal_row(r) for r in results])

    ohlcv = data.reset_index().assign(ticker=TICKER, volume=0)
    bench = data.reset_index()[["date", "close"]]
    PT_DIR.mkdir(parents=True, exist_ok=True)
    from app.web import dashboard_data
    view = dashboard_data.build_view(ohlcv, PT_DIR, adaptive_stop=False, strategy=STRATEGY, bench=bench)
    if view is None:
        return
    supabase_store.write_dashboard_view(STRATEGY, view)
    supabase_store.upsert_nav(STRATEGY, str(last_day.date()), float(view["paper"]["total_value"]))
    supabase_store.upsert_live_prices({p["ticker"]: p["current_price"] for p in view["paper"]["positions"]})
    if not already:
        paper, c, m = view["paper"], data["close"].iloc[-1], sma.iloc[-1]
        estado = "🟢 COMPRADO" if st["positions"] else "⚪ LIQUIDEZ"
        notify_telegram(
            f"✅ <b>SCAI BTC tendencia</b> — vela {last_day.date()}\n"
            f"{estado} · BTC ${c:,.0f} · SMA{SMA_N} ${m:,.0f} ({c / m - 1:+.1%})\n"
            f"Entra > ${m * (1 + BAND):,.0f} · Sale < ${m * (1 - BAND):,.0f}\n"
            f"{'⚡ ' + ', '.join(actions) if actions else 'Sin cambios'}\n"
            f"<b>BTC</b>  €{paper['total_value']:,.2f} ({paper['total_return']:+.2f}%)")
    print(f"  Runtime {(time.time() - t0):.0f}s")


if __name__ == "__main__":
    main()
