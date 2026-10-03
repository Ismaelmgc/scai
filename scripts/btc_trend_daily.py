"""BTC TREND <-> MANAGED FUTURES paper book — daily job, tracked on the web dashboard
like the other strategies (Supabase state + view + NAV).

Frozen rule (research 2026-09-28 / 2026-10-03, survivorship-free Binance data):
  signal     daily BTC close (00:00 UTC candle) vs its 100-day SMA, 3% band:
             IN when close > SMA*1.03, OUT when close < SMA*0.97 (hysteresis)
  in         100% of equity in BTC
  out        100% in managed futures (iMGP DBi Managed Futures UCITS ETF, Euronext
             Paris DBMF, USD class — the EU-retail-buyable twin of the US DBMF)
  execution  (what an IBKR EU-retail account can actually do)
             EXIT  : sell BTC at the signal candle's close (crypto is 24/7), then
                     buy the ETF at the NEXT Paris session OPEN
             ENTRY : sell the ETF at the next Paris session OPEN and buy BTC at that
                     same hour (Binance 1h candle open); from cash, buy BTC at once
  costs      BTC 12bps/side, ETF 10bps/side
Backtest 2019-06..2026-09 (DBMF US as proxy, entries 1 day late): ~59-63%/yr, Sharpe
~1.2, maxDD ~-21%, 2022 +5%. The window excludes 2018 (BTC rule -54% that year) and
is carried by 2020/2023/2024 -> LIVE PAPER IS THE ARBITER.

Data: Binance market-data mirror (api.binance.com is geo-blocked from US runners),
Coinbase fallback; the ETF from yfinance. Only completed candles / finite prices
are ever used: a missing price keeps the order PENDING (retried next run) instead
of writing a NaN into the book (a null in portfolio_state bricks the job).

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
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.data import supabase_store  # noqa: E402
from app.utils import notify_telegram  # noqa: E402

STRATEGY = "btc_trend"
TICKER = "BINANCE:BTCUSDT"      # Finnhub crypto symbol -> the live-prices worker quotes real BTC
CTA = "DBMF.PA"                 # iMGP DBi Managed Futures UCITS ETF, Euronext Paris, USD class
CAPITAL = 1000.0
SMA_N = 100
BAND = 0.03
COST = 0.0012
ETF_COST = 0.0010
MAX_STALE_DAYS = 3
PT_DIR = ROOT / "data/paper_trading/btc_trend"


# ---------------------------------------------------------------- data
def _binance(interval: str, limit: int) -> pd.DataFrame:
    r = httpx.get("https://data-api.binance.vision/api/v3/klines",
                  params={"symbol": "BTCUSDT", "interval": interval, "limit": limit}, timeout=30)
    r.raise_for_status()
    now_ms = datetime.now(UTC).timestamp() * 1000
    rows = [(k[0], float(k[1]), float(k[2]), float(k[3]), float(k[4]))
            for k in r.json() if k[6] < now_ms]                      # close_time passed
    return pd.DataFrame(rows, columns=["t", "open", "high", "low", "close"])


def _coinbase(granularity: int) -> pd.DataFrame:
    r = httpx.get("https://api.exchange.coinbase.com/products/BTC-USD/candles",
                  params={"granularity": granularity}, headers={"User-Agent": "scai"}, timeout=30)
    r.raise_for_status()
    now = datetime.now(UTC).timestamp()
    rows = [(c[0] * 1000, float(c[3]), float(c[2]), float(c[1]), float(c[4]))
            for c in r.json() if c[0] + granularity <= now]           # completed candles only
    return pd.DataFrame(rows, columns=["t", "open", "high", "low", "close"])


def _frame(d: pd.DataFrame) -> pd.DataFrame:
    d["date"] = pd.to_datetime(d["t"], unit="ms")
    return d.drop(columns="t").drop_duplicates("date").sort_values("date").set_index("date")


def fetch_daily() -> tuple[pd.DataFrame, str]:
    """Completed daily candles, oldest first, indexed by UTC day. Raises if both
    sources fail or the data is stale/non-finite."""
    errors = []
    for name, fn in (("binance", lambda: _binance("1d", 400)), ("coinbase", lambda: _coinbase(86400))):
        try:
            d = _frame(fn())
        except httpx.HTTPError as e:
            errors.append(f"{name}: {e}")
            continue
        if len(d) < SMA_N + 20 or not np.isfinite(d[["open", "high", "low", "close"]].to_numpy()).all():
            errors.append(f"{name}: {len(d)} candles / non-finite values")
            continue
        age = (pd.Timestamp(datetime.now(UTC).date()) - d.index[-1]).days
        if age > MAX_STALE_DAYS:
            errors.append(f"{name}: last candle {d.index[-1].date()} is {age}d old")
            continue
        return d, name
    raise RuntimeError("no usable BTC daily data — " + "; ".join(errors))


def fetch_btc_hourly() -> pd.Series:
    """BTC hourly OPEN prices (naive UTC index) — the entry fill at the ETF's open hour.
    Empty on failure (run_days then falls back to that day's daily-candle open)."""
    for fn in (lambda: _binance("1h", 1000), lambda: _coinbase(3600)):
        try:
            h = _frame(fn())["open"]
        except httpx.HTTPError:
            continue
        h = h[np.isfinite(h)]
        if len(h):
            return h
    return pd.Series(dtype=float)


def fetch_cta() -> pd.DataFrame:
    """ETF daily open/close by Paris session date; rows with non-finite prices dropped
    (yfinance can carry a NaN for the latest bar)."""
    h = yf.Ticker(CTA).history(period="120d", auto_adjust=False)
    if h is None or h.empty:
        return pd.DataFrame(columns=["open", "close"])
    out = pd.DataFrame({"open": h["Open"].to_numpy(), "close": h["Close"].to_numpy()},
                       index=pd.to_datetime(h.index.date))
    return out[np.isfinite(out).all(axis=1) & (out > 0).all(axis=1)]


def fetch_nasdaq() -> pd.DataFrame | None:
    """QQQ daily closes — the dashboard's second reference line (opportunity cost vs
    the Nasdaq-100). Optional: None on failure, the chart just omits the line."""
    try:
        h = yf.Ticker("QQQ").history(period="2y", auto_adjust=True)
    except Exception:
        return None
    if h is None or h.empty:
        return None
    out = pd.DataFrame({"date": pd.to_datetime(h.index.date), "close": h["Close"].to_numpy()})
    return out[np.isfinite(out["close"]) & (out["close"] > 0)]


def paris_open_utc(session: pd.Timestamp) -> pd.Timestamp:
    """09:00 Paris time of a session date, as a naive UTC hour (DST-aware)."""
    return (pd.Timestamp(session.date()).tz_localize("Europe/Paris") + pd.Timedelta(hours=9)) \
        .tz_convert("UTC").tz_localize(None)


def regime(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Target (1 = in BTC, 0 = out -> managed futures) at each close, with the band's hysteresis."""
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
def _held(st: dict, ticker: str) -> dict | None:
    return next((p for p in st["positions"] if p["ticker"] == ticker), None)


def _open(st: dict, ticker: str, fill: float, ds: str) -> None:
    st["positions"].append({"ticker": ticker, "entry_date": ds, "entry_price": round(fill, 4),
                            "shares": st["cash"] / fill, "high_price": fill,
                            "entry_day_idx": st["current_day_idx"], "trailing_stop_pct": 0.0})
    st["cash"] = 0.0


def _close(st: dict, p: dict, fill: float, ds: str, reason: str) -> None:
    st["positions"].remove(p)
    st["cash"] += p["shares"] * fill
    st["closed_trades"].append({
        "ticker": p["ticker"], "side": "LONG", "shares": round(p["shares"], 8),
        "entry_price": p["entry_price"], "entry_date": p["entry_date"],
        "exit_price": round(fill, 4), "exit_date": ds, "exit_reason": reason,
        "pnl_pct": round(fill / p["entry_price"] - 1, 4),
        "pnl_usd": round(p["shares"] * (fill - p["entry_price"]), 2),
        "days_held": st["current_day_idx"] - p["entry_day_idx"]})


def _pending(st: dict, action: str) -> dict | None:
    return next((o for o in st["pending_orders"] if o["action"] == action), None)


def execute_pending(st: dict, session: pd.Timestamp, cta_open: float, btc_open: float) -> list[str]:
    """Fill due orders at the Paris session `session` open (BTC at that same hour)."""
    done, ds = [], str(session.date())
    for o in list(st["pending_orders"]):
        if pd.Timestamp(o["after"]) > session:
            continue
        if o["action"] == "buy_cta":
            _open(st, CTA, cta_open * (1 + ETF_COST), ds)
            done.append(f"COMPRA CTA {ds}")
        elif o["action"] == "to_btc":
            cta = _held(st, CTA)
            if cta:
                _close(st, cta, cta_open * (1 - ETF_COST), ds, "rotacion_a_btc")
            _open(st, TICKER, btc_open * (1 + COST), ds)
            done.append(f"CTA→BTC {ds}")
        st["pending_orders"].remove(o)
    return done


def process_day(st: dict, d: pd.Timestamp, close: float, target: int, sma: float) -> dict:
    ds = str(d.date())
    after = str((d + pd.Timedelta(days=1)).date())   # the candle closes at 00:00 UTC of d+1
    st["current_day_idx"] += 1
    btc, action = _held(st, TICKER), None
    if btc and target == 0:
        _close(st, btc, close * (1 - COST), ds, "tendencia_rota")
        st["pending_orders"].append({"action": "buy_cta", "after": after})
        action = "SELL"
    elif not btc and target == 1:
        if _held(st, CTA):
            if not _pending(st, "to_btc"):
                st["pending_orders"].append({"action": "to_btc", "after": after})
                action = "SEÑAL"
        else:                                       # sitting in cash -> buy BTC right now
            st["pending_orders"] = [o for o in st["pending_orders"] if o["action"] != "buy_cta"]
            _open(st, TICKER, close * (1 + COST), ds)
            action = "BUY"
    elif not btc and target == 0:                   # signal flipped back before it was filled
        st["pending_orders"] = [o for o in st["pending_orders"] if o["action"] != "to_btc"]
    btc = _held(st, TICKER)
    if btc:
        btc["high_price"] = max(btc["high_price"], close)
        # trail_trigger (= high*(1-pct)) on the dashboard shows the exit level SMA*(1-BAND)
        btc["trailing_stop_pct"] = round(max(0.0, 1 - sma * (1 - BAND) / btc["high_price"]), 4)
    st["last_update"] = ds
    return {"date": ds, "action": action, "close": close, "sma": sma, "target": target}


def run_days(st: dict, data: pd.DataFrame, target: pd.Series, sma: pd.Series,
             cta: pd.DataFrame, btc_hourly: pd.Series) -> tuple[list[dict], list[str]]:
    """Process every unprocessed candle in order. A Paris session on day d opens
    (07-08 UTC) before candle d closes (00:00 UTC d+1), so its fills go first."""
    results, fills = [], []
    for d in [d for d in data.index if d > pd.Timestamp(st["last_update"])]:
        if st["pending_orders"] and d in cta.index:
            btc_open = float(btc_hourly.get(paris_open_utc(d), data.at[d, "open"]))
            fills += execute_pending(st, d, float(cta.at[d, "open"]), btc_open)
        results.append(process_day(st, d, float(data.at[d, "close"]), int(target[d]), float(sma[d])))
    return results, fills


def _signal_row(r: dict) -> dict:
    gap = r["close"] / r["sma"] - 1
    reason = {"SELL": "SALIDA: cierre < SMA100 −3% → futuros gestionados",
              "BUY": "ENTRADA: cierre > SMA100 +3%",
              "SEÑAL": "ENTRADA: vender CTA y comprar BTC en la apertura"}.get(
        r["action"], "en BTC: sobre SMA100" if r["target"] else "en futuros gestionados")
    return {"signal_date": r["date"], "ticker": TICKER, "score": round(gap, 4),
            "recommendation": "BUY" if r["target"] else "CTA",
            "was_traded": r["action"] is not None, "skip_reason": reason, "actual_ret_20d": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    data, source = fetch_daily()
    target, sma = regime(data["close"])
    cta, btc_hourly = fetch_cta(), fetch_btc_hourly()
    last_day = data.index[-1]
    print(f"BTC {source}: última vela {last_day.date()} cierre {data['close'].iloc[-1]:,.2f} "
          f"· SMA{SMA_N} {sma.iloc[-1]:,.2f} · objetivo {'BTC' if target.iloc[-1] else 'FUTUROS GESTIONADOS'} "
          f"· {CTA} {len(cta)} sesiones", flush=True)

    st = None if args.dry_run else supabase_store.read_state(STRATEGY)
    if st is None:
        # Fresh book: start on the latest completed candle (no backfill of history).
        st = {"initial_capital": CAPITAL, "cash": CAPITAL, "positions": [], "closed_trades": [],
              "current_day_idx": 0, "last_update": str(data.index[-2].date()),
              "pending_signals": [], "max_positions": 1}
    st.setdefault("pending_orders", [])
    if _held(st, CTA) and cta.empty:
        raise RuntimeError(f"holding {CTA} but no usable {CTA} prices — refusing to mark the book")

    results, fills = run_days(st, data, target, sma, cta, btc_hourly)
    already = not results          # idempotent: same candle again -> nothing new, still republish
    actions = [f"{r['action']} {r['date']}" for r in results if r["action"]] + fills
    print(f"  procesadas {len(results)} velas · acciones: {actions or '—'} · pendientes: {st['pending_orders'] or '—'}")

    if args.dry_run:
        print(f"  DRY RUN — cash €{st['cash']:.2f}, posiciones {[p['ticker'] for p in st['positions']]}")
        return

    supabase_store.write_state(STRATEGY, st)
    if results:
        supabase_store.upsert_signals(STRATEGY, [_signal_row(r) for r in results])

    ohlcv = pd.concat([data.reset_index()[["date", "close"]].assign(ticker=TICKER),
                       cta.rename_axis("date").reset_index()[["date", "close"]].assign(ticker=CTA)],
                      ignore_index=True)
    bench = data.reset_index()[["date", "close"]]
    PT_DIR.mkdir(parents=True, exist_ok=True)
    from app.web import dashboard_data
    qqq = fetch_nasdaq()
    view = dashboard_data.build_view(ohlcv, PT_DIR, adaptive_stop=False, strategy=STRATEGY, bench=bench,
                                     bench2=("Nasdaq-100", qqq) if qqq is not None else None)
    if view is None:
        return
    supabase_store.write_dashboard_view(STRATEGY, view)
    supabase_store.upsert_nav(STRATEGY, str(last_day.date()), float(view["paper"]["total_value"]))
    supabase_store.upsert_live_prices({p["ticker"]: p["current_price"] for p in view["paper"]["positions"]})
    if not already:
        paper, c, m = view["paper"], data["close"].iloc[-1], sma.iloc[-1]
        held = {p["ticker"] for p in st["positions"]}
        estado = ("🟢 EN BTC" if TICKER in held else "🔵 EN FUTUROS GESTIONADOS" if CTA in held
                  else "⚪ LIQUIDEZ (orden pendiente)")
        pend = ", ".join(f"{o['action']} desde {o['after']}" for o in st["pending_orders"])
        notify_telegram(
            f"✅ <b>SCAI BTC ↔ CTA</b> — vela {last_day.date()}\n"
            f"{estado} · BTC ${c:,.0f} · SMA{SMA_N} ${m:,.0f} ({c / m - 1:+.1%})\n"
            f"Entra > ${m * (1 + BAND):,.0f} · Sale < ${m * (1 - BAND):,.0f}\n"
            f"{'⚡ ' + ', '.join(actions) if actions else 'Sin cambios'}"
            f"{chr(10) + '⏳ ' + pend if pend else ''}\n"
            f"<b>Cartera</b>  €{paper['total_value']:,.2f} ({paper['total_return']:+.2f}%)")
    print(f"  Runtime {(time.time() - t0):.0f}s")


if __name__ == "__main__":
    main()
