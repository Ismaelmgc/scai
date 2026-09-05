"""BOUNCE paper book — daily EOD job, tracked on the web dashboard like the other
strategies (Supabase state + view + NAV), out-of-sample forward test.

Frozen config (backtest 2014-2026; spread-aware ~+2.1-2.4%/mo, alpha +0.9-1.2%/mo
15-30bps — higher than liquidcap/smallcap AND it survives cost, but in-sample-
mined vs their purged numbers, so LIVE PAPER IS THE ARBITER):
  universe   current S&P 500 members (data/liquidcap/membership_sp500.parquet)
  entry      RSI(14)<30  AND  A/D-normalized(100)<15  AND  ROC(20)<=-7%
  ranking    biggest 20d drop first when >free slots signal the same day
  portfolio  max 8 concurrent, equal-weight 1/8 (idle slots = cash), 5-bar cooldown
  exit       stop -12% until +12% arms a trailing = max(entry*1.10, peak*0.98);
             pre-arm time-stop at 60 bars.  Entry AND fills at that day's CLOSE.
  prices     raw OHLC (yfinance auto_adjust=False) to match the backtest / TV

Unlike the ML books there is no model/retrain and no data commit — signals are
computed from a fresh yfinance pull each run and the only persistent state lives
in Supabase. Self-contained engine; build_view just marks the state to market.

    PYTHONPATH=src python scripts/bounce_daily.py            # update + publish
    PYTHONPATH=src python scripts/bounce_daily.py --dry-run  # compute only, no writes
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.data import supabase_store  # noqa: E402
from app.utils import notify_telegram  # noqa: E402

STRATEGY = "bounce"
CAPITAL = 1000.0
N_SLOTS = 8
RSI_TH, AD_TH, ROC_FLOOR = 35.0, 20.0, -7.0   # robust-selected config (35/20/150)
AD_LOOKBACK = 150                              # A/D-norm min-max window
SL, ACT, GB, FLOOR, NMAX = 0.12, 0.12, 0.02, 0.10, 60
COOLDOWN_BARS = 5
FETCH_DAYS = 500                               # >= AD_LOOKBACK + warmup
INCEPTION = "2026-08-28"
MAX_CANDIDATES = 10                            # top-N ranked signals published/day

MEMBERSHIP_FP = ROOT / "data/liquidcap/membership_sp500.parquet"
PT_DIR = ROOT / "data/paper_trading/bounce"
SEED_FP = PT_DIR / "seed_state.json"           # fresh Aug-28 book under the new config


# ---------------------------------------------------------------- data / indicators
def current_universe() -> list[str]:
    m = pd.read_parquet(MEMBERSHIP_FP)
    m["end"] = pd.to_datetime(m["end"])
    return sorted(m[m["end"] >= m["end"].max()]["ticker"].unique().tolist())


def fetch(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Raw OHLCV per ticker + RSI(14)/ROC(20)/A/D-norm(100) on raw close."""
    out: dict[str, pd.DataFrame] = {}
    syms = [t.replace(".", "-") for t in tickers]
    for s in range(0, len(syms), 120):
        chunk, orig = syms[s:s + 120], tickers[s:s + 120]
        dl = yf.download(chunk, period=f"{FETCH_DAYS}d", auto_adjust=False,
                         progress=False, group_by="ticker", threads=True)
        for tk, sym in zip(orig, chunk):
            try:
                sub = dl[sym] if isinstance(dl.columns, pd.MultiIndex) else dl
                x = sub[["High", "Low", "Close", "Volume"]].dropna()
                if len(x) < 120:
                    continue
                d = pd.DataFrame({"date": pd.to_datetime(x.index), "high": x["High"].values,
                                  "low": x["Low"].values, "close": x["Close"].values,
                                  "vol": x["Volume"].values}).reset_index(drop=True)
                delta = d["close"].diff()
                up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
                dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
                d["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
                d["roc"] = d["close"].pct_change(20) * 100
                hl = (d["high"] - d["low"]).replace(0, np.nan)
                mfv = (((d["close"] - d["low"]) - (d["high"] - d["close"])) / hl * d["vol"]).fillna(0)
                ad = mfv.cumsum()
                lo = ad.rolling(AD_LOOKBACK, min_periods=AD_LOOKBACK).min(); hi = ad.rolling(AD_LOOKBACK, min_periods=AD_LOOKBACK).max()
                d["adnorm"] = np.where((hi - lo) != 0, (ad - lo) / (hi - lo) * 100, 50.0)
                out[tk] = d.set_index("date")
            except Exception:
                pass
        print(f"  fetched {min(s + 120, len(syms))}/{len(syms)}", flush=True)
    return out


# ---------------------------------------------------------------- state helpers
def _effective_trail(pos: dict) -> float:
    """Display-only: current stop distance below the high (build_view shows this)."""
    entry, peak, armed = pos["entry_price"], pos["peak"], pos["armed"]
    if armed:
        stop = max(entry * (1 + FLOOR), peak * (1 - GB))
    else:
        stop = entry * (1 - SL)
    return round(max(0.0, 1 - stop / peak), 4) if peak > 0 else 0.0


def _sync(pos: dict, day_idx: int) -> dict:
    """Keep the build_view mirror fields in sync with the engine fields."""
    pos["high_price"] = pos["peak"]
    pos["entry_day_idx"] = day_idx - pos["bars_in"]
    pos["trailing_stop_pct"] = _effective_trail(pos)
    return pos


def load_seed() -> dict | None:
    """Seed the fresh book from the committed inception state (the Aug-28 book
    re-simulated under the 35/20/150 config). Already in the superset schema
    build_view + the engine both read, so it's returned as-is."""
    if not SEED_FP.exists():
        return None
    st = json.loads(SEED_FP.read_text())
    print(f"  loaded seed book: {len(st.get('positions', []))} open, cash €{st.get('cash', CAPITAL)}, "
          f"last_update {st.get('last_update')}")
    return st


# ---------------------------------------------------------------- engine
def process_day(st: dict, data: dict[str, pd.DataFrame], d: pd.Timestamp) -> dict:
    ds = str(d.date())
    st["current_day_idx"] += 1
    entered, closed = [], []

    # 1) manage open positions on bar d
    survivors = []
    for p in st["positions"]:
        df = data.get(p["ticker"])
        if df is None or d not in df.index:
            survivors.append(p); continue
        row = df.loc[d]; hi, lo, cc = float(row.high), float(row.low), float(row.close)
        entry = p["entry_price"]; p["bars_in"] += 1
        hard, trig, fl = entry * (1 - SL), entry * (1 + ACT), entry * (1 + FLOOR)
        exit_px = reason = None
        if p["armed"]:
            stop = max(fl, p["peak"] * (1 - GB))
            if lo <= stop:
                exit_px, reason = stop, ("floor" if stop == fl else "trailing")
        else:
            if lo <= hard:
                exit_px, reason = hard, "stop_loss"
            elif p["bars_in"] >= NMAX:
                exit_px, reason = cc, "time_stop"
        if exit_px is None:
            p["peak"] = max(p["peak"], hi)
            if not p["armed"] and p["peak"] >= trig:
                p["armed"] = True
            survivors.append(_sync(p, st["current_day_idx"]))
        else:
            st["cash"] += p["shares"] * exit_px
            ret = exit_px / entry - 1
            st["closed_trades"].append({
                "ticker": p["ticker"], "side": "LONG", "shares": round(p["shares"], 6),
                "entry_price": entry, "entry_date": p["entry_date"], "exit_price": round(exit_px, 4),
                "exit_date": ds, "exit_reason": reason, "pnl_pct": round(ret, 4),
                "pnl_usd": round(p["shares"] * (exit_px - entry), 2), "days_held": p["bars_in"]})
            st["cooldown"][p["ticker"]] = ds
            closed.append(p["ticker"])
    st["positions"] = survivors

    # 2) equity mark-to-market on d
    held = {p["ticker"] for p in st["positions"]}
    mtm = sum(p["shares"] * (float(data[p["ticker"]].loc[d].close)
              if (p["ticker"] in data and d in data[p["ticker"]].index) else p["entry_price"])
              for p in st["positions"])
    equity = st["cash"] + mtm

    # 3) today's signals, biggest 20d drop first. Scan ALWAYS (even with the book
    # full) so the daily ranked candidate list is recorded for the signal history;
    # only actually fill the free slots (cands[:free] is empty when free<=0).
    free = N_SLOTS - len(st["positions"])
    cands = []
    for tk, df in data.items():
        if tk in held or d not in df.index:
            continue
        cd = st["cooldown"].get(tk)
        if cd is not None and np.busday_count(pd.Timestamp(cd).date(), d.date()) < COOLDOWN_BARS:
            continue
        r = df.loc[d]
        if pd.notna(r.rsi) and pd.notna(r.adnorm) and pd.notna(r.roc) \
           and r.rsi < RSI_TH and r.adnorm < AD_TH and r.roc <= ROC_FLOOR:
            cands.append((float(r.roc), tk, float(r.close)))
    cands.sort()
    for roc, tk, px in cands[:max(free, 0)]:
        alloc = min(equity / N_SLOTS, st["cash"])
        if alloc <= 0:
            break
        st["cash"] -= alloc
        pos = {"ticker": tk, "entry_date": ds, "entry_price": px, "shares": alloc / px,
               "peak": px, "armed": False, "bars_in": 0, "entry_roc": round(roc, 2)}
        st["positions"].append(_sync(pos, st["current_day_idx"]))
        entered.append(tk)

    st["last_update"] = ds
    st["_today_candidates"] = [(tk, roc) for roc, tk, _ in cands]
    return {"entered": entered, "closed": closed}


def build_ohlcv(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [df.reset_index()[["date", "high", "low", "close"]].assign(ticker=tk)
              for tk, df in data.items()]
    o = pd.concat(frames, ignore_index=True)
    o["open"] = o["close"]; o["volume"] = 0
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    uni = current_universe()
    print(f"universo S&P500 actual: {len(uni)} tickers · descargando precios raw...", flush=True)
    data = fetch(uni)
    dts = sorted({d for df in data.values() for d in df.index})
    if not dts:
        print("sin datos."); return
    today = str(dts[-1].date())

    st = None if args.dry_run else supabase_store.read_state(STRATEGY)
    if st is None:
        st = load_seed() or {
            "initial_capital": CAPITAL, "cash": CAPITAL, "positions": [], "closed_trades": [],
            "current_day_idx": 100, "last_update": str(dts[-2].date()) if len(dts) > 1 else INCEPTION,
            "pending_signals": [], "max_positions": N_SLOTS, "cooldown": {}, "peak_equity": CAPITAL}

    last = pd.Timestamp(st["last_update"])
    todo = [d for d in dts if d.date() > last.date()]

    # Idempotency: never process a session twice (last bar doesn't move until the
    # next US session; a double run would re-enter this session's signals). Still
    # (re)publish view + NAV so a re-run after a partial failure self-heals.
    already = not todo
    result = {"entered": [], "closed": []}
    if already:
        print(f"  sesión {today} ya procesada (last_update={st['last_update']}) — solo refresca view + NAV")
    else:
        for d in todo:
            result = process_day(st, data, d)
        print(f"  {today}: entered={result['entered'] or '[]'}  closed={result['closed'] or '[]'}")

    if args.dry_run:
        held = [f"{p['ticker']}({p['entry_roc']}%)" for p in st["positions"]]
        print(f"  DRY RUN — {len(st['positions'])} abiertas: {held}  cash €{st['cash']:.2f}")
        return

    ohlcv = build_ohlcv(data)
    supabase_store.write_state(STRATEGY, st)
    cands = st.pop("_today_candidates", [])
    if not already and cands:
        entered_set = set(result["entered"])
        supabase_store.upsert_signals(STRATEGY, [{
            "signal_date": today, "ticker": tk, "score": round(-roc, 4),
            "recommendation": "BUY", "was_traded": tk in entered_set,
            "skip_reason": "" if tk in entered_set else "slots llenos / ranking",
            "actual_ret_20d": None} for tk, roc in cands[:MAX_CANDIDATES]])

    PT_DIR.mkdir(parents=True, exist_ok=True)
    from app.web import dashboard_data
    view = dashboard_data.build_view(ohlcv, PT_DIR, adaptive_stop=False, strategy=STRATEGY)
    if view is not None:
        supabase_store.write_dashboard_view(STRATEGY, view)
        supabase_store.upsert_nav(STRATEGY, today, float(view["paper"]["total_value"]))
        # Overwrite live_prices with the official close so the web matches the
        # Telegram/snapshot from job-run until the next open (the live-prices
        # worker resumes with Finnhub quotes when the market reopens).
        supabase_store.upsert_live_prices(
            {p["ticker"]: p["current_price"]
             for p in view["paper"].get("positions", [])})
        paper = view["paper"]
        if not already:
            notify_telegram(
                f"✅ <b>SCAI Bounce (shadow OOS)</b> — {date.today():%Y-%m-%d}\n"
                f"📅 Última sesión: <b>{today}</b>\n"
                f"🟢 Abiertas: {', '.join(result['entered']) if result['entered'] else '—'}\n"
                f"🔴 Cerradas: {', '.join(result['closed']) if result['closed'] else '—'}\n"
                f"<b>Bounce</b>  €{paper['total_value']:,.2f} ({paper['total_return']:+.2f}%) · "
                f"{paper['n_open']} pos · WR {paper['win_rate']:.0f}%")
    print(f"  Runtime {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
