"""ILLIQUID paper book — out-of-sample test of the small-cap illiquidity edge
(see [[smallcap-illiquidity-edge]]) as a real €1000 rolling-8 book, tracked on the
web dashboard like the other strategies (Supabase state + view + NAV), WITHOUT
touching the baseline book.

Selection = production tradability gate → bottom ADV tercile → drop widest-20%
cs_spread_20d → rank by the production model score (all from OHLC-derived features;
no quotes). Exits = champion policy (trailing ATR clip[10%,16%] / +40% PT / 20d).

⚠ GROSS caveat: this fills at the open like every paper book, but illiquid names
have far worse REAL fills (market impact). Net-of-impact the edge is ~+0.64%/mo at
$5k and dies by ~$100k (capacity analysis). Read the dashboard curve as SELECTION
alpha vs IWM, NOT an executable return at size.

Flow mirrors scripts/liquidcap/daily_liquidcap.py (proven PaperTrader+Supabase
pattern). Run daily AFTER the daily pipeline refreshes features_smallcap.parquet.

    PYTHONPATH=src python scripts/illiquid_shadow.py            # update + publish
    PYTHONPATH=src python scripts/illiquid_shadow.py --dry-run  # score only, no writes
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.features.tradability import tradable_mask  # noqa: E402
from app.data import supabase_store  # noqa: E402
from app.paper_trading import PaperTrader  # noqa: E402
from app.utils import notify_telegram  # noqa: E402

# reuse the EXACT production feature lists + model path from the daily pipeline
_spec = importlib.util.spec_from_file_location("dp", ROOT / "scripts/daily_pipeline.py")
dp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(dp)
FEAT = [f for f in (dp.V2_FEATURES + dp.V2_EDGAR_FEATURES + dp.V2_META_FEATURES)]

FEATURES_FP = ROOT / "data/processed/features_smallcap.parquet"
OHLCV_FP = ROOT / "data/processed/ohlcv_smallcap.parquet"
MODEL_FP = ROOT / str(dp.MODEL_PATH)
PT_DIR = ROOT / "data/paper_trading/illiquid"
STRATEGY = "illiquid"

MAX_POS = 8
HOLD_DAYS = 20
TRAIL_MULT, TRAIL_MIN, TRAIL_MAX = 5.3, 0.10, 0.16
PROFIT_TARGET = 0.40
ADV_TERCILE = 1.0 / 3.0
SPREAD_TRIM_Q = 0.80

# the pre-Supabase isolated book (gross rolling-8 log) — migrated ONCE into the
# new PaperTrader/Supabase book so the ~1 month it already ran is preserved.
LEGACY_FP = ROOT / "data/paper_trading/illiquid_shadow/book.json"


def _migrate_legacy() -> dict | None:
    """Seed the fresh Supabase book from the legacy gross log so the month it
    already ran (8 open + 9 closed) carries over. Models each slot as a fixed €125
    notional (1000/8): closed trades bank their realised € to cash, opens keep
    their €125 cost basis → total = 1000 + realised + unrealised. Returns a
    PortfolioState-shaped dict, or None if the legacy log is absent."""
    if not LEGACY_FP.exists():
        return None
    b = json.loads(LEGACY_FP.read_text())
    notional = 1000.0 / MAX_POS
    day_idx = 100  # arbitrary base; entry_day_idx preserves each pos's days_held
    positions = []
    for p in b.get("positions", []):
        ep = float(p["entry_price"])
        trail = float(np.clip(float(p.get("atr", 0.03)) * TRAIL_MULT, TRAIL_MIN, TRAIL_MAX))
        peak = float(p.get("peak", ep))
        positions.append({
            "ticker": p["ticker"], "side": "LONG", "shares": round(notional / ep, 6),
            "entry_price": ep, "entry_date": p["entry_date"],
            "entry_day_idx": day_idx - int(p.get("days_held", 0)),
            "trailing_stop_pct": round(trail, 4), "holding_period_days": HOLD_DAYS,
            "high_price": peak, "low_price": ep,
            "stop_loss_price": round(peak * (1 - trail), 4),
            "position_size_pct": 1.0 / MAX_POS,
        })
    closed = []
    for t in b.get("closed_trades", []):
        ep = float(t["entry_price"]); ret = float(t["ret"])
        closed.append({
            "ticker": t["ticker"], "side": "LONG", "shares": round(notional / ep, 6),
            "entry_price": ep, "entry_date": t["entry_date"],
            "exit_price": float(t["exit_price"]), "exit_date": t["exit_date"],
            "exit_reason": t["exit_reason"], "pnl_pct": ret,
            "pnl_usd": round(notional * ret, 2), "days_held": int(t.get("holding_days", 0)),
        })
    cash = round(sum(c["pnl_usd"] for c in closed), 2)
    print(f"  migrated legacy book: {len(positions)} open + {len(closed)} closed (cash €{cash})")
    return {
        "initial_capital": 1000.0, "cash": cash,
        "positions": positions, "closed_trades": closed, "current_day_idx": day_idx,
        "last_update": b.get("last_marked", ""), "pending_signals": [], "cooldown_until": {},
        "max_positions": MAX_POS, "holding_period_days": HOLD_DAYS,
        "commission_bps": 5.0, "slippage_bps": 10.0, "cooldown_days": 5,
    }


def _reconstruct_nav(ohlcv: pd.DataFrame) -> list[tuple[str, float]]:
    """Rebuild the daily €NAV series over the legacy month from the trade ledger,
    so the dashboard chart shows the real evolution (+ IWM for the same window)
    like the other books. Same fixed-€125-notional model as the migration: each
    day NAV = 1000 − 125·(entered≤d) + Σ realised(exited≤d) + Σ held·close(d).
    Missing closes forward-fill; unknown → entry price. [] if no legacy log."""
    if not LEGACY_FP.exists():
        return []
    b = json.loads(LEGACY_FP.read_text())
    notional = 1000.0 / MAX_POS
    trades = []
    for t in b.get("closed_trades", []):
        trades.append((t["ticker"], pd.Timestamp(t["entry_date"]), float(t["entry_price"]),
                       pd.Timestamp(t["exit_date"]), float(t["exit_price"])))
    for p in b.get("positions", []):
        trades.append((p["ticker"], pd.Timestamp(p["entry_date"]), float(p["entry_price"]),
                       None, None))
    if not trades:
        return []
    start = min(t[1] for t in trades)
    end = pd.Timestamp(b["last_marked"]) if b.get("last_marked") else max(t[1] for t in trades)
    held = {t[0] for t in trades}
    oh = ohlcv[ohlcv["ticker"].isin(held)]
    px = (oh.pivot_table(index="date", columns="ticker", values="close")
          .sort_index().ffill())
    sessions = [d for d in px.index if start <= d <= end]
    series = []
    for d in sessions:
        cash, eq = 1000.0, 0.0
        for tk, ed, ep, xd, xp in trades:
            if ed > d:
                continue
            shares = notional / ep
            cash -= notional
            if xd is not None and xd <= d:
                cash += shares * xp
            else:
                c = px.loc[d, tk] if tk in px.columns else np.nan
                eq += shares * (float(c) if np.isfinite(c) else ep)
        series.append((str(d.date()), round(cash + eq, 2)))
    return series


def _backfill_nav_if_missing(ohlcv: pd.DataFrame) -> None:
    """One-off: upsert the reconstructed legacy-month NAV points that Supabase is
    missing (idempotent — keyed by date). Runs every time but writes nothing once
    the history is present, so it's safe to leave in the daily path."""
    try:
        have = {str(n["date"])[:10] for n in (supabase_store.read_nav(STRATEGY) or [])}
        missing = [(d, v) for d, v in _reconstruct_nav(ohlcv) if d not in have]
        for d, v in missing:
            supabase_store.upsert_nav(STRATEGY, d, v)
        if missing:
            print(f"  backfilled {len(missing)} historical NAV points "
                  f"({missing[0][0]} → {missing[-1][0]})")
    except Exception as e:  # best-effort; never break the daily publish
        print(f"  NAV backfill skipped: {e}")


def illiquid_ranked(day: pd.DataFrame, model) -> pd.DataFrame:
    """Tradable → bottom-ADV tercile → drop widest-20% spread → ranked by score."""
    day = day[tradable_mask(day)].copy()
    if len(day) < MAX_POS:
        return pd.DataFrame()
    cols = [f for f in FEAT if f in day.columns]
    day["score"] = model.predict(day[cols].fillna(0).values)
    low = day[day["adv_usd_20d"].rank(pct=True) <= ADV_TERCILE]
    if low.empty:
        return pd.DataFrame()
    thr = low["cs_spread_20d"].quantile(SPREAD_TRIM_Q)
    low = low[low["cs_spread_20d"].fillna(low["cs_spread_20d"].median()) <= thr]
    return low.sort_values("score", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    feat = pd.read_parquet(FEATURES_FP)
    feat["date"] = pd.to_datetime(feat["date"])
    with open(MODEL_FP, "rb") as f:
        model = pickle.load(f)  # noqa: S301
    latest = feat["date"].max()
    today = str(latest.date())

    ranked = illiquid_ranked(feat[feat["date"] == latest], model)
    if ranked.empty:
        print(f"  {today}: no illiquid candidates (need >= {MAX_POS} tradable) — skip")
        return
    top = ranked.head(MAX_POS).copy()
    atr = top["atr_pct_20d"].fillna(0.03).clip(lower=0.005)
    top["recommendation"] = "BUY"
    top["position_size_pct"] = 1.0 / MAX_POS
    top["trailing_stop_pct"] = (atr * TRAIL_MULT).clip(TRAIL_MIN, TRAIL_MAX)
    top["stop_loss_pct"] = top["trailing_stop_pct"]
    signals = top[["ticker", "recommendation", "position_size_pct",
                   "trailing_stop_pct", "stop_loss_pct"]]
    print(f"  {today}: illiquid top-{MAX_POS} = {list(signals.ticker)}")

    if args.dry_run:
        print("  DRY RUN — no state changes")
        return

    ohlcv = pd.read_parquet(OHLCV_FP)
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    PT_DIR.mkdir(parents=True, exist_ok=True)
    state = supabase_store.read_state(STRATEGY)
    if state is None:
        state = _migrate_legacy()  # first run: carry over the legacy month
    p_path = PT_DIR / "portfolio.json"
    if state is not None:
        p_path.write_text(json.dumps(state))
    pt = PaperTrader.load_or_create(
        str(p_path), initial_capital=1000.0, max_positions=MAX_POS,
        holding_period_days=HOLD_DAYS, adaptive_stop=False, profit_target=PROFIT_TARGET)

    # Idempotency guard (same as liquidcap): never PROCESS a session twice — `today`
    # is the latest bar and doesn't move until the next US session, so a double run
    # would fill this session's own pending at this session's open (look-ahead) and
    # inflate the day index. Skip mutations when already advanced, but STILL
    # (re)publish the view + NAV so a re-run after a partial failure self-heals.
    already = bool(pt.state.last_update and str(pt.state.last_update) >= today)
    if already:
        print(f"  session {today} already processed "
              f"(last_update={pt.state.last_update}) — refreshing view + NAV only")
        entered, closed, traded, skipped = [], [], set(), {}
    else:
        entered = pt.execute_pending(ohlcv, today)
        closed = pt.update_positions(ohlcv, today)
        traded, skipped = pt.process_signals(signals, today)
        pt.save()
        supabase_store.write_state(STRATEGY, asdict(pt.state))
        supabase_store.upsert_signals(STRATEGY, [{
            "signal_date": today,
            "ticker": r["ticker"],
            "score": round(float(r["score"]), 6),
            "recommendation": "BUY",
            "was_traded": r["ticker"] in traded,
            "skip_reason": skipped.get(r["ticker"], ""),
            "actual_ret_20d": None,
        } for _, r in top.iterrows()])

    from app.web import dashboard_data
    view = dashboard_data.build_view(ohlcv, PT_DIR, adaptive_stop=False, strategy=STRATEGY)
    if view is not None:
        supabase_store.write_dashboard_view(STRATEGY, view)
        supabase_store.upsert_nav(STRATEGY, today, float(view["paper"]["total_value"]))
        _backfill_nav_if_missing(ohlcv)  # fill the legacy month's daily NAV (once)
    print(f"  entered={entered or '[]'} closed={[t.ticker for t in closed] or '[]'} "
          f"queued={sorted(traded) or '[]'} skipped={skipped}")

    if view is not None and not already:
        paper = view["paper"]
        opened = ", ".join(entered) if entered else "—"
        closed_names = ", ".join(t.ticker for t in closed) if closed else "—"
        notify_telegram(
            f"✅ <b>SCAI Illiquid (shadow OOS)</b> — {date.today():%Y-%m-%d}\n"
            f"📅 Última sesión: <b>{today}</b>\n"
            f"🟢 Abiertas: {opened}\n"
            f"🔴 Cerradas: {closed_names}\n"
            f"<b>Illiquid</b>  €{paper['total_value']:,.2f} "
            f"({paper['total_return']:+.2f}%) · {paper['n_open']} pos · "
            f"WR {paper['win_rate']:.0f}%  ⚠️ GROSS (no descuenta impacto)"
        )
    print(f"  Runtime {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
