"""ILLIQUID shadow BOOK — out-of-sample test of the small-cap illiquidity edge
(see [[smallcap-illiquidity-edge]]) as a FAITHFUL rolling 8-position book, WITHOUT
touching the live paper books.

Unlike a cohort logger, this holds each name until its REAL exit (champion policy:
trailing stop clip(ATR*5.3,[10%,16%]) / +40% profit target / 20 trading-day time
stop), refilling freed slots from the top illiquid picks (5-day cooldown after an
exit) — exactly how the live book behaves, so "the tickers only change when one
exits", not on a fixed cadence. Isolated: own JSON state + trades parquet, no
Supabase, no dashboard. Marks positions every trading day; refills once per new
feature date. Trade returns are GROSS (raw price moves) — real net is lower for
illiquid names (market impact), see the capacity analysis; this measures whether
the held book's SELECTION beats IWM over each position's actual holding window.

Selection = production tradability gate → bottom ADV tercile → drop widest-20%
cs_spread_20d → rank by model score (all from OHLC-derived features; no quotes).

    PYTHONPATH=src python scripts/illiquid_shadow.py          # update book + score
    PYTHONPATH=src python scripts/illiquid_shadow.py --score  # score only
Run daily AFTER the daily pipeline refreshes features (features_smallcap.parquet).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.features.tradability import tradable_mask  # noqa: E402

# reuse the EXACT production feature lists + model path from the daily pipeline
_spec = importlib.util.spec_from_file_location("dp", ROOT / "scripts/daily_pipeline.py")
dp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(dp)
FEAT = [f for f in (dp.V2_FEATURES + dp.V2_EDGAR_FEATURES + dp.V2_META_FEATURES)]

FEATURES_FP = ROOT / "data/processed/features_smallcap.parquet"
OHLCV_FP = ROOT / "data/processed/ohlcv_smallcap.parquet"
MODEL_FP = ROOT / str(dp.MODEL_PATH)
STATE_FP = ROOT / "data/paper_trading/illiquid_shadow/book.json"
TRADES_FP = ROOT / "data/paper_trading/illiquid_shadow/trades.parquet"
BENCH_FP = ROOT / "data/processed/bench_iwm_spy.parquet"

MAX_POS = 8
HOLD_DAYS = 20        # time stop (trading days)
COOLDOWN = 5          # trading days after an exit before re-entering a name
TRAIL_MULT, TRAIL_MIN, TRAIL_MAX = 5.3, 0.10, 0.16
PROFIT_TARGET = 0.40
ADV_TERCILE = 1.0 / 3.0
SPREAD_TRIM_Q = 0.80


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


def load_state() -> dict:
    if STATE_FP.exists():
        return json.loads(STATE_FP.read_text())
    return {"last_marked": None, "last_refill": None, "positions": [], "closed_trades": []}


def save_state(state: dict) -> None:
    STATE_FP.parent.mkdir(parents=True, exist_ok=True)
    STATE_FP.write_text(json.dumps(state, indent=2))
    if state["closed_trades"]:
        pd.DataFrame(state["closed_trades"]).to_parquet(TRADES_FP, index=False)


def update_book() -> None:
    feat = pd.read_parquet(FEATURES_FP)
    feat["date"] = pd.to_datetime(feat["date"])
    latest_feat = feat["date"].max()
    with open(MODEL_FP, "rb") as f:
        model = pickle.load(f)  # noqa: S301
    ohlcv = pd.read_parquet(OHLCV_FP, columns=["date", "ticker", "close"])
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    all_dates = np.sort(ohlcv["date"].unique())
    ohlcv_max = all_dates[-1]

    state = load_state()
    pos, closed = state["positions"], state["closed_trades"]
    last_marked = pd.Timestamp(state["last_marked"]) if state["last_marked"] else None

    # ── 1. refill freed slots at the latest feature/signal date (5-day cooldown),
    #        once per feature date. New names enter at that day's close. ──
    if state["last_refill"] != str(latest_feat.date()) and len(pos) < MAX_POS:
        recent = [d for d in all_dates if d <= latest_feat]
        cutoff = recent[-COOLDOWN] if len(recent) >= COOLDOWN else recent[0]
        cooling = {t["ticker"] for t in closed
                   if pd.Timestamp(t["exit_date"]) >= pd.Timestamp(cutoff)}
        held = {p["ticker"] for p in pos}
        ranked = illiquid_ranked(feat[feat["date"] == latest_feat], model)
        for _, row in ranked.iterrows():
            if len(pos) >= MAX_POS:
                break
            tk = row["ticker"]
            if tk in held or tk in cooling:
                continue
            atr = float(row.get("atr_pct_20d", 0.03))
            pos.append({
                "ticker": tk, "entry_date": str(latest_feat.date()),
                "entry_price": float(row["close"]), "peak": float(row["close"]),
                "atr": atr if np.isfinite(atr) and atr > 0 else 0.03, "days_held": 0,
            })
            held.add(tk)
        state["last_refill"] = str(latest_feat.date())

    # ── 2. mark every trading day since the last mark (skipping each position's
    #        own entry day), apply exits. Includes names just refilled above. ──
    mark_from = last_marked if last_marked is not None else latest_feat
    mark_days = [d for d in all_dates if d > mark_from and d <= ohlcv_max]
    if mark_days and pos:
        held = list({p["ticker"] for p in pos})
        lut = ohlcv[ohlcv["ticker"].isin(held)].set_index(["ticker", "date"])["close"]
        for d in mark_days:
            for p in list(pos):
                if pd.Timestamp(p["entry_date"]) >= d:
                    continue  # not yet held on day d
                try:
                    px = float(lut.loc[(p["ticker"], d)])
                except KeyError:
                    continue  # no bar this day (halt/missing) — skip
                p["days_held"] += 1
                p["peak"] = max(p["peak"], px)
                trail = float(np.clip(p["atr"] * TRAIL_MULT, TRAIL_MIN, TRAIL_MAX))
                reason = None
                if px >= p["entry_price"] * (1 + PROFIT_TARGET):
                    reason = "profit_target"
                elif (px - p["peak"]) / p["peak"] <= -trail:
                    reason = "trailing_stop"
                elif p["days_held"] >= HOLD_DAYS:
                    reason = "time_stop"
                if reason:
                    closed.append({
                        "ticker": p["ticker"], "entry_date": p["entry_date"],
                        "entry_price": round(p["entry_price"], 4),
                        "exit_date": str(pd.Timestamp(d).date()), "exit_price": round(px, 4),
                        "ret": round(px / p["entry_price"] - 1, 4), "exit_reason": reason,
                        "holding_days": p["days_held"],
                    })
                    pos.remove(p)
    state["last_marked"] = str(pd.Timestamp(ohlcv_max).date())

    save_state(state)
    n_closed_today = sum(1 for t in closed if t["exit_date"] == str(pd.Timestamp(ohlcv_max).date()))
    print(f"  book @ {ohlcv_max.astype('datetime64[D]')}: {len(pos)} held, "
          f"{len(closed)} closed total ({n_closed_today} today)")
    print(f"    holding: {', '.join(p['ticker'] for p in pos) or '(none)'}")


def _iwm(start, end) -> pd.DataFrame:
    b = pd.read_parquet(BENCH_FP) if BENCH_FP.exists() else pd.DataFrame(columns=["date", "IWM"])
    if not b.empty:
        b["date"] = pd.to_datetime(b["date"])
    if b.empty or b["date"].max() < end or b["date"].min() > start:
        import yfinance as yf
        d = yf.download("IWM", start=str((start - pd.Timedelta(days=5)).date()),
                        end=str((end + pd.Timedelta(days=5)).date()), auto_adjust=True, progress=False)
        s = d["Close"]; s = s.iloc[:, 0] if hasattr(s, "columns") else s
        b = s.rename("IWM").reset_index(); b.columns = ["date", "IWM"]; b["date"] = pd.to_datetime(b["date"])
        b.to_parquet(BENCH_FP, index=False)
    return b


def do_score() -> None:
    state = load_state()
    closed = state["closed_trades"]
    if not closed:
        print("  no closed trades yet — book still holding its first positions")
        _show_open(state)
        return
    tr = pd.DataFrame(closed)
    tr["entry_date"] = pd.to_datetime(tr["entry_date"]); tr["exit_date"] = pd.to_datetime(tr["exit_date"])
    bench = _iwm(tr["entry_date"].min(), tr["exit_date"].max())
    alphas = []
    for _, t in tr.iterrows():
        bw = bench[(bench["date"] >= t["entry_date"]) & (bench["date"] <= t["exit_date"])].dropna(subset=["IWM"])
        iwm = float(bw["IWM"].iloc[-1] / bw["IWM"].iloc[0] - 1) if len(bw) >= 2 else np.nan
        alphas.append(t["ret"] - iwm)
    tr["alpha_vs_iwm"] = alphas
    print(f"\n  === ILLIQUID BOOK — {len(tr)} closed trades (held to real exits) ===")
    print(f"  win rate {(tr['ret'] > 0).mean():.0%}   mean ret {tr['ret'].mean():+.2%}   "
          f"median {tr['ret'].median():+.2%}   mean α vs IWM {np.nanmean(tr['alpha_vs_iwm']):+.2%}")
    print(f"  exit reasons: {tr['exit_reason'].value_counts().to_dict()}   "
          f"mean hold {tr['holding_days'].mean():.0f}d")
    print("  (GROSS returns — real net lower for illiquid names via market impact)")
    _show_open(state)


def _show_open(state: dict) -> None:
    pos = state["positions"]
    if not pos:
        return
    ohlcv = pd.read_parquet(OHLCV_FP, columns=["date", "ticker", "close"])
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    last = ohlcv["date"].max()
    px = ohlcv[ohlcv["date"] == last].set_index("ticker")["close"]
    print(f"\n  open positions ({len(pos)}):")
    for p in pos:
        cur = float(px.get(p["ticker"], p["entry_price"]))
        print(f"    {p['ticker']:6s} entry {p['entry_price']:.2f} → {cur:.2f}  "
              f"({cur/p['entry_price']-1:+.1%})  held {p['days_held']}d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true", help="score closed trades only")
    args = ap.parse_args()
    if not args.score:
        update_book()
    do_score()


if __name__ == "__main__":
    main()
