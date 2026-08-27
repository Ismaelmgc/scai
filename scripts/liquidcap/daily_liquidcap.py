"""LIQUIDCAP paper-trading book — daily EOD job (phase 1: runs locally).

Frozen spec (combo screen 2026-07-06, `GI_fs15_fund_20d`):
  universe   current S&P 500 members (membership_sp500.parquet; refresh the
             membership monthly with build_universe_sp500.py)
  model      LGB LambdaRank 16 bins, 300 trees, 25 features =
             FS15 price/volume + 10 SEC-EDGAR fundamental ratios
  portfolio  top-8 equal-weight, hold 20d, rebalance implicit via slots,
             trailing ATR [10%,16%] + profit target +40%, cost-aware fills
  backtest   +1.69%/mo flat 5bps (+0.95%/mo spread-aware), fold-Sharpe 2.43,
             34 purged folds 2018-2026 — LIVE PAPER MUST CONFIRM THIS.

Flow per run (after US close):
  1. incremental yfinance update of the panel (current members + SPY)
  2. weekly: refresh EDGAR facts + rebuild fundamentals + FULL feature build
     + retrain (purged by construction: recent rows have NaN 20d targets and
     are dropped) -> model pkl
  3. daily: features on a trailing window, score the latest session
  4. PaperTrader: update positions/exits at close(S) -> select today's top-8
     -> enter them at that SAME close(S) (MOC-sim close-entry, matches backtest)
  5. state -> Supabase strategy "liquidcap"

Usage:
    PYTHONPATH=src python scripts/liquidcap/daily_liquidcap.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "liquidcap"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import lightgbm as lgb  # noqa: E402
from run_smallcap_pipeline import _add_lag_features  # noqa: E402

from app.data import supabase_store  # noqa: E402
from app.data.free_sources.yahoo import download_yahoo_ohlcv  # noqa: E402
from app.features.pipeline import build_feature_matrix  # noqa: E402
from app.paper_trading import PaperTrader  # noqa: E402
from app.utils import notify_telegram  # noqa: E402
from build_features_liquidcap import add_new_features  # noqa: E402
from build_fundamentals import RATIOS  # noqa: E402

DATA = ROOT / "data" / "liquidcap"
PANEL_FP = DATA / "ohlcv_sp500.parquet"
SPY_FP = DATA / "spy.parquet"
FUND_FP = DATA / "fundamentals_daily.parquet"
MODEL_FP = DATA / "model_liquidcap.pkl"
PT_DIR = DATA / "paper_trading"
STRATEGY = "liquidcap"
TARGET = "fwd_ret_20d_sector_rel"
RETRAIN_DAYS = 7
SCORE_WINDOW_DAYS = 700  # calendar; covers the 252-trading-day features

FS15 = ["ret_252d", "sector_ret_60d", "rev_21d", "realized_vol_120d", "amihud_60d",
        "beta_60d", "downside_vol_60d", "sma_200", "ema_26", "max_dd_60d",
        "vol_of_vol_ratio", "adv_60d", "intraday_avg_20d", "price_roc_smooth_120d",
        "obv"]
FEATURES = FS15 + RATIOS

LGB_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg",
    "num_leaves": 31, "max_depth": 6,
    "learning_rate": 0.05, "min_child_samples": 30,
    "subsample": 0.75, "colsample_bytree": 0.7,
    "reg_lambda": 5.0, "n_jobs": 1, "seed": 42, "verbose": -1,
    "lambdarank_truncation_level": 8, "label_gain": list(range(16)),
}


DELISTED_FP = DATA / "ohlcv_delisted.parquet"  # committed, static (anti-survivorship)


def current_members() -> list[str]:
    mem = pd.read_parquet(DATA / "membership_sp500.parquet")
    return sorted(mem[mem["end"] >= mem["end"].max()]["ticker"].unique())


def assemble_panel(members: list[str]) -> pd.DataFrame:
    """Full training/scoring panel = committed static delisted slice (Tiingo,
    never changes) + a FRESH yfinance download of current members + SPY.

    Rebuilding current bars every run (instead of an incremental append) both
    kills the retroactive dividend re-adjustment drift and keeps the CI stateless
    — only the tiny delisted slice, model and EDGAR facts need to persist. The
    merged panel is written for morning_execute's view and gitignored."""
    delisted = pd.read_parquet(DELISTED_FP)
    delisted["date"] = pd.to_datetime(delisted["date"])
    print(f"  delisted (committed, static): {delisted.ticker.nunique()} tickers")
    fresh = download_yahoo_ohlcv(members + ["SPY"], start_date="2014-01-01")
    fresh = fresh[["date", "ticker", "open", "high", "low", "close", "volume"]].copy()
    fresh["date"] = pd.to_datetime(fresh["date"]).dt.tz_localize(None).dt.normalize()
    fresh["close_unadj"] = np.nan
    panel = pd.concat([delisted, fresh], ignore_index=True)
    panel = panel.drop_duplicates(["date", "ticker"], keep="last")
    panel.to_parquet(PANEL_FP, index=False)
    spy = panel[panel.ticker == "SPY"]
    spy.to_parquet(SPY_FP, index=False)
    print(f"  panel: {len(panel):,} rows -> {panel['date'].max().date()} "
          f"({panel.ticker.nunique()} tickers)")
    return panel


def build_features(panel: pd.DataFrame, spy: pd.DataFrame,
                   members: list[str]) -> pd.DataFrame:
    """Same path as the validated backtest build, restricted to live members."""
    px = panel[panel.ticker.isin(members)].copy()
    universe = [{"ticker": t} for t in px["ticker"].unique()]
    feats = build_feature_matrix(px, fundamentals=None, market_df=spy,
                                 universe=universe, horizons=[1, 5, 10, 20])
    feats = _add_lag_features(feats)
    feats = add_new_features(feats, px, spy)
    feats["date"] = pd.to_datetime(feats["date"])
    fnd = pd.read_parquet(FUND_FP)
    fnd["date"] = pd.to_datetime(fnd["date"])
    return feats.merge(fnd, on=["date", "ticker"], how="left")


def retrain(panel: pd.DataFrame, spy: pd.DataFrame, members: list[str]) -> None:
    """Weekly full retrain. Purged by construction: the last 20 sessions have
    NaN forward targets and are dropped — the model never sees the future."""
    print("  refreshing EDGAR fundamentals (new filings)...")
    subprocess.run([sys.executable, str(ROOT / "scripts/liquidcap/build_fundamentals.py"),
                    "--refresh-facts"],
                   env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
                   check=True)
    print("  full feature build + retrain...")
    feats = build_features(panel, spy, members)
    tr = feats.dropna(subset=[TARGET]).copy()
    tr["_rel"] = tr.groupby("date")[TARGET].transform(
        lambda s: pd.qcut(s.rank(method="first"), 16, labels=False, duplicates="drop"))
    tr["_rel"] = tr["_rel"].fillna(0).astype(int).clip(0, 15)
    tr = tr.sort_values("date")
    feat_cols = [f for f in FEATURES if f in tr.columns]
    ds = lgb.Dataset(tr[feat_cols].fillna(0).values, tr["_rel"].values,
                     group=tr.groupby("date").size().values,
                     feature_name=feat_cols, free_raw_data=True)
    model = lgb.train(LGB_PARAMS, ds, num_boost_round=300,
                      callbacks=[lgb.log_evaluation(0)])
    model.save_model(str(MODEL_FP))
    (DATA / "model_meta.json").write_text(json.dumps({
        "trained": date.today().isoformat(), "rows": len(tr),
        "features": feat_cols, "spec": "GI_fs15_fund_20d"}))
    print(f"  model saved ({len(tr):,} rows, {len(feat_cols)} features)")


def model_stale() -> bool:
    meta_fp = DATA / "model_meta.json"
    if not MODEL_FP.exists() or not meta_fp.exists():
        return True
    trained = date.fromisoformat(json.loads(meta_fp.read_text())["trained"])
    return (date.today() - trained).days >= RETRAIN_DAYS


def ensure_fundamentals() -> None:
    """fundamentals_daily (66M) is regenerated, not committed. Rebuild it from
    the committed EDGAR facts cache (7.7M) when absent — scoring needs it every
    run. No --refresh-facts here (no re-download); retrain() refreshes weekly."""
    if FUND_FP.exists():
        return
    print("  rebuilding fundamentals_daily from committed EDGAR facts...")
    subprocess.run([sys.executable, str(ROOT / "scripts/liquidcap/build_fundamentals.py")],
                   env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
                   check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    members = current_members()
    panel = assemble_panel(members)
    spy = panel[panel.ticker == "SPY"].copy()

    ensure_fundamentals()
    retrained = model_stale()
    if retrained:
        retrain(panel, spy, members)

    # Daily scoring on a trailing window (features need 252 trading days)
    cutoff = panel["date"].max() - pd.Timedelta(days=SCORE_WINDOW_DAYS)
    feats = build_features(panel[panel.date >= cutoff], spy[spy.date >= cutoff],
                           members)
    latest = feats["date"].max()
    today = str(latest.date())
    day = feats[feats["date"] == latest].dropna(subset=["close"]).copy()
    meta = json.loads((DATA / "model_meta.json").read_text())
    model = lgb.Booster(model_file=str(MODEL_FP))
    day["score"] = model.predict(day[meta["features"]].fillna(0).values)
    day = day.sort_values("score", ascending=False)

    top = day.head(8).copy()
    atr = top["atr_pct_20d"].fillna(0.02).clip(lower=0.005)
    top["recommendation"] = "BUY"
    top["position_size_pct"] = 1.0 / 8
    top["trailing_stop_pct"] = (atr * 5.3).clip(0.10, 0.16)
    top["stop_loss_pct"] = top["trailing_stop_pct"]
    signals = top[["ticker", "recommendation", "position_size_pct",
                   "trailing_stop_pct", "stop_loss_pct"]]
    print(f"  {today}: top-8 = {list(signals.ticker)}")

    if args.dry_run:
        print("  DRY RUN — no state changes")
        return

    PT_DIR.mkdir(parents=True, exist_ok=True)
    state = supabase_store.read_state(STRATEGY)
    p_path = PT_DIR / "portfolio.json"
    if state is not None:
        p_path.write_text(json.dumps(state))
    pt = PaperTrader.load_or_create(
        str(p_path), initial_capital=1000.0, max_positions=8,
        holding_period_days=20, adaptive_stop=False, profit_target=0.40)

    # Idempotency guard: never PROCESS the same session twice. `today` is the
    # latest bar in the panel, which does NOT change until the next US session —
    # so a second run on a weekend/holiday (or a double CI run) sees the same
    # `today`. Re-processing would double-enter the top-8 (at the close, below)
    # and double-advance the day index. So we skip the trading mutations when
    # already advanced — but STILL (re)publish the view + NAV below, so a re-run
    # after a partial failure (e.g. a crash between the state write and the view
    # write) self-heals the dashboard instead of leaving it frozen on a stale
    # session.
    ohlcv_today = panel[panel.ticker.isin(members)]
    already = bool(pt.state.last_update and str(pt.state.last_update) >= today)
    if already:
        print(f"  session {today} already processed "
              f"(last_update={pt.state.last_update}) — refreshing view + NAV only")
        entered, closed, traded, skipped = [], [], set(), {}
    else:
        # CLOSE-entry (MOC-sim, 2026-08): the signal is generated from close(S)
        # and we enter at that SAME close(S) — matching the backtest exactly and
        # capturing the overnight drift that the old open(S+1) fill forfeited
        # (~+0.16pp/mo). Order: exits on held positions first, then queue today's
        # top-8, then fill them at close(S) in this same nightly run. The official
        # yfinance close is the fill price (no Finnhub → no source-gap). For real
        # money this is a ~15:45-ET MOC order; the paper book simulates it since
        # the slow daily model's top-8 barely moves in the last 15 min.
        closed = pt.update_positions(ohlcv_today, today)
        traded, skipped = pt.process_signals(signals, today)
        entered = pt.execute_pending(ohlcv_today, today, price_col="close")
        pt.save()
        supabase_store.write_state(STRATEGY, asdict(pt.state))
        # Persist the day's top-8 to the signals table so the S&P 500 tab shows a
        # signal history like the small-cap books (load_signal_history reads it).
        # actual_ret_20d stays null (no live-IC monitor here).
        supabase_store.upsert_signals(STRATEGY, [{
            "signal_date": today,
            "ticker": r["ticker"],
            "score": round(float(r["score"]), 6),
            "recommendation": "BUY",
            "was_traded": r["ticker"] in traded,
            "skip_reason": skipped.get(r["ticker"], ""),
            "actual_ret_20d": None,
        } for _, r in top.iterrows()])

    # Always (re)publish the render-ready view + one NAV point for the session.
    # Both are idempotent upserts (keyed by strategy[,date]) reflecting current
    # state, so republishing on an already-processed re-run repairs a stale web.
    from app.web import dashboard_data
    view = dashboard_data.build_view(ohlcv_today, PT_DIR, adaptive_stop=False,
                                     strategy=STRATEGY)
    if view is not None:
        supabase_store.write_dashboard_view(STRATEGY, view)
        supabase_store.upsert_nav(STRATEGY, today, float(view["paper"]["total_value"]))
    print(f"  entered={entered or '[]'} closed={[t.ticker for t in closed] or '[]'} "
          f"queued={sorted(traded) or '[]'} skipped={skipped}")

    # Telegram run summary (best-effort; no-op without secrets) — only on a real
    # processing run, not a view-only refresh (which has nothing new to report).
    if view is not None and not already:
        paper = view["paper"]
        opened = ", ".join(entered) if entered else "—"
        closed_names = ", ".join(t.ticker for t in closed) if closed else "—"
        notify_telegram(
            f"✅ <b>SCAI LiquidCap OK</b> — {date.today():%Y-%m-%d}\n"
            f"📅 Última sesión en datos: <b>{today}</b>\n"
            f"🟢 Señales BUY (top-8): <b>{len(traded)}</b>   "
            f"🔁 Retrain: {'sí' if retrained else 'no'}\n"
            f"🟢 Abiertas: {opened}\n"
            f"🔴 Cerradas: {closed_names}\n"
            f"<b>LiquidCap</b>  €{paper['total_value']:,.2f} "
            f"({paper['total_return']:+.2f}%) · {paper['n_open']} pos · "
            f"WR {paper['win_rate']:.0f}%"
        )
    print(f"  Runtime {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
