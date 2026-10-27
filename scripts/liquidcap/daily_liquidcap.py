"""LIQUIDCAP paper-trading book — daily EOD job (GitHub Actions, liquidcap.yml).

Frozen spec (combo screen 2026-07-06, `GI_fs15_fund_20d`):
  universe   current S&P 500 members (membership_sp500.parquet; refresh the
             membership monthly with build_universe_sp500.py)
  model      LGB LambdaRank 16 bins, 300 trees, 25 features =
             FS15 price/volume + 10 SEC-EDGAR fundamental ratios
  portfolio  app.engine.cohort_book — the SAME engine the backtest runs
             (scripts/liquidcap/backtest_cohorts.py): 4 overlapping cohorts, one
             rotates every 5 sessions into the top-8 (max 2 per GICS sector),
             trailing ATR [10%,16%] + profit target +40% + 20-session expiry,
             fills at the close ±15 bps. Until 2026-10 the live book ran 8
             refill-on-exit slots instead (≈1.3x the validated volatility).
  backtest   +1.69%/mo flat 5bps (+0.95%/mo spread-aware), 34 purged folds
             2018-2026. "Sharpe 2.43" is the MEAN of per-fold 3-month Sharpes; the
             monthly portfolio Sharpe rebuilt from the trades is ~0.7 (≈ SPY's) —
             the edge is excess return, not a smoother ride. LIVE PAPER IS THE ARBITER.

Flow per run (after US close):
  1. incremental yfinance update of the panel (current members + SPY)
  2. weekly: refresh EDGAR facts + rebuild fundamentals + FULL feature build
     + retrain (purged by construction: recent rows have NaN 20d targets and
     are dropped) -> model pkl
  3. daily: features on a trailing window; score EVERY session not processed
     yet (a missed run is caught up automatically, in order)
  4. engine step per session: exits at close(S), then the due cohort enters
     the top-8 at that SAME close(S) (market-on-close simulation)
  5. publish: state, signals, NAV per session, dashboard view -> Supabase

Usage:
    PYTHONPATH=src python scripts/liquidcap/daily_liquidcap.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "liquidcap"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import lightgbm as lgb  # noqa: E402
from build_features_liquidcap import add_new_features  # noqa: E402
from build_fundamentals import RATIOS  # noqa: E402

from app.books.publish import publish_book  # noqa: E402
from app.data import supabase_store  # noqa: E402
from app.data.free_sources.yahoo import download_yahoo_ohlcv  # noqa: E402
from app.engine import cohort_book as cb  # noqa: E402
from app.features.lags import add_lag_features  # noqa: E402
from app.features.leak_checks import assert_no_leak  # noqa: E402
from app.features.pipeline import build_feature_matrix  # noqa: E402
from app.features.tradability import tradable_mask  # noqa: E402
from app.utils import notify_telegram  # noqa: E402

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
CFG = cb.BookConfig()    # frozen live config (cost 15 bps/side = IBKR-like + slippage)
SECTORS_FP = DATA / "gics_sectors.json"

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
    merged panel is also written to ohlcv_sp500.parquet (gitignored) for local
    research runs (the evaluation harness reads it)."""
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
    feats = add_lag_features(feats)
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
    assert_no_leak(tr, feat_cols, TARGET)                     # raises -> model not saved
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

    # Score on a trailing window (features need 252 trading days) — every
    # session the book has not processed yet, not just the latest one.
    cutoff = panel["date"].max() - pd.Timedelta(days=SCORE_WINDOW_DAYS)
    feats = build_features(panel[panel.date >= cutoff], spy[spy.date >= cutoff], members)
    meta = json.loads((DATA / "model_meta.json").read_text())
    model = lgb.Booster(model_file=str(MODEL_FP))
    sectors = json.loads(SECTORS_FP.read_text(encoding="utf-8")) if SECTORS_FP.exists() else {}

    raw = supabase_store.read_state(STRATEGY)
    if raw is None:
        state = cb.new_book(1000.0, CFG, last_update=str((feats["date"].max()
                                                          - pd.Timedelta(days=1)).date()))
        print("  new cohort book (€1,000)")
    elif raw.get("engine") != "cohorts_v1":
        state = cb.from_slot_book(raw, CFG, sectors)       # one-off, no sales
        print(f"  migrated 8-slot book -> cohorts ({len(state.legacy)} legacy positions)")
    else:
        state = cb.from_dict(raw)

    ohlcv_live = panel[panel.ticker.isin(members)]
    closes_by = {str(d.date()): dict(zip(g.ticker, g.close, strict=True))
                 for d, g in ohlcv_live.groupby("date")}
    todo = sorted(d for d in feats["date"].unique()
                  if str(pd.Timestamp(d).date()) > state.last_update)
    today = str(feats["date"].max().date())

    # Corporate actions: today's (split-adjusted) history vs the stored entry closes.
    entry_closes = {(p.ticker, p.entry_date): closes_by.get(p.entry_date, {}).get(p.ticker)
                    for c in state.cohorts for p in c.positions} | {
                    (p.ticker, p.entry_date): closes_by.get(p.entry_date, {}).get(p.ticker)
                    for p in state.legacy}
    rescaled = cb.reconcile_splits(state, entry_closes, CFG)
    if rescaled:
        print(f"  split-adjusted positions: {rescaled}")

    nav, entered, exited, latest_ranked = {}, [], [], []
    for d in todo:
        ds = str(pd.Timestamp(d).date())
        day = feats[feats["date"] == d].dropna(subset=["close"]).copy()
        day = day[tradable_mask(day)]
        day["score"] = model.predict(day[meta["features"]].fillna(0).values)
        day = day.sort_values("score", ascending=False)
        ranked = [cb.Candidate(r.ticker, float(r.score), float(r.atr_pct_20d),
                               sectors.get(r.ticker)) for r in day.itertuples()]
        out = cb.step(state, ds, closes_by.get(ds, {}), ranked, CFG)
        nav[ds] = cb.equity(state, closes_by.get(ds, {}))
        entered += [f"{t} {ds}" for t in out["entered"]]
        exited += out["exited"]
        latest_ranked, latest_out = ranked, out
    already = not todo
    print(f"  {today}: processed {len(todo)} session(s) · entered {entered or '—'} · "
          f"exited {[(t['ticker'], t['exit_reason']) for t in exited] or '—'}")

    if args.dry_run:
        print(f"  DRY RUN — equity €{cb.equity(state, closes_by.get(today, {})):,.2f}, "
              f"cohort cash {[round(c.cash, 2) for c in state.cohorts]}, "
              f"reserve {state.reserve:.2f}")
        return

    # Idempotent: a re-run on the same session processes nothing and only
    # republishes the view/NAV (repairs a stale web after a partial failure).
    signals = None
    if not already:
        top = {c.ticker for c in latest_ranked[:CFG.top_k]}
        signals = [{"signal_date": today, "ticker": c.ticker, "score": round(c.score, 6),
                    "recommendation": "BUY", "was_traded": c.ticker in latest_out["entered"],
                    "skip_reason": "" if c.ticker in latest_out["entered"] else
                    ("cohorte no rota hoy" if latest_out["cohort"] is None else "límite de sector"),
                    "actual_ret_20d": None} for c in latest_ranked if c.ticker in top]
    view = publish_book(STRATEGY, cb.to_dict(state, CFG, closes_by.get(today, {})),
                        ohlcv_live, PT_DIR, nav=nav or None, signals=signals,
                        write_state=not already)
    if view is not None and not already:
        paper = view["paper"]
        notify_telegram(
            f"✅ <b>SCAI LiquidCap OK</b> — {date.today():%Y-%m-%d}\n"
            f"📅 Sesiones procesadas: <b>{len(todo)}</b> (última {today})   "
            f"🔁 Retrain: {'sí' if retrained else 'no'}\n"
            f"🟢 Abiertas: {', '.join(entered) if entered else '—'}\n"
            f"🔴 Cerradas: {', '.join(t['ticker'] for t in exited) if exited else '—'}\n"
            f"<b>LiquidCap</b>  €{paper['total_value']:,.2f} "
            f"({paper['total_return']:+.2f}%) · {paper['n_open']} pos · "
            f"WR {paper['win_rate']:.0f}%")
    print(f"  Runtime {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
