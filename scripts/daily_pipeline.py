#!/usr/bin/env python3
"""SCAI small-cap daily pipeline — data, features, weekly retrain and model signals.

Model: LightGBM LambdaRank on fwd_ret_20d_sector_rel (sector-relative 20-day
return binned into 16 relevance levels), 28 features (26 price/volume + 2 EDGAR),
retrained every 7 days on the FULL stored OHLCV (delisted names included →
no survivorship bias). Selection = tradability gate → top-8 by score.

No paper book trades these signals any more: the "baseline" book was retired
2026-10 (no edge after realistic costs) and replaced by the BTC trend book. The
pipeline still runs because (1) the illiquid shadow book scores with this model
and these features, and (2) the live-IC monitor measures the model out of sample
from the signals recorded here (Supabase `signals`, historical key "baseline").

Usage:
    python scripts/daily_pipeline.py
    python scripts/daily_pipeline.py --force-retrain
    python scripts/daily_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from app.config import get_settings
from app.smallcap.data import update_ohlcv  # noqa: E402
from app.smallcap.features import rebuild_features  # noqa: E402
from app.smallcap.model import (  # noqa: E402,F401 — re-exported for illiquid_shadow / tests / leak CLI
    MODEL_PATH,
    RETRAIN_EVERY_DAYS,
    V2_EDGAR_FEATURES,
    V2_FEATURES,
    V2_TARGET,
    _days_since_train,
    train_or_load_models,
)
from app.smallcap.signals import (  # noqa: E402,F401
    conviction_sizes,
    generate_today_signals,
    record_signals,
)
from app.utils import get_logger, notify_telegram, set_global_seed, setup_logging

log = get_logger(__name__)

DAILY_LOG_PATH = "data/paper_trading/daily_log.jsonl"




def _already_ran_today() -> bool:
    """Check if the pipeline already ran today (idempotency guard)."""
    p = Path(DAILY_LOG_PATH)
    if not p.exists():
        return False
    today_str = date.today().isoformat()
    # Read last few lines efficiently
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("date") == today_str and not entry.get("dry_run"):
                    return True
            except json.JSONDecodeError:
                continue
    return False


def _log_daily(entry: dict) -> None:
    """Append a JSON-lines entry to the daily log."""
    p = Path(DAILY_LOG_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")






# ── Step 1: Incremental data download ─────────────────────



# ── Step 2: Rebuild features ──────────────────────────────





# ── Step 3: Train or load models ──────────────────────────





# ── Step 4: Generate today's signals ──────────────────────





# ── Step 5: Record signals (live-IC monitor) ──────────



# ── Main ──────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SCAI Daily Pipeline")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Force model retraining regardless of schedule")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate signals but don't write them anywhere")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip OHLCV download (use cached data)")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip feature rebuild (use cached features)")
    parser.add_argument("--train-start", default="2020-01-01",
                        help="Training data start date")
    parser.add_argument("--predict-from", default=None,
                        help="Prediction start date (default: 30 days ago)")
    args = parser.parse_args()

    cfg = get_settings()
    setup_logging("INFO")
    set_global_seed(cfg.seed)

    today = date.today().isoformat()

    # Never act on an in-progress (partial) daily bar. If the US market hasn't
    # closed yet, the current day's bar is incomplete and would trigger trailing
    # stops against intraday prices. In that case operate as of the last
    # completed session (this mirrors running `scai` after close, as on macOS).
    now_et = datetime.now(ZoneInfo("America/New_York"))
    market_closed = now_et.weekday() < 5 and now_et.time() >= time(16, 15)
    # Target the most recent ET session. The daily cron runs in the EVENING
    # (~23:00 UTC, 2-3h after the 16:00 ET close), so today's bar is already
    # settled and published — not in-progress. (The old "free publishes on T+1"
    # belief was wrong: the only 403 is "today's data before end of day"; once the
    # session closes the grouped bar is served. Running well after the close — not
    # pegged to it — avoids that 403, and a stray one is caught + skipped by
    # download_ohlcv_grouped.) now_et.date() stays correct even if GitHub delays the
    # run past midnight UTC (ET is still the same trading day). Weekend/holiday gaps
    # and any in-progress day are handled downstream by bdate_range + empty/skipped
    # grouped responses. `market_closed` below still guards partial-bar drops.
    predict_to = now_et.date().isoformat()

    # Idempotency: skip if already ran today (safe to re-trigger on wake/login)
    if not args.dry_run and _already_ran_today():
        print(f"  ✓ Already ran today ({today}). Skipping. Use --dry-run to re-check.\n")
        return

    from app.data.store.parquet_store import ParquetStore
    store = ParquetStore()

    # Weekends (ET): normally skip — no new session — BUT still run if the data is
    # behind, so a failed Friday run self-heals on Sat/Sun instead of waiting until
    # Monday. "Behind" = there's at least one business day after our last stored bar
    # up to today (a holiday in that range just fetches empty, harmless). ET is used
    # so a delayed Friday-evening run (lands on Sat UTC but is still Fri ET) isn't
    # mistaken for a weekend; and on a true weekend the catch-up still rescues a
    # delayed run that did roll over.
    if now_et.weekday() >= 5:  # Saturday=5, Sunday=6 (ET)
        stored_max = pd.to_datetime(store.read("ohlcv_smallcap")["date"]).max().date()
        behind = len(pd.bdate_range(stored_max + timedelta(days=1), now_et.date())) > 0
        if not behind:
            print(f"  ℹ Weekend ({now_et:%A}), data current ({stored_max}). Skipping.\n")
            if not args.dry_run:
                notify_telegram(f"ℹ️ <b>SCAI daily</b> — {now_et:%A}: fin de semana, "
                                 f"datos al día (sesión {stored_max}). Sin ejecución.")
            return
        print(f"  ⚠ Weekend ({now_et:%A}) but data is behind (last={stored_max}) "
              f"— running to catch up.\n")

    # Default predict_from: 30 days back (enough for signal generation context)
    if args.predict_from is None:
        predict_from = (date.today() - timedelta(days=30)).isoformat()
    else:
        predict_from = args.predict_from

    print()
    print("=" * 60)
    print("  SCAI Daily Pipeline")
    print(f"  Date:          {today}")
    print(f"  Mode:          {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"  Retrain:       {'FORCED' if args.force_retrain else f'every {RETRAIN_EVERY_DAYS}d (last: {_days_since_train()}d ago)'}")
    print("=" * 60)
    print()

    # ── Step 1: Update data ──────────────────────────
    print("STEP 1/5 ▸ Updating market data...")
    if args.skip_download:
        ohlcv = store.read("ohlcv_smallcap")
        ohlcv["date"] = pd.to_datetime(ohlcv["date"])
        print(f"  Using cached OHLCV ({ohlcv['date'].max().date()})")
    else:
        ohlcv = update_ohlcv(cfg, predict_to)

    # Drop any partial bar for the in-progress day (e.g. left by a mid-session
    # download) so trailing stops are only ever evaluated on completed sessions.
    if not market_closed:
        cal_today = date.today()
        mask_today = ohlcv["date"].dt.date == cal_today
        if mask_today.any():
            n_partial = int(mask_today.sum())
            ohlcv = ohlcv[~mask_today].copy()
            if not args.dry_run:
                store.write("ohlcv_smallcap", ohlcv)  # purge so it's re-fetched after close
            print(f"  ⚠ Market open (ET {now_et:%H:%M}): dropped {n_partial} partial bars "
                  f"for {cal_today}; using last completed session")
    # Re-anchor 'today' to the latest completed session actually present in data
    today = ohlcv["date"].max().date().isoformat()
    print()

    # ── Step 2: Rebuild features ─────────────────────
    print("STEP 2/5 ▸ Building features...")
    if args.skip_features:
        features = store.read("features_smallcap")
        print(f"  Using cached features ({len(features):,} rows)")
    else:
        features = rebuild_features(ohlcv, cfg)
    print()

    # ── Step 3: Train or load models ──────────────────
    print("STEP 3/5 ▸ Models...")
    model, predict_data, train_metrics = (
        train_or_load_models(features, predict_from, cfg, args.force_retrain)
    )
    print()

    # ── Step 4: Generate signals ─────────────────────
    print("STEP 4/5 ▸ Generating signals...")
    signals = generate_today_signals(model, predict_data, ohlcv, today)

    # Save signals
    if not signals.empty:
        signals_path = f"data/paper_trading/signals_{today}.parquet"
        Path(signals_path).parent.mkdir(parents=True, exist_ok=True)
        signals.to_parquet(signals_path, index=False)
    print()

    # ── Step 5: Record signals ───────────────────────
    print("STEP 5/5 ▸ Recording signals...")
    record_signals(signals, ohlcv, today, args.dry_run)
    print()

    # Telegram run summary (best-effort; no-op without secrets). Shows the latest
    # session actually in the data so a stale-but-green run is still visible.
    if not args.dry_run:
        n_buy = int((signals["recommendation"] == "BUY").sum()) if not signals.empty else 0
        # Download health: tickers in the latest session vs the median of the 5
        # prior sessions. A partial frontier (e.g. 2026-07-01 landed with 1 ticker
        # when the grouped bar hadn't published) shows up as ⚠️ PARCIAL instead of
        # hiding behind a green run.
        _counts = (ohlcv.assign(_d=pd.to_datetime(ohlcv["date"]))
                   .groupby("_d")["ticker"].nunique().sort_index())
        n_latest = int(_counts.iloc[-1])
        n_ref = int(_counts.iloc[-6:-1].median()) if len(_counts) > 1 else n_latest
        cov_ok = n_ref == 0 or n_latest >= 0.9 * n_ref
        cov_line = (
            f"📥 Descarga: <b>{n_latest}</b>/{n_ref} tickers del universo ✅"
            if cov_ok else
            f"📥 Descarga: <b>{n_latest}</b>/{n_ref} tickers "
            f"⚠️ <b>PARCIAL</b> — revisar"
        )
        # Name the ACTIVE tickers missing the latest session (with how many
        # sessions behind) so the "315/316" count says WHICH ones: a "(-1)" is a
        # benign self-healing miss, a "(-3)" is a real gap worth investigating.
        gap_line = ""
        try:
            from app.data.store.parquet_store import ParquetStore as _PS
            _uni = _PS().read("smallcap_universe")
            _active = (set(_uni[_uni["active"] == True]["ticker"])
                       if "active" in _uni.columns else set(_uni["ticker"]))
            _sess = sorted(pd.to_datetime(ohlcv["date"]).dt.normalize().unique())
            _last = (ohlcv[ohlcv["ticker"].isin(_active)]
                     .assign(_d=pd.to_datetime(ohlcv["date"]))
                     .groupby("ticker")["_d"].max())
            _miss = _last[_last < _sess[-1]]
            if len(_miss):
                _behind = sorted(((tk, int(sum(s > d for s in _sess)))
                                  for tk, d in _miss.items()), key=lambda x: -x[1])
                _names = ", ".join(f"{tk}(-{n})" for tk, n in _behind[:12])
                gap_line = (f"\n⚠️ Sin barra hoy ({len(_miss)}): {_names}"
                            f"{' …' if len(_miss) > 12 else ''}")
        except Exception:
            gap_line = ""
        notify_telegram(
            f"✅ <b>SCAI daily OK</b> — {date.today():%Y-%m-%d}\n"
            f"📅 Última sesión en datos: <b>{today}</b>\n"
            f"{cov_line}{gap_line}\n"
            f"🟢 Señales BUY: <b>{n_buy}</b>   🔁 Retrain: {'sí' if train_metrics else 'no'}"
        )

    _log_daily({
        "date": today,
        "n_signals": len(signals),
        "retrained": bool(train_metrics),
        "dry_run": args.dry_run,
    })

    print("  ✓ Done\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Loud failure: alert on Telegram (best-effort) then re-raise so the
        # GitHub run goes RED — no more silent green runs on stale/failed data.
        # Escape the error text: an unescaped '<' (e.g. a 525 "<none>" body) breaks
        # Telegram's HTML parse and 400s the alert — the failure would go unnotified.
        import html
        detail = html.escape(f"{type(e).__name__}: {str(e)[:300]}")
        notify_telegram(
            f"❌ <b>SCAI daily FALLÓ</b> — {date.today():%Y-%m-%d}\n"
            f"<code>{detail}</code>\n"
            f"Revisa el run en GitHub Actions."
        )
        raise
