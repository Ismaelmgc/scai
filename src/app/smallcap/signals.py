"""Small-cap daily signals: tradability gate, top-k, conviction sizes; recording
for the live-IC monitor."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.data import supabase_store
from app.features.tradability import is_stale, tradable_mask
from app.smallcap.model import V2_EDGAR_FEATURES, V2_FEATURES, V2_HOLD_DAYS, V2_TOP_K
from app.utils import get_logger

log = get_logger(__name__)


SIGNAL_HISTORY_PATH = "data/paper_trading/signal_history.parquet"


# Supabase `signals` key: kept as "baseline" (the retired book's name) so the
# live-IC monitor keeps one continuous history.
SIGNALS_KEY = "baseline"


def conviction_sizes(data: pd.DataFrame, top_k: int) -> pd.Series:
    """Conviction-linear position sizing over the tradable top-K.

    Returns a position_size_pct Series aligned to `data`'s index. `data` must
    carry `v2_score` and a boolean `_tradable` column, sorted by score desc.
    Among the top-K tradable names, weight ∝ max(z, 0.1) where z is the
    cross-sectional z-score of v2_score over the FULL tradable universe (so the
    tilt reflects each name's standing vs the whole field, not just within the
    top-K); normalized to sum to 1. Non-BUYs get 0. Degenerate cases (no spread
    in scores, single name) fall back to equal-weight.

    Replaces the old flat 1/K. The pred carries within-top-8 information (the
    #1-ranked beats the #8 on average), so tilting capital toward conviction
    adds return without touching drawdown: +1.6%/fold vs equal-weight, Sharpe
    1.60->1.74, paired-bootstrap CI[+0.3%,+2.9%]>0, 12/16 folds. Validated in
    scripts/v3/43_capital_deployment.py on the v4_fs_base28 prediction cache.
    """
    sizes = pd.Series(0.0, index=data.index)
    tradable = data[data["_tradable"]]
    if tradable.empty:
        return sizes
    buys = tradable.head(top_k)
    sd = float(tradable["v2_score"].std())
    if not (sd > 0) or len(buys) < 2:
        sizes.loc[buys.index] = 1.0 / len(buys)
        return sizes
    z = (buys["v2_score"] - float(tradable["v2_score"].mean())) / sd
    conv = z.clip(lower=0.1)
    sizes.loc[buys.index] = conv / conv.sum()
    return sizes


def generate_today_signals(model, predict_data, ohlcv, today: str) -> pd.DataFrame:
    """Generate V2 signals: rank stocks by predicted sector-relative return."""
    today_ts = pd.Timestamp(today)
    predict_data["date"] = pd.to_datetime(predict_data["date"])

    available_dates = sorted(predict_data["date"].unique())
    if not available_dates:
        print("  ⚠ No prediction data available")
        return pd.DataFrame()

    latest = available_dates[-1]
    today_data = predict_data[predict_data["date"] == latest].copy()

    if today_data.empty:
        print(f"  ⚠ No data for {latest.date()}")
        return pd.DataFrame()

    # Freshness guard: stale features mean the download failed silently —
    # never generate signals from old data.
    if is_stale(latest, today_ts):
        print(f"  ✗ Features are stale (latest={latest.date()}, today={today_ts.date()}) "
              f"— skipping signal generation")
        return pd.DataFrame()

    # Tradability gate: delisted/illiquid tickers stay in OHLCV for training
    # (anti-survivorship) but must never be SELECTED. Live trading bought
    # sub-penny zombies (SRNE @ $0.0006) before this filter existed.
    today_data["_tradable"] = tradable_mask(today_data)
    n_excluded = int((~today_data["_tradable"]).sum())
    print(f"  ✓ Tradability gate: {today_data['_tradable'].sum()} tradable, "
          f"{n_excluded} excluded (price/liquidity)")

    # Score all stocks (base + EDGAR features)
    available_features = [f for f in V2_FEATURES if f in today_data.columns]
    available_edgar = [f for f in V2_EDGAR_FEATURES if f in today_data.columns]
    all_features = available_features + available_edgar
    X = today_data[all_features].fillna(0).values
    today_data["v2_score"] = model.predict(X)

    # Rank by predicted sector-relative return (higher = better)
    today_data = today_data.sort_values("v2_score", ascending=False)
    pos_sizes = conviction_sizes(today_data, V2_TOP_K)

    records = []
    n_buys = 0
    for idx, row in today_data.iterrows():
        ticker = str(row.get("ticker", ""))
        dt_str = str(latest.date())

        rejection = ""
        if not bool(row["_tradable"]):
            recommendation = "HOLD"
            position_size = 0.0
            rejection = "untradable: below min price/liquidity"
        elif n_buys < V2_TOP_K:
            recommendation = "BUY"
            position_size = float(pos_sizes.loc[idx])  # conviction-linear weight
            n_buys += 1
        else:
            recommendation = "HOLD"
            position_size = 0.0

        # ATR-adaptive trailing stop
        vol = float(row.get("realized_vol_20d", 0.3))
        atr_pct = float(row.get("atr_pct_20d", vol / np.sqrt(252) * 2))
        median_atr = (today_data.head(V2_TOP_K)["atr_pct_20d"].median()
                      if "atr_pct_20d" in today_data.columns else 0.03)
        if median_atr > 0 and recommendation == "BUY":
            adaptive_trail = np.clip(0.16 * (atr_pct / median_atr), 0.10, 0.16)
        else:
            adaptive_trail = 0.16

        records.append({
            "ticker": ticker,
            "date": dt_str,
            "recommendation": recommendation,
            "ensemble_score": float(row["v2_score"]),
            "calibrated_prob": float(row["v2_score"]),  # compatibility
            "expected_return": float(row["v2_score"]),
            "position_size_pct": position_size,
            "trailing_stop_pct": adaptive_trail,
            "stop_loss_pct": adaptive_trail,
            "rejection_reasons": rejection,
        })

    signals = pd.DataFrame(records)
    buys = signals[signals["recommendation"] == "BUY"]
    print(f"  ✓ Signals: {len(signals)} total, {len(buys)} BUY")
    if not buys.empty:
        for _, b in buys.iterrows():
            print(f"    BUY {b['ticker']:6s} — score: {b['ensemble_score']:.4f}, "
                  f"trail: {b['trailing_stop_pct']:.0%}")

    return signals


def record_signals(signals: pd.DataFrame, ohlcv: pd.DataFrame, today: str,
                   dry_run: bool) -> None:
    """Track the model's daily signals: local SignalTracker history (20d outcomes
    backfilled as they mature) + Supabase `signals` rows for the live-IC monitor."""
    from app.signal_tracker import SignalTracker

    tracker = SignalTracker(path=SIGNAL_HISTORY_PATH)
    if not signals.empty:
        tracker.record_signals(signals, set(), {}, today)
    n_filled = tracker.backfill_outcomes(ohlcv, horizon_days=V2_HOLD_DAYS)
    if n_filled:
        print(f"  ✓ Outcomes backfilled: {n_filled} signals")
    stats = tracker.summary_stats()
    if "score_return_correlation" in stats:
        print(f"  ✓ Signal history: {stats['total_signals']} signals, "
              f"{stats['outcomes_available']} with outcomes, "
              f"score↔return corr {stats['score_return_correlation']:.3f}")
    tracker.save()

    if dry_run or signals.empty or not supabase_store.is_configured():
        return
    try:
        supabase_store.upsert_signals(SIGNALS_KEY, [{
            "signal_date": today,
            "ticker": r["ticker"],
            "score": float(r["ensemble_score"]),
            "recommendation": r["recommendation"],
            "was_traded": False,
            "skip_reason": r.get("rejection_reasons") or None,
            "actual_ret_20d": None,
        } for _, r in signals.iterrows()])
        print(f"  ✓ Supabase: {len(signals)} signals")
    except Exception as e:  # best-effort: the monitor tolerates a missing day
        log.warning("supabase_signals_failed", error=str(e))
        print(f"  ⚠ Supabase signals write failed: {e}")

