#!/usr/bin/env python3
"""Reset paper trading and replay May 19-21 with the new clean 28-feature model.

This script:
1. Archives current portfolios (both baseline + adaptive)
2. Creates fresh portfolios ($1000 each)
3. Replays all available trading days from May 19 using the new model
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from app.paper_trading import PaperTrader
from app.signal_tracker import SignalTracker

# ── Config ──────────────────────────────────────────────────
MODEL_PATH = "data/models/smallcap_v3_lambdarank.pkl"
FEATURES_PATH = "data/processed/features_smallcap.parquet"
OHLCV_PATH = "data/processed/ohlcv_smallcap.parquet"
PORTFOLIO_PATH = "data/paper_trading/portfolio.json"
PORTFOLIO_ADAPTIVE = "data/paper_trading/adaptive/portfolio.json"

V2_FEATURES = [
    'max_dd_60d', 'vol_of_vol_60d', 'ret_kurtosis_60d', 'avg_trade_size_20d',
    'obv', 'obv_vs_sma_60d', 'amihud_60d', 'downside_vol_60d', 'ema_26',
    'spread_proxy_20d', 'ret_skew_60d', 'ret_252d', 'sma_200', 'sector_ret_60d',
    'realized_vol_120d', 'macd_hist', 'pct_from_52w_low', 'adv_60d',
    'ret_vs_sector_60d', 'vol_of_vol_ratio', 'price_roc_smooth_120d',
    'vwap_dev_avg_20d', 'reversal_20v60', 'vol_regime_change', 'beta_60d',
    'macd_signal',
]
V2_EDGAR_FEATURES = ['dilution_pct', 'current_ratio']
V2_TOP_K = 8
REPLAY_FROM = "2026-05-19"


def generate_signals(model, features: pd.DataFrame, ohlcv: pd.DataFrame,
                     target_date: str) -> pd.DataFrame:
    """Generate signals for a specific date using the new model."""
    target_ts = pd.Timestamp(target_date)
    day_data = features[features["date"] == target_ts].copy()

    if day_data.empty:
        print(f"    No feature data for {target_date}")
        return pd.DataFrame()

    all_features = [f for f in V2_FEATURES + V2_EDGAR_FEATURES if f in day_data.columns]
    X = day_data[all_features].fillna(0).values
    day_data["v2_score"] = model.predict(X)
    day_data = day_data.sort_values("v2_score", ascending=False)

    records = []
    for rank, (_, row) in enumerate(day_data.iterrows()):
        ticker = str(row.get("ticker", ""))
        if rank < V2_TOP_K:
            recommendation = "BUY"
            position_size = 1.0 / V2_TOP_K
        else:
            recommendation = "HOLD"
            position_size = 0.0

        vol = float(row.get("realized_vol_20d", 0.3))
        atr_pct = float(row.get("atr_pct_20d", vol / np.sqrt(252) * 2))
        median_atr = (
            day_data.head(V2_TOP_K)["atr_pct_20d"].median()
            if "atr_pct_20d" in day_data.columns else 0.03
        )
        if median_atr > 0 and recommendation == "BUY":
            adaptive_trail = np.clip(0.16 * (atr_pct / median_atr), 0.10, 0.16)
        else:
            adaptive_trail = 0.16

        records.append({
            "ticker": ticker, "date": target_date,
            "recommendation": recommendation,
            "ensemble_score": float(row["v2_score"]),
            "calibrated_prob": float(row["v2_score"]),
            "expected_return": float(row["v2_score"]),
            "position_size_pct": position_size,
            "trailing_stop_pct": adaptive_trail,
            "stop_loss_pct": adaptive_trail,
            "rejection_reasons": "",
        })

    return pd.DataFrame(records)


def replay_strategy(model, features, ohlcv, trading_days, portfolio_path, adaptive_stop, label):
    """Replay trading days for one strategy."""
    print(f"\n  ── {label} ──")

    # Create fresh portfolio
    Path(portfolio_path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "initial_capital": 1000.0,
        "cash": 1000.0,
        "positions": [],
        "closed_trades": [],
        "current_day_idx": 0,
        "last_update": "",
        "pending_signals": [],
        "max_positions": 8,
        "holding_period_days": 20,
        "commission_bps": 5.0,
        "slippage_bps": 10.0,
    }
    with open(portfolio_path, "w") as f:
        json.dump(state, f, indent=2)

    pt = PaperTrader.load_or_create(
        portfolio_path, initial_capital=1000.0,
        max_positions=8, holding_period_days=20,
        adaptive_stop=adaptive_stop,
    )

    tracker_path = str(Path(portfolio_path).parent / "signal_history.parquet")
    tracker = SignalTracker(path=tracker_path)

    for i, day in enumerate(trading_days):
        print(f"    Day {i+1}: {day}")

        # 1. Execute pending signals at today's open
        entered = pt.execute_pending(ohlcv, day)
        if entered:
            print(f"      Entered: {', '.join(entered)}")

        # 2. Check trailing stops
        closed = pt.update_positions(ohlcv, day)
        if closed:
            for t in closed:
                print(f"      Closed {t.ticker}: {t.pnl_pct:+.2%} ({t.exit_reason})")

        # 3. Generate signals for this day
        signals = generate_signals(model, features, ohlcv, day)
        if not signals.empty:
            buys = signals[signals["recommendation"] == "BUY"]
            tickers_str = ", ".join(f"{r['ticker']}({r['ensemble_score']:.3f})" for _, r in buys.iterrows())
            print(f"      Signals: {tickers_str}")
            traded, skipped = pt.process_signals(signals, day)
            tracker.record_signals(signals, traded, skipped, day)
            if traded:
                print(f"      Queued: {', '.join(traded)}")
            if skipped:
                skip_str = ", ".join(f"{t}({r})" for t, r in skipped.items())
                print(f"      Skipped: {skip_str}")

    # Save
    pt.save()
    tracker.save()

    # Summary
    summary = pt.summary(ohlcv)
    print(f"\n    Portfolio: €{summary['total_value']:,.2f} ({summary['total_return']})")
    print(f"    Cash: €{summary['cash']:,.2f}")
    print(f"    Positions: {summary['n_open_positions']}")
    if summary["open_positions"]:
        print(f"    {'Ticker':8s} {'Entry':>8s} {'Current':>8s} {'P&L':>8s}")
        for pos in summary["open_positions"]:
            print(f"    {pos['ticker']:8s} {pos['entry_price']:8.2f} {pos['current_price']:8.2f} {pos['pnl_pct']:>8s}")

    return summary


def main():
    import pickle

    print("=" * 60)
    print("  PAPER TRADING RESET + REPLAY")
    print(f"  Replay from: {REPLAY_FROM}")
    print("=" * 60)

    # ── 1. Archive old files ──
    print("\n  Step 1: Archiving old portfolios...")
    for folder in ["data/paper_trading", "data/paper_trading/adaptive"]:
        archive_dir = Path(folder) / "archive_v3_reset"
        archive_dir.mkdir(parents=True, exist_ok=True)
        for fname in ["portfolio.json", "signal_history.parquet", "trades.parquet", "daily_log.jsonl"]:
            fp = Path(folder) / fname
            if fp.exists():
                shutil.copy2(fp, archive_dir / fname)
                print(f"    Archived: {fp}")
    # Also archive signals files
    for f in Path("data/paper_trading").glob("signals_*.parquet"):
        shutil.move(str(f), str(Path("data/paper_trading/archive_v3_reset") / f.name))

    # ── 2. Load model + data ──
    print("\n  Step 2: Loading model and data...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"    Model: {MODEL_PATH}")

    features = pd.read_parquet(FEATURES_PATH)
    features["date"] = pd.to_datetime(features["date"])
    print(f"    Features: {len(features):,} rows")

    ohlcv = pd.read_parquet(OHLCV_PATH)
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    print(f"    OHLCV: {len(ohlcv):,} rows, max date {ohlcv['date'].max().date()}")

    # ── 3. Identify trading days ──
    replay_ts = pd.Timestamp(REPLAY_FROM)
    trading_days = sorted(
        ohlcv[ohlcv["date"] >= replay_ts]["date"].dt.strftime("%Y-%m-%d").unique()
    )
    print(f"\n  Trading days to replay: {trading_days}")

    # ── 4. Replay both strategies ──
    print("\n  Step 3: Replaying...")
    summary_a = replay_strategy(
        model, features, ohlcv, trading_days,
        PORTFOLIO_PATH, adaptive_stop=False, label="Baseline"
    )
    summary_b = replay_strategy(
        model, features, ohlcv, trading_days,
        PORTFOLIO_ADAPTIVE, adaptive_stop=True, label="Adaptive Stop"
    )

    print("\n" + "=" * 60)
    print("  RESET + REPLAY COMPLETE")
    print("=" * 60)
    print(f"  Baseline:  €{summary_a['total_value']:,.2f} | {summary_a['n_open_positions']} positions")
    print(f"  Adaptive:  €{summary_b['total_value']:,.2f} | {summary_b['n_open_positions']} positions")
    print()


if __name__ == "__main__":
    main()
