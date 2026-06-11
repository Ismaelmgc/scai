#!/usr/bin/env python3
"""Cooldown analysis: measure impact of preventing immediate re-entry after trailing stop.

Simulates the production backtest (LambdaRank, 28 features, TOP_K=8, trailing stops ATR [10-16%])
with and without a cooldown period that blocks re-entry after a trailing stop exit.

Tests cooldown periods of 0 (baseline), 3, 5, 7, 10 days.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ── Production config (from daily_pipeline.py) ──────────────────────
V2_TARGET = "fwd_ret_20d_sector_rel"
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
ALL_FEATURES = V2_FEATURES + V2_EDGAR_FEATURES

V3_N_BINS = 16
V2_LGB_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg",
    "num_leaves": 31, "max_depth": 6,
    "learning_rate": 0.05, "min_child_samples": 30,
    "subsample": 0.75, "colsample_bytree": 0.7,
    "reg_lambda": 5.0,
    "lambdarank_truncation_level": 8,
    "label_gain": list(range(V3_N_BINS)),
    "n_jobs": 1, "seed": 42, "verbose": -1,
}
NUM_BOOST_ROUND = 600
TOP_K = 8
HOLD_DAYS = 20
COST_BPS = 15  # commission + slippage


def load_data():
    """Load features + OHLCV using same method as daily_pipeline."""
    from app.data.store.parquet_store import ParquetStore
    from app.features.pipeline import build_feature_matrix

    store = ParquetStore()
    ohlcv = store.read("ohlcv_smallcap")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    # Add Yahoo backfill
    if store.exists("ohlcv_smallcap_yahoo"):
        yahoo = store.read("ohlcv_smallcap_yahoo")
        if yahoo is not None and not yahoo.empty:
            yahoo["date"] = pd.to_datetime(yahoo["date"])
            ohlcv = pd.concat([yahoo, ohlcv], ignore_index=True)
            ohlcv = ohlcv.drop_duplicates(subset=["ticker", "date"], keep="last")
            ohlcv = ohlcv.sort_values(["ticker", "date"])

    # Market data for beta
    market_df = None
    try:
        spy = store.read("smallcap_spy")
        if spy is not None and not spy.empty:
            market_df = spy
    except Exception:
        pass

    features = build_feature_matrix(ohlcv, market_df=market_df, horizons=[1, 5, 10, 20])
    features["date"] = pd.to_datetime(features["date"])

    # Add EDGAR
    try:
        from app.data.free_sources.edgar import load_edgar_features
        edgar = load_edgar_features()
        if edgar is not None and not edgar.empty:
            edgar["date"] = pd.to_datetime(edgar["date"])
            edgar_cols = [c for c in V2_EDGAR_FEATURES if c in edgar.columns]
            if edgar_cols:
                features = features.merge(edgar[["ticker", "date"] + edgar_cols],
                                          on=["ticker", "date"], how="left")
    except Exception:
        pass

    # Target is already computed by build_feature_matrix (fwd_ret_20d_sector_rel)

    # Add OHLCV for backtest execution (only if not already present)
    ohlcv_cols = ["open", "close", "high", "low", "volume"]
    missing_cols = [c for c in ohlcv_cols if c not in features.columns]
    if missing_cols:
        merge_cols = ["ticker", "date"] + [c for c in ohlcv_cols if c in ohlcv.columns]
        features = features.merge(
            ohlcv[merge_cols],
            on=["ticker", "date"], how="left",
        )
    return features


def train_model(train_data: pd.DataFrame):
    """Train LambdaRank model on training data."""
    avail = [f for f in ALL_FEATURES if f in train_data.columns]
    clean = train_data.dropna(subset=[V2_TARGET]).copy()
    clean["_rel"] = clean.groupby("date")[V2_TARGET].transform(
        lambda s: pd.qcut(s.rank(method="first"), V3_N_BINS,
                          labels=False, duplicates="drop")
    )
    clean["_rel"] = clean["_rel"].fillna(0).astype(int).clip(0, V3_N_BINS - 1)

    groups = clean.groupby("date").size().values
    ds = lgb.Dataset(clean[avail], label=clean["_rel"], group=groups, free_raw_data=False)
    model = lgb.train(V2_LGB_PARAMS, ds, num_boost_round=NUM_BOOST_ROUND)
    return model, avail


def generate_signals(model, features_today, feature_cols):
    """Score and rank tickers, return top-K as BUY signals."""
    avail = [f for f in feature_cols if f in features_today.columns]
    X = features_today[avail].copy()
    scores = model.predict(X)
    features_today = features_today.copy()
    features_today["score"] = scores
    ranked = features_today.sort_values("score", ascending=False)
    top_k = ranked.head(TOP_K)

    # ATR-adaptive trailing stop
    if "close" in features_today.columns and "high" in features_today.columns:
        # Simple ATR proxy from features
        atr_col = None
        for c in features_today.columns:
            if "atr" in c.lower():
                atr_col = c
                break
        if atr_col is None:
            # Compute simple ATR proxy
            top_k = top_k.copy()
            top_k["_trail"] = 0.16
        else:
            med_atr = top_k[atr_col].median()
            if med_atr > 0:
                top_k = top_k.copy()
                top_k["_trail"] = np.clip(0.16 * (top_k[atr_col] / med_atr), 0.10, 0.16)
            else:
                top_k = top_k.copy()
                top_k["_trail"] = 0.16
    else:
        top_k = top_k.copy()
        top_k["_trail"] = 0.16

    return top_k


def run_backtest_with_cooldown(features: pd.DataFrame, cooldown_days: int,
                               train_end: str = "2025-06-01",
                               test_start: str = "2025-06-01",
                               test_end: str = "2026-05-20",
                               retrain_every: int = 63) -> dict:
    """Run walk-forward backtest with specified cooldown.

    cooldown_days: after a trailing stop exit, block re-entry for this many trading days.
                   0 = no cooldown (current behavior).
    """
    features = features.sort_values("date").copy()
    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)

    # Get all trading dates in test period
    all_dates = sorted(features["date"].unique())
    test_dates = [d for d in all_dates if test_start_ts <= d <= test_end_ts]

    if not test_dates:
        return {}

    cost_pct = COST_BPS / 10_000

    # State
    capital = 1000.0
    positions = {}  # ticker -> {shares, entry_price, entry_date, high_price, trail_pct, entry_idx}
    pending = []    # [{ticker, trail_pct, signal_date}]
    cooldown_until = {}  # ticker -> date (cannot re-enter before this date)

    trades = []
    portfolio_values = []
    model = None
    feature_cols = None
    last_train_date = None
    last_signal_date = None
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    total_blocked_by_cooldown = 0
    total_immediate_reentries = 0  # would-have-been reentries (cooldown=0 tracking)

    for dt in test_dates:
        dt_idx = date_to_idx[dt]
        day_data = features[features["date"] == dt]

        # Retrain if needed
        if model is None or (last_train_date and (dt - last_train_date).days >= retrain_every):
            train_data = features[features["date"] <= dt]
            train_data = train_data[train_data[V2_TARGET].notna()]
            if len(train_data) >= 5000:
                model, feature_cols = train_model(train_data)
                last_train_date = dt

        if model is None:
            portfolio_values.append({"date": dt, "value": capital})
            continue

        # 1. Execute pending signals at today's open
        new_pending = []
        for sig in pending:
            ticker = sig["ticker"]
            ticker_data = day_data[day_data["ticker"] == ticker]
            if ticker_data.empty:
                sig.setdefault("_retry", 0)
                sig["_retry"] += 1
                if sig["_retry"] <= 3:
                    new_pending.append(sig)
                continue

            # Check cooldown
            if ticker in cooldown_until and dt < cooldown_until[ticker]:
                total_blocked_by_cooldown += 1
                continue

            if ticker in positions:
                continue

            open_price = float(ticker_data.iloc[0]["open"])
            entry_price = open_price * (1 + cost_pct)
            alloc = (1.0 / TOP_K) * (capital + sum(
                p["shares"] * day_data[day_data["ticker"] == t].iloc[0]["close"]
                for t, p in positions.items()
                if not day_data[day_data["ticker"] == t].empty
            ))
            shares = int(alloc / entry_price) if entry_price > 0 else 0
            if shares <= 0:
                continue

            capital -= shares * entry_price
            positions[ticker] = {
                "shares": shares,
                "entry_price": entry_price,
                "entry_date": dt,
                "high_price": open_price,
                "trail_pct": sig["trail_pct"],
                "entry_idx": dt_idx,
            }
            trades.append({"date": dt, "ticker": ticker, "action": "BUY",
                           "price": entry_price, "shares": shares})
        pending = new_pending

        # 2. Update positions: trailing stop + expiry
        to_close = []
        for ticker, pos in positions.items():
            ticker_data = day_data[day_data["ticker"] == ticker]
            if ticker_data.empty:
                continue
            current_price = float(ticker_data.iloc[0]["close"])

            # Update watermark
            if current_price > pos["high_price"]:
                pos["high_price"] = current_price

            exit_reason = None
            trail_trigger = pos["high_price"] * (1 - pos["trail_pct"])
            if current_price <= trail_trigger:
                exit_reason = "trailing_stop"

            days_held = dt_idx - pos["entry_idx"]
            if days_held >= HOLD_DAYS:
                exit_reason = "expiry"

            if exit_reason:
                to_close.append((ticker, exit_reason, current_price))

        for ticker, reason, exit_px in to_close:
            pos = positions.pop(ticker)
            sell_price = exit_px * (1 - cost_pct)
            proceeds = pos["shares"] * sell_price
            capital += proceeds
            pnl_pct = (sell_price / pos["entry_price"]) - 1
            trades.append({"date": dt, "ticker": ticker, "action": "SELL",
                           "price": sell_price, "shares": pos["shares"],
                           "reason": reason, "pnl_pct": pnl_pct})

            # Set cooldown if trailing stop
            if reason == "trailing_stop" and cooldown_days > 0:
                cd_target_idx = dt_idx + cooldown_days
                cd_dates = [d for d in all_dates if date_to_idx[d] >= cd_target_idx]
                if cd_dates:
                    cooldown_until[ticker] = cd_dates[0]

        # 3. Generate new signals (rebalance every 5 days)
        if last_signal_date is None or dt_idx - date_to_idx.get(last_signal_date, 0) >= 5:
            day_features = day_data.dropna(subset=[f for f in feature_cols if f in day_data.columns][:5])
            if len(day_features) >= 10:
                top_k = generate_signals(model, day_features, feature_cols)
                for _, row in top_k.iterrows():
                    ticker = row["ticker"]
                    if ticker in positions:
                        continue
                    if any(p["ticker"] == ticker for p in pending):
                        continue
                    if len(positions) + len(pending) >= TOP_K:
                        break

                    # Track immediate reentries
                    just_closed = [t[0] for t in to_close]
                    if ticker in just_closed:
                        total_immediate_reentries += 1

                    # Check cooldown
                    if ticker in cooldown_until and dt < cooldown_until[ticker]:
                        total_blocked_by_cooldown += 1
                        continue

                    pending.append({
                        "ticker": ticker,
                        "trail_pct": float(row.get("_trail", 0.16)),
                        "signal_date": dt,
                    })
                last_signal_date = dt

        # Mark-to-market
        position_value = 0
        for ticker, pos in positions.items():
            ticker_data = day_data[day_data["ticker"] == ticker]
            if not ticker_data.empty:
                position_value += pos["shares"] * float(ticker_data.iloc[0]["close"])
            else:
                position_value += pos["shares"] * pos["entry_price"]
        portfolio_values.append({"date": dt, "value": capital + position_value})

    # Compute metrics
    pv = pd.DataFrame(portfolio_values).set_index("date")["value"]
    daily_ret = pv.pct_change().dropna()

    trades_df = pd.DataFrame(trades)
    sells = trades_df[trades_df["action"] == "SELL"] if not trades_df.empty else pd.DataFrame()

    total_return = (pv.iloc[-1] / pv.iloc[0] - 1) * 100 if len(pv) > 1 else 0
    ann_vol = daily_ret.std() * np.sqrt(252) if len(daily_ret) > 1 else 0
    sharpe = (daily_ret.mean() * 252) / ann_vol if ann_vol > 0 else 0

    cum = (1 + daily_ret).cumprod()
    max_dd = (cum / cum.cummax() - 1).min() * 100 if len(cum) > 0 else 0

    n_trades = len(sells)
    n_winners = len(sells[sells["pnl_pct"] > 0]) if not sells.empty and "pnl_pct" in sells.columns else 0
    win_rate = n_winners / n_trades * 100 if n_trades > 0 else 0

    n_stop_exits = len(sells[sells["reason"] == "trailing_stop"]) if not sells.empty and "reason" in sells.columns else 0
    avg_pnl_stop = sells[sells["reason"] == "trailing_stop"]["pnl_pct"].mean() * 100 if n_stop_exits > 0 else 0
    avg_pnl_expiry = sells[sells["reason"] == "expiry"]["pnl_pct"].mean() * 100 if not sells.empty and len(sells[sells["reason"] == "expiry"]) > 0 else 0

    return {
        "cooldown": cooldown_days,
        "total_return": round(total_return, 1),
        "sharpe": round(sharpe, 2),
        "max_dd": round(max_dd, 1),
        "n_trades": n_trades,
        "win_rate": round(win_rate, 1),
        "n_stop_exits": n_stop_exits,
        "avg_pnl_stop": round(avg_pnl_stop, 1),
        "avg_pnl_expiry": round(avg_pnl_expiry, 1),
        "blocked_by_cooldown": total_blocked_by_cooldown,
        "immediate_reentries": total_immediate_reentries,
        "final_value": round(pv.iloc[-1], 2) if len(pv) > 0 else 1000,
    }


def main():
    print("=" * 70)
    print("  COOLDOWN ANALYSIS: Impact of blocking immediate re-entry")
    print("  after trailing stop exit")
    print("=" * 70)
    print()

    print("Loading data and building features...")
    features = load_data()
    print(f"  {len(features):,} rows, {features['date'].nunique()} dates, "
          f"{features['ticker'].nunique()} tickers")
    print()

    cooldowns = [0, 3, 5, 7, 10]
    results = []

    for cd in cooldowns:
        label = f"cooldown={cd}d" if cd > 0 else "NO cooldown (current)"
        print(f"Running backtest: {label} ...", end=" ", flush=True)
        r = run_backtest_with_cooldown(features, cooldown_days=cd)
        results.append(r)
        print(f"done → return={r['total_return']:+.1f}%, sharpe={r['sharpe']:.2f}")

    print()
    print("=" * 70)
    print("  RESULTS COMPARISON")
    print("=" * 70)
    print()

    header = f"{'Cooldown':>12} {'Return%':>9} {'Sharpe':>7} {'MaxDD%':>7} {'Trades':>7} {'WinR%':>6} {'StopEx':>7} {'AvgStop%':>9} {'AvgExp%':>8} {'Blocked':>8} {'Reentries':>10}"
    print(header)
    print("-" * len(header))

    baseline = results[0]
    for r in results:
        cd_label = f"{r['cooldown']}d" if r['cooldown'] > 0 else "none"
        diff = r['total_return'] - baseline['total_return']
        diff_str = f"({diff:+.1f})" if r['cooldown'] > 0 else ""
        print(f"{cd_label:>12} {r['total_return']:>+8.1f}% {r['sharpe']:>7.2f} {r['max_dd']:>6.1f}% "
              f"{r['n_trades']:>7} {r['win_rate']:>5.1f}% {r['n_stop_exits']:>7} "
              f"{r['avg_pnl_stop']:>+8.1f}% {r['avg_pnl_expiry']:>+7.1f}% "
              f"{r['blocked_by_cooldown']:>8} {r['immediate_reentries']:>10} {diff_str}")

    print()
    print("Legend:")
    print("  StopEx     = Exits via trailing stop (vs expiry)")
    print("  AvgStop%   = Avg PnL on trailing stop exits")
    print("  AvgExp%    = Avg PnL on holding period expiry exits")
    print("  Blocked    = Signals blocked by cooldown rule")
    print("  Reentries  = Same-day stop+rebuy (the problem we're measuring)")


if __name__ == "__main__":
    main()
