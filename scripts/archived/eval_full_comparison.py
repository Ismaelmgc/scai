#!/usr/bin/env python3
"""Full evaluation: 5yr model vs 2yr baseline — Walk-Forward backtest comparison.

Trains two models:
  A) BASELINE: trained on 2024-05-20 → 2025-05-18 (2yr Polygon-only, ~162K rows)
  B) FULL:     trained on 2021-05-20 → 2025-05-18 (5yr Massive+Yahoo, ~375K rows)

Evaluates both on the SAME validation period: 2025-05-19 → 2026-05-18.
Computes: IC, Sharpe, Top-K returns, trade list, win rate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── CONFIG ──────────────────────────────────────────────────
VALIDATION_START = "2025-05-19"
TRAIN_START_BASELINE = "2024-05-20"  # ~2yr window (original Polygon)
TRAIN_START_FULL = "2021-05-20"      # 5yr window (Massive + Yahoo)

TOP_K = 8
HOLD_DAYS = 20
REBALANCE_EVERY = 5  # days

V2_TARGET = "fwd_ret_20d_sector_rel"
V2_RAW_COL = "fwd_ret_20d"

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
V2_META_FEATURES = [
    'model_error_ticker_5', 'model_error_sector_20d', 'model_hit_rate_30d',
    'model_ic_rolling_20d', 'model_error_vol_20d',
]

V2_LGB_PARAMS = {
    "objective": "regression", "metric": "rmse",
    "num_leaves": 20, "max_depth": 5,
    "learning_rate": 0.02, "min_child_samples": 60,
    "subsample": 0.75, "colsample_bytree": 0.7,
    "reg_alpha": 0.5, "reg_lambda": 5.0,
    "n_jobs": 1, "seed": 42, "verbose": -1,
}


def load_features() -> pd.DataFrame:
    """Load the full feature matrix (already built with Yahoo+Massive+EDGAR+Meta)."""
    from app.data.store.parquet_store import ParquetStore
    store = ParquetStore()
    features = store.read("features_smallcap")
    features["date"] = pd.to_datetime(features["date"])
    return features


def train_model(features: pd.DataFrame, train_start: str, train_end: str,
                use_edgar: bool = True, use_meta: bool = True) -> tuple:
    """Train LGB model on specified date range."""
    import lightgbm as lgb

    mask = (features["date"] >= pd.Timestamp(train_start)) & \
           (features["date"] < pd.Timestamp(train_end))
    train_data = features[mask].copy()

    # Select features
    feat_cols = [f for f in V2_FEATURES if f in train_data.columns]
    if use_edgar:
        feat_cols += [f for f in V2_EDGAR_FEATURES if f in train_data.columns]
    if use_meta:
        feat_cols += [f for f in V2_META_FEATURES if f in train_data.columns]

    train_clean = train_data.dropna(subset=[V2_TARGET])
    X = train_clean[feat_cols].fillna(0).values
    y = train_clean[V2_TARGET].values

    ds = lgb.Dataset(X, y, feature_name=feat_cols, free_raw_data=True)
    model = lgb.train(V2_LGB_PARAMS, ds, num_boost_round=400,
                      callbacks=[lgb.log_evaluation(0)])

    return model, feat_cols, len(train_clean)


def evaluate_on_validation(model, features: pd.DataFrame, feat_cols: list,
                           val_start: str) -> dict:
    """Evaluate model on validation period. Returns metrics + trade list."""
    val_data = features[features["date"] >= pd.Timestamp(val_start)].copy()
    val_data = val_data.dropna(subset=[V2_TARGET])

    if val_data.empty:
        return {"error": "No validation data"}

    # Predictions
    X_val = val_data[feat_cols].fillna(0).values
    val_data["pred"] = model.predict(X_val)

    # ── IC: rank correlation between predictions and actual returns ──
    from scipy.stats import spearmanr
    daily_ics = []
    dates = sorted(val_data["date"].unique())
    for d in dates:
        day = val_data[val_data["date"] == d]
        if len(day) < 10:
            continue
        ic, _ = spearmanr(day["pred"], day[V2_TARGET])
        daily_ics.append({"date": d, "ic": ic})

    ic_df = pd.DataFrame(daily_ics)
    mean_ic = ic_df["ic"].mean()
    ic_std = ic_df["ic"].std()
    ic_ir = mean_ic / ic_std if ic_std > 0 else 0
    hit_rate_ic = (ic_df["ic"] > 0).mean()

    # ── RMSE ──
    rmse = np.sqrt(((val_data["pred"] - val_data[V2_TARGET]) ** 2).mean())

    # ── TOP-K BACKTEST: rebalance every REBALANCE_EVERY days ──
    rebalance_dates = dates[::REBALANCE_EVERY]
    trades = []
    portfolio_returns = []

    ohlcv = pd.read_parquet("data/processed/ohlcv_smallcap.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    for i, reb_date in enumerate(rebalance_dates):
        day = val_data[val_data["date"] == reb_date].copy()
        if len(day) < TOP_K:
            continue

        # Rank by prediction, pick top K
        day = day.sort_values("pred", ascending=False)
        top_k = day.head(TOP_K)

        # Compute actual forward returns for each pick
        for _, row in top_k.iterrows():
            ticker = row["ticker"]
            entry_date = row["date"]
            fwd_ret = row.get(V2_RAW_COL, np.nan)
            fwd_ret_sr = row[V2_TARGET]

            # Get actual entry/exit prices
            ticker_ohlcv = ohlcv[(ohlcv["ticker"] == ticker) & (ohlcv["date"] >= entry_date)]
            entry_price = ticker_ohlcv.iloc[0]["close"] if len(ticker_ohlcv) > 0 else np.nan
            exit_row = ticker_ohlcv.iloc[min(HOLD_DAYS, len(ticker_ohlcv) - 1)] if len(ticker_ohlcv) > 1 else None
            exit_price = exit_row["close"] if exit_row is not None else np.nan
            exit_date = exit_row["date"] if exit_row is not None else entry_date

            # ATR trailing stop simulation
            vol = float(row.get("atr_pct_20d", 0.03))
            trail_pct = np.clip(vol * 5.3, 0.10, 0.16)  # adaptive

            # Simple trailing stop
            if len(ticker_ohlcv) > 1:
                prices = ticker_ohlcv.head(HOLD_DAYS + 1)["close"].values
                peak = prices[0]
                actual_ret = 0.0
                hit_stop = False
                for p_idx in range(1, len(prices)):
                    peak = max(peak, prices[p_idx])
                    drawdown = (prices[p_idx] - peak) / peak
                    if drawdown <= -trail_pct:
                        actual_ret = (prices[p_idx] - prices[0]) / prices[0]
                        exit_price = prices[p_idx]
                        exit_date = ticker_ohlcv.iloc[p_idx]["date"]
                        hit_stop = True
                        break
                if not hit_stop:
                    actual_ret = (prices[-1] - prices[0]) / prices[0]
                    exit_price = prices[-1]
                    exit_date = ticker_ohlcv.iloc[min(HOLD_DAYS, len(ticker_ohlcv) - 1)]["date"]
            else:
                actual_ret = 0.0
                hit_stop = False

            trades.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "ticker": ticker,
                "pred_score": float(row["pred"]),
                "fwd_ret_20d_raw": float(fwd_ret) if not np.isnan(fwd_ret) else None,
                "fwd_ret_20d_sr": float(fwd_ret_sr),
                "actual_ret": actual_ret,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "trail_pct": trail_pct,
                "hit_stop": hit_stop,
            })

        # Portfolio return for this rebalance period (equal weight)
        period_rets = [t["actual_ret"] for t in trades[-TOP_K:]]
        avg_ret = np.mean(period_rets) if period_rets else 0
        portfolio_returns.append({"date": reb_date, "return": avg_ret})

    trades_df = pd.DataFrame(trades)
    port_df = pd.DataFrame(portfolio_returns)

    # ── Aggregate metrics ──
    if not trades_df.empty:
        total_trades = len(trades_df)
        win_rate = (trades_df["actual_ret"] > 0).mean()
        avg_return = trades_df["actual_ret"].mean()
        median_return = trades_df["actual_ret"].median()
        total_return = (1 + port_df["return"]).prod() - 1 if not port_df.empty else 0

        # Sharpe (annualized from ~5-day rebalance periods)
        if not port_df.empty and port_df["return"].std() > 0:
            periods_per_year = 252 / REBALANCE_EVERY
            sharpe = (port_df["return"].mean() / port_df["return"].std()) * np.sqrt(periods_per_year)
        else:
            sharpe = 0

        # Max drawdown
        if not port_df.empty:
            cum_ret = (1 + port_df["return"]).cumprod()
            peak = cum_ret.cummax()
            dd = (cum_ret - peak) / peak
            max_dd = dd.min()
        else:
            max_dd = 0
    else:
        total_trades = win_rate = avg_return = median_return = 0
        total_return = sharpe = max_dd = 0

    return {
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "hit_rate_ic": hit_rate_ic,
        "rmse": rmse,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_return_per_trade": avg_return,
        "median_return_per_trade": median_return,
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_rebalance_dates": len(rebalance_dates),
        "val_days": len(dates),
        "trades_df": trades_df,
        "port_df": port_df,
        "ic_df": ic_df,
    }


def print_comparison(baseline: dict, full: dict):
    """Print side-by-side comparison."""
    print("\n" + "═" * 70)
    print("  COMPARISON: BASELINE (2yr) vs FULL (5yr + EDGAR + Meta)")
    print("═" * 70)
    print(f"\n  {'Metric':<30} {'BASELINE (2yr)':>15} {'FULL (5yr)':>15} {'Δ':>10}")
    print(f"  {'─' * 30} {'─' * 15} {'─' * 15} {'─' * 10}")

    metrics = [
        ("Mean IC", "mean_ic", "{:.4f}"),
        ("IC Std", "ic_std", "{:.4f}"),
        ("IC IR (IC/std)", "ic_ir", "{:.3f}"),
        ("IC Hit Rate (>0)", "hit_rate_ic", "{:.1%}"),
        ("RMSE", "rmse", "{:.4f}"),
        ("Total Trades", "total_trades", "{:.0f}"),
        ("Win Rate", "win_rate", "{:.1%}"),
        ("Avg Return/Trade", "avg_return_per_trade", "{:.2%}"),
        ("Median Return/Trade", "median_return_per_trade", "{:.2%}"),
        ("Total Return (Top-8)", "total_return", "{:.2%}"),
        ("Sharpe Ratio", "sharpe", "{:.3f}"),
        ("Max Drawdown", "max_drawdown", "{:.2%}"),
    ]

    for name, key, fmt in metrics:
        b_val = baseline.get(key, 0)
        f_val = full.get(key, 0)
        b_str = fmt.format(b_val)
        f_str = fmt.format(f_val)
        if isinstance(b_val, (int, float)) and isinstance(f_val, (int, float)):
            if "%" in fmt:
                delta = f_val - b_val
                d_str = f"{delta:+.2%}" if abs(delta) > 0.001 else "="
            else:
                delta = f_val - b_val
                d_str = f"{delta:+.4f}" if abs(delta) > 0.0001 else "="
        else:
            d_str = ""
        print(f"  {name:<30} {b_str:>15} {f_str:>15} {d_str:>10}")

    print()


def print_trades(trades_df: pd.DataFrame, title: str, max_show: int = 30):
    """Print trade details."""
    if trades_df.empty:
        print(f"\n  No trades for {title}")
        return

    print(f"\n{'─' * 70}")
    print(f"  TRADES — {title}")
    print(f"  ({len(trades_df)} total, showing top {min(max_show, len(trades_df))} by score)")
    print(f"{'─' * 70}")
    print(f"  {'Date':<12} {'Ticker':<7} {'Score':>7} {'Ret':>8} {'Entry':>7} {'Exit':>7} {'Stop':>5} {'Days':>5}")
    print(f"  {'─' * 12} {'─' * 7} {'─' * 7} {'─' * 8} {'─' * 7} {'─' * 7} {'─' * 5} {'─' * 5}")

    show = trades_df.sort_values("entry_date").head(max_show)
    for _, t in show.iterrows():
        entry_d = pd.Timestamp(t["entry_date"]).strftime("%Y-%m-%d")
        stop_flag = "✗" if t.get("hit_stop") else ""
        ret_str = f"{t['actual_ret']:+.1%}" if not np.isnan(t["actual_ret"]) else "N/A"
        entry_p = f"{t['entry_price']:.2f}" if not np.isnan(t["entry_price"]) else "N/A"
        exit_p = f"{t['exit_price']:.2f}" if not np.isnan(t["exit_price"]) else "N/A"
        days = (pd.Timestamp(t["exit_date"]) - pd.Timestamp(t["entry_date"])).days
        print(f"  {entry_d:<12} {t['ticker']:<7} {t['pred_score']:>7.3f} {ret_str:>8} "
              f"{entry_p:>7} {exit_p:>7} {stop_flag:>5} {days:>5}")

    # Summary
    winners = trades_df[trades_df["actual_ret"] > 0]
    losers = trades_df[trades_df["actual_ret"] <= 0]
    print(f"\n  Summary: {len(winners)} winners (avg +{winners['actual_ret'].mean():.1%}), "
          f"{len(losers)} losers (avg {losers['actual_ret'].mean():.1%})")
    print(f"  Best: {trades_df.loc[trades_df['actual_ret'].idxmax(), 'ticker']} "
          f"+{trades_df['actual_ret'].max():.1%} | "
          f"Worst: {trades_df.loc[trades_df['actual_ret'].idxmin(), 'ticker']} "
          f"{trades_df['actual_ret'].min():.1%}")


def verify_paper_trading():
    """Check paper trading is using the new model with EDGAR + Meta features."""
    import pickle

    model_path = Path("data/models/smallcap_v2_secrel20d.pkl")
    registry_path = Path("data/paper_trading/model_registry.json")

    print("\n" + "═" * 70)
    print("  PAPER TRADING MODEL VERIFICATION")
    print("═" * 70)

    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)  # noqa: S301
        feat_names = model.feature_name()
        n_trees = model.num_trees()

        has_edgar = [f for f in V2_EDGAR_FEATURES if f in feat_names]
        has_meta = [f for f in V2_META_FEATURES if f in feat_names]

        print(f"\n  Model file: {model_path}")
        print(f"  Trees: {n_trees}")
        print(f"  Features: {len(feat_names)} total")
        print(f"  EDGAR features: {has_edgar if has_edgar else '❌ MISSING'}")
        print(f"  Meta features: {has_meta if has_meta else '❌ MISSING'}")

        if has_edgar and has_meta:
            print(f"\n  ✅ Paper trading model has ALL 33 features (26 base + 2 EDGAR + 5 meta)")
        else:
            print(f"\n  ⚠️  Model is missing some features!")
    else:
        print(f"  ❌ Model not found at {model_path}")

    if registry_path.exists():
        import json
        with open(registry_path) as f:
            reg = json.load(f)
        print(f"\n  Registry:")
        print(f"    Last train: {reg.get('last_train_date', 'unknown')}")
        print(f"    N features: {reg.get('n_features', '?')}")
        print(f"    EDGAR features: {reg.get('n_edgar_features', '?')}")
        print(f"    Meta features: {reg.get('n_meta_features', '?')}")
        n_train = reg.get('n_train', '?')
        print(f"    Training rows: {n_train:,}" if isinstance(n_train, int) else f"    Training rows: {n_train}")


def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  SCAI Full Evaluation — 5yr Model vs 2yr Baseline                  ║")
    print("║  Validation: 2025-05-19 → 2026-05-18 (same for both)              ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # Load features
    print("\n[1/4] Loading features...")
    features = load_features()
    print(f"  Total: {len(features):,} rows, {features['date'].min().date()} → {features['date'].max().date()}")
    print(f"  Tickers: {features['ticker'].nunique()}")

    # Check data coverage
    pre_val = features[features["date"] < pd.Timestamp(VALIDATION_START)]
    print(f"  Pre-validation rows: {len(pre_val):,}")
    baseline_data = features[(features["date"] >= pd.Timestamp(TRAIN_START_BASELINE)) &
                             (features["date"] < pd.Timestamp(VALIDATION_START))]
    full_data = features[(features["date"] >= pd.Timestamp(TRAIN_START_FULL)) &
                         (features["date"] < pd.Timestamp(VALIDATION_START))]
    print(f"  Baseline training window: {TRAIN_START_BASELINE} → {VALIDATION_START} ({len(baseline_data):,} rows)")
    print(f"  Full training window:     {TRAIN_START_FULL} → {VALIDATION_START} ({len(full_data):,} rows)")

    # ── TRAIN BASELINE (2yr, no EDGAR, no Meta) ──
    print(f"\n[2/4] Training BASELINE model (2yr, 26 features, no EDGAR/Meta)...")
    model_base, feat_base, n_train_base = train_model(
        features, TRAIN_START_BASELINE, VALIDATION_START,
        use_edgar=False, use_meta=False
    )
    print(f"  ✓ Baseline: {n_train_base:,} rows, {len(feat_base)} features")

    # ── TRAIN FULL (5yr + EDGAR + Meta) ──
    print(f"\n[3/4] Training FULL model (5yr, 33 features: 26 base + 2 EDGAR + 5 meta)...")
    model_full, feat_full, n_train_full = train_model(
        features, TRAIN_START_FULL, VALIDATION_START,
        use_edgar=True, use_meta=True
    )
    print(f"  ✓ Full: {n_train_full:,} rows, {len(feat_full)} features")

    # ── EVALUATE BOTH ON SAME VALIDATION PERIOD ──
    print(f"\n[4/4] Evaluating on validation period ({VALIDATION_START} → latest)...")
    print("  Running baseline backtest...")
    results_base = evaluate_on_validation(model_base, features, feat_base, VALIDATION_START)
    print("  Running full model backtest...")
    results_full = evaluate_on_validation(model_full, features, feat_full, VALIDATION_START)

    # ── RESULTS ──
    print_comparison(results_base, results_full)

    # Trades
    print_trades(results_full.get("trades_df", pd.DataFrame()),
                 "FULL MODEL (5yr + EDGAR + Meta)")

    # Paper trading verification
    verify_paper_trading()

    print("\n" + "═" * 70)
    print("  DONE")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
