"""Rigorous comparison: WF with vs without meta features (all 16 folds).

Also reports train/test sizes per fold to verify split balance.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from _v3_harness import (
    V2_TARGET, V2_FEATURES_BASE, V2_EDGAR_FEATURES,
    TOP_K, HOLD_DAYS, REBALANCE_EVERY, N_COHORTS, define_folds,
)
from app.data.store.parquet_store import ParquetStore

# The leaky candidates we want to test
META_CANDIDATES = [
    'model_error_ticker_5', 'model_error_sector_20d', 'model_hit_rate_30d',
    'model_ic_rolling_20d', 'model_error_vol_20d',
]

N_BINS = 16
PARAMS = {
    "objective": "lambdarank", "metric": "ndcg",
    "num_leaves": 31, "max_depth": 6, "learning_rate": 0.05,
    "min_child_samples": 30, "subsample": 0.75, "colsample_bytree": 0.7,
    "reg_lambda": 5.0, "lambdarank_truncation_level": 8,
    "label_gain": list(range(N_BINS)),
    "n_jobs": 1, "seed": 42, "verbose": -1,
}


def run_one_fold(features, ohlcv, fold, feat_cols):
    train_mask = (features.date >= fold["train_start"]) & (features.date < fold["train_end"])
    train_data = features[train_mask].dropna(subset=[V2_TARGET]).copy()

    train_data["_rel"] = train_data.groupby("date")[V2_TARGET].transform(
        lambda s: pd.qcut(s.rank(method="first"), N_BINS, labels=False, duplicates="drop")
    )
    train_data["_rel"] = train_data["_rel"].fillna(0).astype(int).clip(0, N_BINS - 1)
    train_data = train_data.sort_values("date")

    X_tr = train_data[feat_cols].fillna(0).values
    y_tr = train_data["_rel"].values
    group = train_data.groupby("date").size().values

    ds = lgb.Dataset(X_tr, y_tr, group=group, feature_name=feat_cols, free_raw_data=True)
    model = lgb.train(PARAMS, ds, num_boost_round=600, callbacks=[lgb.log_evaluation(0)])

    test_mask = (features.date >= fold["test_start"]) & (features.date < fold["test_end"])
    test_data = features[test_mask].dropna(subset=[V2_TARGET]).copy()
    test_data["pred"] = model.predict(test_data[feat_cols].fillna(0).values)

    # IC
    daily_ics = []
    test_dates_sorted = sorted(test_data.date.unique())
    for d in test_dates_sorted:
        day = test_data[test_data.date == d]
        if len(day) >= 10:
            ic, _ = spearmanr(day["pred"], day[V2_TARGET])
            daily_ics.append(ic)
    mean_ic = float(np.mean(daily_ics)) if daily_ics else 0.0

    # Backtest with correct non-overlapping compounding
    rebalance_dates = test_dates_sorted[::REBALANCE_EVERY]
    port_returns = []
    trade_rets = []
    for reb_date in rebalance_dates:
        day = test_data[test_data.date == reb_date]
        if len(day) < TOP_K:
            continue
        top_k = day.sort_values("pred", ascending=False).head(TOP_K)
        period_rets = []
        for _, row in top_k.iterrows():
            t_oh = ohlcv[(ohlcv.ticker == row["ticker"]) & (ohlcv.date >= reb_date)]
            if len(t_oh) < 2:
                continue
            prices = t_oh.head(HOLD_DAYS + 1)["close"].values
            vol = float(row.get("atr_pct_20d", 0.03))
            trail_pct = np.clip(vol * 5.3, 0.10, 0.16)
            peak = prices[0]
            hit_stop = False
            for p_idx in range(1, len(prices)):
                peak = max(peak, prices[p_idx])
                if (prices[p_idx] - peak) / peak <= -trail_pct:
                    actual_ret = (prices[p_idx] - prices[0]) / prices[0]
                    hit_stop = True
                    break
            if not hit_stop:
                actual_ret = (prices[-1] - prices[0]) / prices[0]
            period_rets.append(actual_ret)
            trade_rets.append(actual_ret)
        if period_rets:
            port_returns.append(float(np.mean(period_rets)))

    # Non-overlapping streams
    if port_returns:
        streams = [port_returns[c::N_COHORTS] for c in range(N_COHORTS)]
        stream_cum = [float((1 + pd.Series(s)).prod()) for s in streams if s]
        total_ret = float(np.mean(stream_cum)) - 1 if stream_cum else 0.0
    else:
        total_ret = 0.0

    wr = float(np.mean([r > 0 for r in trade_rets])) if trade_rets else 0.0
    return {
        "train_rows": len(train_data),
        "test_rows": len(test_data),
        "train_dates": train_data.date.nunique(),
        "test_dates": test_data.date.nunique(),
        "ic": mean_ic,
        "ret": total_ret,
        "wr": wr,
        "n_trades": len(trade_rets),
    }


def main():
    store = ParquetStore()
    features = store.read("features_smallcap")
    ohlcv = store.read("ohlcv_smallcap")
    features["date"] = pd.to_datetime(features["date"])
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    # Which meta cols actually exist in the parquet?
    avail_meta = [c for c in META_CANDIDATES if c in features.columns]
    base_cols = V2_FEATURES_BASE + V2_EDGAR_FEATURES
    full_cols = base_cols + avail_meta

    folds = define_folds(features)
    print(f"\nTotal folds: {len(folds)}")
    print(f"Base features: {len(base_cols)}, Meta features: {len(avail_meta)}")
    print(f"\nFold split summary:")
    print(f"  {'Fold':>4s} {'Train period':<24s} {'Test period':<24s} {'Train rows':>12s} {'Test rows':>10s} {'%test':>6s}")
    for i, fold in enumerate(folds):
        # rough size estimate without running model
        tr_mask = (features.date >= fold["train_start"]) & (features.date < fold["train_end"])
        te_mask = (features.date >= fold["test_start"]) & (features.date < fold["test_end"])
        nt = features[tr_mask].dropna(subset=[V2_TARGET]).shape[0]
        ne = features[te_mask].dropna(subset=[V2_TARGET]).shape[0]
        pct = ne / (nt + ne) * 100
        print(f"  {i+1:>4d} {str(fold['train_start'].date())+' -> '+str(fold['train_end'].date()):<24s} "
              f"{str(fold['test_start'].date())+' -> '+str(fold['test_end'].date()):<24s} "
              f"{nt:>12,d} {ne:>10,d} {pct:>5.1f}%")

    print("\n" + "=" * 100)
    print("  WF COMPARISON: WITH META vs WITHOUT META (16 folds)")
    print("=" * 100)
    print(f"  {'Fold':>4s} {'Period':<22s} │ {'IC base':>8s} {'IC meta':>8s} {'Δ IC':>7s} │ "
          f"{'Ret base':>8s} {'Ret meta':>8s} {'Δ Ret':>7s} │ {'WR base':>7s} {'WR meta':>7s}")
    print(f"  {'─'*4} {'─'*22} │ {'─'*8} {'─'*8} {'─'*7} │ {'─'*8} {'─'*8} {'─'*7} │ {'─'*7} {'─'*7}")

    results_base, results_meta = [], []
    for i, fold in enumerate(folds):
        rb = run_one_fold(features, ohlcv, fold, base_cols)
        rm = run_one_fold(features, ohlcv, fold, full_cols)
        results_base.append(rb)
        results_meta.append(rm)
        period = f"{fold['test_start'].strftime('%Y-%m')}->{fold['test_end'].strftime('%Y-%m')}"
        d_ic = rm["ic"] - rb["ic"]
        d_ret = rm["ret"] - rb["ret"]
        print(f"  {i+1:>4d} {period:<22s} │ {rb['ic']:>+8.4f} {rm['ic']:>+8.4f} {d_ic:>+7.4f} │ "
              f"{rb['ret']:>+7.1%} {rm['ret']:>+7.1%} {d_ret:>+7.1%} │ "
              f"{rb['wr']:>6.1%} {rm['wr']:>6.1%}")

    # Aggregate
    print(f"  {'─'*4} {'─'*22} ──{'─'*8} {'─'*8} {'─'*7} ──{'─'*8} {'─'*8} {'─'*7} ──{'─'*7} {'─'*7}")
    mean_ic_b = np.mean([r["ic"] for r in results_base])
    mean_ic_m = np.mean([r["ic"] for r in results_meta])
    mean_ret_b = np.mean([r["ret"] for r in results_base])
    mean_ret_m = np.mean([r["ret"] for r in results_meta])
    wr_b = np.mean([r["wr"] for r in results_base])
    wr_m = np.mean([r["wr"] for r in results_meta])
    # Non-overlap compounded total return (mean of fold returns)
    total_b = float((1 + pd.Series([r["ret"] for r in results_base])).prod() - 1)
    total_m = float((1 + pd.Series([r["ret"] for r in results_meta])).prod() - 1)
    print(f"  {'MEAN':>4s} {'(across folds)':<22s} │ {mean_ic_b:>+8.4f} {mean_ic_m:>+8.4f} {mean_ic_m-mean_ic_b:>+7.4f} │ "
          f"{mean_ret_b:>+7.1%} {mean_ret_m:>+7.1%} {mean_ret_m-mean_ret_b:>+7.1%} │ "
          f"{wr_b:>6.1%} {wr_m:>6.1%}")
    print(f"  {'TOTAL':>4s} {'(compounded, non-ov.)':<22s} │ {'':>8s} {'':>8s} {'':>7s} │ "
          f"{total_b:>+7.1%} {total_m:>+7.1%} {total_m-total_b:>+7.1%} │ {'':>7s} {'':>7s}")
    n_better = sum(1 for b, m in zip(results_base, results_meta) if m["ic"] > b["ic"])
    print(f"\n  Folds where META has better IC: {n_better}/{len(folds)}")
    n_better_ret = sum(1 for b, m in zip(results_base, results_meta) if m["ret"] > b["ret"])
    print(f"  Folds where META has better Return: {n_better_ret}/{len(folds)}")


if __name__ == "__main__":
    main()
