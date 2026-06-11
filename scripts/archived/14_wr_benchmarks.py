"""Sprint 3 — Win Rate improvement benchmarks.

Tests 4 strategies to raise WR from ~50% towards 60%+:
  1. score_gate_8:   TOP_K=8 but positions 5-8 only if score > threshold
  2. score_gate_4:   Always top-4, positions 5-8 gated by score
  3. adaptive_stop:  Tighten trail after day 5 if profitable
  4. combo:          score_gate_8 + adaptive_stop

All measured against v3_hp_combo baseline on the same 16-fold WF.

Usage:
    DYLD_LIBRARY_PATH=.local/lib PYTHONPATH=src python scripts/v3/14_wr_benchmarks.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _v3_harness import (
    V2_TARGET, V2_RAW_COL, V2_FEATURES_BASE, V2_EDGAR_FEATURES,
    V2_META_FEATURES, TOP_K, HOLD_DAYS, REBALANCE_EVERY,
    MIN_TRAIN_END, FOLD_DAYS,
    FoldMetrics, RunResult, define_folds, print_comparison,
)


# V3 production params (v3_hp_combo)
V3_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_child_samples": 30,
    "subsample": 0.75,
    "colsample_bytree": 0.7,
    "reg_lambda": 5.0,
    "lambdarank_truncation_level": 8,
    "label_gain": list(range(16)),
    "n_jobs": 1,
    "seed": 42,
    "verbose": -1,
}
N_BINS = 16
N_ROUNDS = 600


def _run_wr_benchmark(
    features: pd.DataFrame,
    ohlcv: pd.DataFrame,
    feat_cols: list[str],
    config_name: str,
    score_gate: float | None = None,  # min score for positions 5-8
    adaptive_stop: bool = False,      # tighten trail after day 5 if profitable
    min_picks: int = 4,               # min positions to take (always take top min_picks)
) -> RunResult:
    """Modified walk-forward with WR improvement strategies."""
    params = dict(V3_PARAMS)
    folds = define_folds(features)
    result = RunResult(config_name=config_name, feat_cols=feat_cols, n_features=len(feat_cols))

    for i, fold in enumerate(folds):
        train_mask = (features.date >= fold["train_start"]) & (features.date < fold["train_end"])
        train_data = features[train_mask].dropna(subset=[V2_TARGET]).copy()

        # LambdaRank: relevance bins per date
        train_data["_rel"] = train_data.groupby("date")[V2_TARGET].transform(
            lambda s: pd.qcut(s.rank(method="first"), N_BINS, labels=False, duplicates="drop")
        )
        train_data["_rel"] = train_data["_rel"].fillna(0).astype(int).clip(0, N_BINS - 1)
        train_data = train_data.sort_values("date")

        X_tr = train_data[feat_cols].fillna(0).values
        y_tr = train_data["_rel"].values
        group = train_data.groupby("date").size().values

        ds = lgb.Dataset(X_tr, y_tr, group=group, feature_name=feat_cols, free_raw_data=True)
        model = lgb.train(params, ds, num_boost_round=N_ROUNDS,
                          callbacks=[lgb.log_evaluation(0)])

        test_mask = (features.date >= fold["test_start"]) & (features.date < fold["test_end"])
        test_data = features[test_mask].dropna(subset=[V2_TARGET]).copy()
        if test_data.empty:
            continue
        test_data["pred"] = model.predict(test_data[feat_cols].fillna(0).values)

        # IC
        daily_ics = []
        test_dates_sorted = sorted(test_data.date.unique())
        for d in test_dates_sorted:
            day = test_data[test_data.date == d]
            if len(day) < 10:
                continue
            ic, _ = spearmanr(day["pred"], day[V2_TARGET])
            daily_ics.append(ic)
        mean_ic = float(np.mean(daily_ics)) if daily_ics else 0.0
        ic_std = float(np.std(daily_ics)) if daily_ics else 0.0
        ic_ir = mean_ic / ic_std if ic_std > 0 else 0.0
        hit_rate_ic = float(np.mean([1 for ic in daily_ics if ic > 0]) / len(daily_ics)) if daily_ics else 0.0

        # Trades with strategies
        rebalance_dates = test_dates_sorted[::REBALANCE_EVERY]
        trades = []
        port_returns = []
        topk_returns = []

        for reb_date in rebalance_dates:
            day = test_data[test_data.date == reb_date].copy()
            if len(day) < TOP_K:
                continue

            ranked = day.sort_values("pred", ascending=False)

            # Score-gate logic: always take top min_picks, gate the rest
            if score_gate is not None:
                top_core = ranked.head(min_picks)
                candidates = ranked.iloc[min_picks:TOP_K]
                # Compute threshold: score_gate is a z-score threshold
                day_mean = day["pred"].mean()
                day_std = day["pred"].std()
                threshold = day_mean + score_gate * day_std if day_std > 0 else day_mean
                gated = candidates[candidates["pred"] > threshold]
                top_k = pd.concat([top_core, gated])
            else:
                top_k = ranked.head(TOP_K)

            period_rets = []
            for _, row in top_k.iterrows():
                ticker = row["ticker"]
                t_oh = ohlcv[(ohlcv.ticker == ticker) & (ohlcv.date >= reb_date)]
                if len(t_oh) < 2:
                    continue
                prices = t_oh.head(HOLD_DAYS + 1)["close"].values
                vol = float(row.get("atr_pct_20d", 0.03))
                trail_pct = np.clip(vol * 5.3, 0.10, 0.16)
                peak = prices[0]
                actual_ret = 0.0
                hit_stop = False
                for p_idx in range(1, len(prices)):
                    peak = max(peak, prices[p_idx])

                    # Adaptive stop: tighten after day 5 if profitable
                    current_trail = trail_pct
                    if adaptive_stop and p_idx > 5:
                        if prices[p_idx] > prices[0]:  # profitable
                            current_trail = min(trail_pct, 0.06)

                    drawdown = (prices[p_idx] - peak) / peak
                    if drawdown <= -current_trail:
                        actual_ret = (prices[p_idx] - prices[0]) / prices[0]
                        hit_stop = True
                        break
                if not hit_stop:
                    actual_ret = (prices[-1] - prices[0]) / prices[0]
                trades.append({"actual_ret": actual_ret})
                period_rets.append(actual_ret)
                topk_returns.append(actual_ret)
            if period_rets:
                port_returns.append(float(np.mean(period_rets)))

        port = pd.Series(port_returns)
        if not port.empty:
            # Fix: compound non-overlapping streams (4 cohorts)
            n_cohorts = HOLD_DAYS // REBALANCE_EVERY
            streams = [port_returns[c::n_cohorts] for c in range(n_cohorts)]
            stream_cum = [float((1 + pd.Series(s)).prod()) for s in streams if s]
            total_ret = float(np.mean(stream_cum)) - 1 if stream_cum else 0.0
            eff = pd.Series([r / n_cohorts for r in port_returns])
            sharpe = float((eff.mean() / eff.std()) * np.sqrt(252 / REBALANCE_EVERY)) if eff.std() > 0 else 0.0
            cum = (1 + eff).cumprod()
            max_dd = float(((cum - cum.cummax()) / cum.cummax()).min())
        else:
            total_ret = sharpe = max_dd = 0.0

        n_trades = len(trades)
        win_rate = float(np.mean([t["actual_ret"] > 0 for t in trades])) if trades else 0.0
        med_ret = float(np.median(topk_returns)) if topk_returns else 0.0

        # Market return
        fold_oh = ohlcv[(ohlcv.date >= fold["test_start"]) & (ohlcv.date < fold["test_end"])]
        if not fold_oh.empty:
            sp = fold_oh.drop_duplicates("ticker", keep="first").set_index("ticker")["close"]
            ep = fold_oh.drop_duplicates("ticker", keep="last").set_index("ticker")["close"]
            common = sp.index.intersection(ep.index)
            market_ret = float(((ep[common] / sp[common]) - 1).median()) if len(common) > 10 else 0.0
        else:
            market_ret = 0.0

        fm = FoldMetrics(
            fold=i + 1,
            period=f"{fold['test_start'].strftime('%Y-%m')}->{fold['test_end'].strftime('%Y-%m')}",
            train_rows=len(train_data), test_rows=len(test_data),
            mean_ic=mean_ic, ic_ir=ic_ir, hit_rate_ic=hit_rate_ic,
            total_return=total_ret, sharpe=sharpe, max_dd=max_dd,
            n_trades=n_trades, win_rate=win_rate,
            median_return=med_ret, market_return=market_ret,
        )
        result.folds.append(fm)
        alpha = total_ret - market_ret
        print(f"  Fold {i+1:2d} {fm.period}: IC={mean_ic:+.4f} Ret={total_ret:+7.1%} "
              f"Sharpe={sharpe:+5.2f} WR={win_rate:.0%} α={alpha:+7.1%} "
              f"selMed={med_ret:+.2%} trades={n_trades}")

    return result


def main():
    from app.data.store.parquet_store import ParquetStore
    store = ParquetStore()
    features = store.read("features_smallcap")
    ohlcv = store.read("ohlcv_smallcap")
    features["date"] = pd.to_datetime(features["date"])
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    feat_cols = [c for c in V2_FEATURES_BASE + V2_EDGAR_FEATURES + V2_META_FEATURES
                 if c in features.columns]
    print(f"Features: {len(feat_cols)}")
    print(f"Data: {len(features):,} rows, {features.date.nunique()} dates\n")

    # ── 0. Baseline: v3_hp_combo (original, no changes) ──
    print("=" * 70)
    print("  [0] BASELINE — v3_hp_combo (TOP_K=8, no gate, standard trail)")
    print("=" * 70)
    baseline = _run_wr_benchmark(features, ohlcv, feat_cols, "baseline_8")

    # ── 1. Score gate: top-4 always, positions 5-8 if score > 0.5σ ──
    print("\n" + "=" * 70)
    print("  [1] SCORE_GATE_0.5 — Top 4 always + 5-8 if score > mean+0.5σ")
    print("=" * 70)
    sg05 = _run_wr_benchmark(features, ohlcv, feat_cols, "gate_0.5σ",
                             score_gate=0.5, min_picks=4)

    # ── 2. Score gate stricter: top-4 always, 5-8 if score > 1.0σ ──
    print("\n" + "=" * 70)
    print("  [2] SCORE_GATE_1.0 — Top 4 always + 5-8 if score > mean+1.0σ")
    print("=" * 70)
    sg10 = _run_wr_benchmark(features, ohlcv, feat_cols, "gate_1.0σ",
                             score_gate=1.0, min_picks=4)

    # ── 3. Adaptive stop: tighten trail to 6% after day 5 if profitable ──
    print("\n" + "=" * 70)
    print("  [3] ADAPTIVE_STOP — Trail tightens to 6% after day 5 if in profit")
    print("=" * 70)
    astop = _run_wr_benchmark(features, ohlcv, feat_cols, "adapt_stop",
                              adaptive_stop=True)

    # ── 4. Combo: score_gate 0.5σ + adaptive stop ──
    print("\n" + "=" * 70)
    print("  [4] COMBO — gate_0.5σ + adaptive stop")
    print("=" * 70)
    combo = _run_wr_benchmark(features, ohlcv, feat_cols, "combo",
                              score_gate=0.5, min_picks=4, adaptive_stop=True)

    # ── 5. Score gate: top-4 always, 5-8 if score > 1.5σ (most selective) ──
    print("\n" + "=" * 70)
    print("  [5] SCORE_GATE_1.5 — Top 4 always + 5-8 only if score > mean+1.5σ")
    print("=" * 70)
    sg15 = _run_wr_benchmark(features, ohlcv, feat_cols, "gate_1.5σ",
                             score_gate=1.5, min_picks=4)

    # ── Comparison ──
    print_comparison(baseline, sg05, sg10, sg15, astop, combo)

    # Save results
    out = Path("data/v3_benchmarks")
    out.mkdir(parents=True, exist_ok=True)
    for r in [baseline, sg05, sg10, sg15, astop, combo]:
        data = {
            "config_name": r.config_name,
            "n_features": r.n_features,
            "folds": [
                {
                    "fold": f.fold, "period": f.period,
                    "train_rows": f.train_rows, "test_rows": f.test_rows,
                    "mean_ic": f.mean_ic, "sharpe": f.sharpe,
                    "total_return": f.total_return, "win_rate": f.win_rate,
                    "median_return": f.median_return, "n_trades": f.n_trades,
                    "market_return": f.market_return,
                }
                for f in r.folds
            ],
            "aggregate": r.aggregate(),
        }
        fp = out / f"v3_wr_{r.config_name}.json"
        fp.write_text(json.dumps(data, indent=2))
        print(f"  Saved {fp}")


if __name__ == "__main__":
    main()
