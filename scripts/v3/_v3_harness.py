"""Shared V3 benchmark harness.

Runs the same 16-fold walk-forward used to baseline V2, with configurable
feature lists and LGB params, and returns deterministic metrics.

This is THE single source of truth for V3 measurements. Every V3 change
must call run_walkforward() with the same fold definition.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import lightgbm as lgb

V2_TARGET = "fwd_ret_20d_sector_rel"
V2_RAW_COL = "fwd_ret_20d"

V2_FEATURES_BASE = [
    'max_dd_60d', 'vol_of_vol_60d', 'ret_kurtosis_60d', 'avg_trade_size_20d',
    'obv', 'obv_vs_sma_60d', 'amihud_60d', 'downside_vol_60d', 'ema_26',
    'spread_proxy_20d', 'ret_skew_60d', 'ret_252d', 'sma_200', 'sector_ret_60d',
    'realized_vol_120d', 'macd_hist', 'pct_from_52w_low', 'adv_60d',
    'ret_vs_sector_60d', 'vol_of_vol_ratio', 'price_roc_smooth_120d',
    'vwap_dev_avg_20d', 'reversal_20v60', 'vol_regime_change', 'beta_60d',
    'macd_signal',
]
V2_EDGAR_FEATURES = ['dilution_pct', 'current_ratio']
# A/B test 2026-05-22 (leak-free): meta-features are NEUTRAL/NEGATIVE.
# Original "+10.75pp validation gain" was caused by a leak in compute_meta_features:
# filter `signal_date < current_date` allowed signals whose 20d outcome reached
# into the future (price T..T+20 leaked back as feature at T).
# Fixed filter is `signal_date < current_date - 30d`, after which meta-features
# show no improvement (IC 0.005 vs 0.012, return 20% vs 26%, WR equal).
# Removed from production. See `_ab_meta.py` for the test.
V2_META_FEATURES: list[str] = []

V2_LGB_PARAMS = {
    "objective": "regression", "metric": "rmse",
    "num_leaves": 20, "max_depth": 5,
    "learning_rate": 0.02, "min_child_samples": 60,
    "subsample": 0.75, "colsample_bytree": 0.7,
    "reg_alpha": 0.5, "reg_lambda": 5.0,
    "n_jobs": 1, "seed": 42, "verbose": -1,
}

TOP_K = 8
HOLD_DAYS = 20
REBALANCE_EVERY = 5
N_COHORTS = HOLD_DAYS // REBALANCE_EVERY  # overlapping cohorts (4)
MIN_TRAIN_END = pd.Timestamp("2022-06-01")
FOLD_DAYS = 63


@dataclass
class FoldMetrics:
    fold: int
    period: str
    train_rows: int
    test_rows: int
    mean_ic: float
    ic_ir: float          # IC / std(IC)
    hit_rate_ic: float    # % of days with IC > 0
    total_return: float
    sharpe: float
    max_dd: float
    n_trades: int
    win_rate: float
    median_return: float  # selectivity: median actual return of top-K
    market_return: float  # median return of all stocks in this fold


@dataclass
class RunResult:
    config_name: str
    feat_cols: list
    n_features: int
    folds: list[FoldMetrics] = field(default_factory=list)

    def aggregate(self) -> dict:
        if not self.folds:
            return {}
        ics = [f.mean_ic for f in self.folds]
        sharpes = [f.sharpe for f in self.folds]
        rets = [f.total_return for f in self.folds]
        wrs = [f.win_rate for f in self.folds]
        meds = [f.median_return for f in self.folds]
        alphas = [f.total_return - f.market_return for f in self.folds]
        n_pos_ret = sum(1 for r in rets if r > 0)
        n_pos_alpha = sum(1 for a in alphas if a > 0)
        total_trades = sum(f.n_trades for f in self.folds)
        all_wins = sum(f.win_rate * f.n_trades for f in self.folds) / total_trades if total_trades else 0
        return {
            "n_features": self.n_features,
            "n_folds": len(self.folds),
            "mean_ic": float(np.mean(ics)),
            "ic_std": float(np.std(ics)),
            "mean_sharpe": float(np.mean(sharpes)),
            "mean_return": float(np.mean(rets)),
            "median_return": float(np.median(rets)),
            "mean_alpha": float(np.mean(alphas)),
            "mean_win_rate": float(all_wins),
            "selectivity_median": float(np.mean(meds)),
            "folds_positive_ret": f"{n_pos_ret}/{len(self.folds)}",
            "folds_positive_alpha": f"{n_pos_alpha}/{len(self.folds)}",
            "total_trades": int(total_trades),
        }


def define_folds(features: pd.DataFrame) -> list[dict]:
    """Same fold structure used in V2 walk-forward baseline."""
    all_dates = sorted(features["date"].unique())
    data_start = all_dates[0]
    data_end = all_dates[-1]
    folds = []
    test_start = MIN_TRAIN_END
    while test_start < data_end - pd.Timedelta(days=30):
        test_end = min(test_start + pd.Timedelta(days=FOLD_DAYS * 1.5), data_end)
        test_dates = [d for d in all_dates if test_start <= d < test_end]
        if len(test_dates) < 10:
            break
        folds.append({
            "train_start": data_start, "train_end": test_start,
            "test_start": test_start, "test_end": test_end,
            "test_dates": test_dates,
        })
        test_start = test_end
    return folds


def run_walkforward(
    features: pd.DataFrame,
    ohlcv: pd.DataFrame,
    feat_cols: list,
    config_name: str = "v2_baseline",
    lgb_params: dict | None = None,
    objective_lambdarank: bool = False,
    verbose: bool = True,
) -> RunResult:
    """Run 16-fold walk-forward; return per-fold + aggregate metrics."""
    params = dict(lgb_params or V2_LGB_PARAMS)
    folds = define_folds(features)

    n_bins = 16
    if objective_lambdarank:
        params = dict(params)
        params["objective"] = "lambdarank"
        params["metric"] = "ndcg"
        params["lambdarank_truncation_level"] = TOP_K
        params["label_gain"] = list(range(n_bins))
        params.pop("reg_alpha", None)  # keep simple

    result = RunResult(config_name=config_name, feat_cols=feat_cols, n_features=len(feat_cols))

    for i, fold in enumerate(folds):
        train_mask = (features.date >= fold["train_start"]) & (features.date < fold["train_end"])
        train_data = features[train_mask].dropna(subset=[V2_TARGET]).copy()
        X_tr = train_data[feat_cols].fillna(0).values
        y_tr = train_data[V2_TARGET].values

        if objective_lambdarank:
            # Convert continuous target to relevance levels 0..n_bins-1 per date
            train_data["_rel"] = train_data.groupby("date")[V2_TARGET].transform(
                lambda s: pd.qcut(s.rank(method="first"), n_bins, labels=False, duplicates="drop")
            )
            train_data["_rel"] = train_data["_rel"].fillna(0).astype(int).clip(0, n_bins - 1)
            # drop groups with no rows
            train_data = train_data.sort_values("date")
            X_tr = train_data[feat_cols].fillna(0).values
            y_tr = train_data["_rel"].values
            group = train_data.groupby("date").size().values
            ds = lgb.Dataset(X_tr, y_tr, group=group, feature_name=feat_cols, free_raw_data=True)
        else:
            ds = lgb.Dataset(X_tr, y_tr, feature_name=feat_cols, free_raw_data=True)

        model = lgb.train(params, ds, num_boost_round=400,
                          callbacks=[lgb.log_evaluation(0)])

        test_mask = (features.date >= fold["test_start"]) & (features.date < fold["test_end"])
        test_data = features[test_mask].dropna(subset=[V2_TARGET]).copy()
        if test_data.empty:
            continue
        X_te = test_data[feat_cols].fillna(0).values
        test_data["pred"] = model.predict(X_te)

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

        # Trades with trailing stop
        rebalance_dates = test_dates_sorted[::REBALANCE_EVERY]
        trades = []
        port_returns = []
        topk_returns = []  # raw fwd returns for selectivity
        for reb_date in rebalance_dates:
            day = test_data[test_data.date == reb_date].copy()
            if len(day) < TOP_K:
                continue
            top_k = day.sort_values("pred", ascending=False).head(TOP_K)
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
                    drawdown = (prices[p_idx] - peak) / peak
                    if drawdown <= -trail_pct:
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
            # Fix: compound non-overlapping streams, not overlapping 20d returns
            # With HOLD_DAYS=20 and REBALANCE_EVERY=5, 4 cohorts run concurrently.
            # Split into N_COHORTS interleaved streams, each non-overlapping.
            streams = [port_returns[c::N_COHORTS] for c in range(N_COHORTS)]
            stream_cum = [float((1 + pd.Series(s)).prod()) for s in streams if s]
            total_ret = float(np.mean(stream_cum)) - 1 if stream_cum else 0.0
            # Sharpe: each period's portfolio contribution = ret / N_COHORTS
            eff = pd.Series([r / N_COHORTS for r in port_returns])
            sharpe = float((eff.mean() / eff.std()) * np.sqrt(252 / REBALANCE_EVERY)) if eff.std() > 0 else 0.0
            cum = (1 + eff).cumprod()
            max_dd = float(((cum - cum.cummax()) / cum.cummax()).min())
        else:
            total_ret = sharpe = max_dd = 0.0

        n_trades = len(trades)
        win_rate = float(np.mean([t["actual_ret"] > 0 for t in trades])) if trades else 0.0
        med_ret = float(np.median(topk_returns)) if topk_returns else 0.0

        # Market return for this fold (median of all available stocks)
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

        if verbose:
            alpha = total_ret - market_ret
            print(f"  Fold {i+1:2d} {fm.period}: IC={mean_ic:+.4f} Ret={total_ret:+7.1%} "
                  f"Sharpe={sharpe:+5.2f} WR={win_rate:.0%} α={alpha:+7.1%} "
                  f"selMed={med_ret:+.2%}")

    return result


def print_comparison(baseline: RunResult, *others: RunResult) -> None:
    """Print side-by-side comparison vs baseline."""
    rows = [baseline] + list(others)
    print("\n" + "=" * 100)
    print(f"  COMPARISON  ({len(rows)} configs)")
    print("=" * 100)

    keys = ["n_features", "mean_ic", "ic_std", "mean_sharpe", "mean_return", "median_return",
            "mean_alpha", "mean_win_rate", "selectivity_median",
            "folds_positive_ret", "folds_positive_alpha", "total_trades"]

    aggs = [r.aggregate() for r in rows]
    col_w = 16
    name_w = 24
    print(f"  {'Metric':<{name_w}}" + "".join(f"{r.config_name:>{col_w}}" for r in rows))
    print(f"  {'─' * name_w}" + "".join(f" {'─' * (col_w - 1)}" for _ in rows))
    for k in keys:
        line = f"  {k:<{name_w}}"
        for a in aggs:
            v = a.get(k, "")
            if isinstance(v, float):
                if "rate" in k or "alpha" in k or "return" in k or "selectivity" in k:
                    line += f"{v:>{col_w}.2%}"
                else:
                    line += f"{v:>{col_w}.4f}"
            else:
                line += f"{str(v):>{col_w}}"
        print(line)


def save_result(result: RunResult, out_dir: str = "data/v3_benchmarks") -> Path:
    """Save full result to JSON."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fp = out_path / f"{result.config_name}.json"
    payload = {
        "config_name": result.config_name,
        "n_features": result.n_features,
        "feat_cols": result.feat_cols,
        "aggregate": result.aggregate(),
        "folds": [asdict(f) for f in result.folds],
    }
    fp.write_text(json.dumps(payload, indent=2, default=str))
    return fp
