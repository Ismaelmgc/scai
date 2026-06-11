"""Shared V3/V4 benchmark harness.

Runs the same 16-fold walk-forward used to baseline V2, with configurable
feature lists and LGB params, and returns deterministic metrics.

This is THE single source of truth for V3/V4 measurements. Every change
must call run_walkforward() with the same fold definition.

V4 additions (2026-06-11):
- Tradability filter at selection time (min_price / min_adv_usd) via
  app.features.tradability.tradable_mask — training rows are NEVER filtered
  (anti-survivorship). Old unfiltered benchmarks selected untradeable
  sub-penny names; their metrics are not comparable to filtered runs.
- Tradable-only IC (mean_ic_tradable) — the deployable number; promotion
  decisions use this, full-cross-section mean_ic kept for continuity.
- ExitPolicy: parameterized trailing/adaptive/profit-target/breakeven/time
  stops so exit sweeps don't require retraining.
- Per-fold prediction caching (cache_dir) + replay_walkforward(): train 16
  models once, then re-evaluate filter/exit/cost variants in seconds.
- Round-trip transaction cost (cost_bps per side; optional spread-aware
  floor via cs_spread_20d). Old benchmarks were zero-cost.

The features DataFrame passed in must contain, besides feat_cols and the
target: date, ticker, atr_pct_20d, close, adv_usd_20d (and cs_spread_20d
if use_spread_cost).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from app.features.tradability import tradable_mask  # noqa: E402

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

# Columns persisted to the prediction cache (besides date/ticker/pred/target).
_CACHE_AUX_COLS = ["atr_pct_20d", "close", "adv_usd_20d", "cs_spread_20d"]


@dataclass
class ExitPolicy:
    """Exit rules applied to each position in the simulation.

    Defaults reproduce the historical harness behaviour (ATR trailing stop
    clipped [10%, 16%], exit at HOLD_DAYS).
    """
    trail_mult: float = 5.3
    trail_min: float = 0.10
    trail_max: float = 0.16
    adaptive_tighten: float | None = None   # e.g. 0.06 → live "adaptive stop"
    adaptive_after_days: int = 5
    profit_target: float | None = None      # e.g. 0.25 → sell at +25%
    breakeven_after: float | None = None    # e.g. 0.08 → stop to entry after +8% peak
    time_stop: int = HOLD_DAYS

    def describe(self) -> str:
        parts = [f"trail[{self.trail_min:.0%},{self.trail_max:.0%}]"]
        if self.adaptive_tighten:
            parts.append(f"adapt{self.adaptive_tighten:.0%}@d{self.adaptive_after_days}")
        if self.profit_target:
            parts.append(f"pt{self.profit_target:.0%}")
        if self.breakeven_after:
            parts.append(f"be@{self.breakeven_after:.0%}")
        if self.time_stop != HOLD_DAYS:
            parts.append(f"ts{self.time_stop}")
        return "+".join(parts)


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
    mean_ic_tradable: float = 0.0   # IC on the tradable cross-section only
    n_skipped_rebalances: int = 0   # rebalances with < TOP_K tradable names
    avg_candidates: float = 0.0     # avg tradable names per rebalance


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
        ics_tr = [f.mean_ic_tradable for f in self.folds]
        sharpes = [f.sharpe for f in self.folds]
        rets = [f.total_return for f in self.folds]
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
            "mean_ic_tradable": float(np.mean(ics_tr)),
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
            "skipped_rebalances": int(sum(f.n_skipped_rebalances for f in self.folds)),
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


def _simulate_trade(prices: np.ndarray, atr_pct: float,
                    policy: ExitPolicy) -> float:
    """Simulate one position over daily closes; return gross return."""
    trail_pct = float(np.clip(atr_pct * policy.trail_mult,
                              policy.trail_min, policy.trail_max))
    entry = prices[0]
    peak = entry
    n = min(len(prices) - 1, policy.time_stop)
    for p_idx in range(1, n + 1):
        price = prices[p_idx]
        peak = max(peak, price)
        # Profit target: exit immediately at target level
        if policy.profit_target and price >= entry * (1 + policy.profit_target):
            return (price - entry) / entry
        eff_trail = trail_pct
        # Adaptive tighten (live "adaptive stop" engine behaviour)
        if (policy.adaptive_tighten and p_idx > policy.adaptive_after_days
                and price > entry):
            eff_trail = min(eff_trail, policy.adaptive_tighten)
        # Trailing stop from peak
        if (price - peak) / peak <= -eff_trail:
            return (price - entry) / entry
        # Breakeven stop: once peak cleared +X%, never give back below entry
        if (policy.breakeven_after and peak >= entry * (1 + policy.breakeven_after)
                and price <= entry):
            return (price - entry) / entry
    return (prices[n] - entry) / entry


def _evaluate_fold(
    test_data: pd.DataFrame,
    ohlcv: pd.DataFrame,
    fold_bounds: tuple[pd.Timestamp, pd.Timestamp],
    min_price: float | None,
    min_adv_usd: float | None,
    policy: ExitPolicy,
    cost_bps: float,
    use_spread_cost: bool,
) -> dict:
    """Trade simulation + tradable IC for one fold's predictions.

    Shared by run_walkforward (fresh predictions) and replay_walkforward
    (cached predictions). Filtering applies to SELECTION only.
    """
    filtering = min_price is not None or min_adv_usd is not None
    test_dates_sorted = sorted(test_data.date.unique())

    # Tradable-only IC (the deployable number)
    tr_ics = []
    if filtering:
        masked = test_data[tradable_mask(test_data, min_price or 0.0, min_adv_usd or 0.0)]
    else:
        masked = test_data
    for d in test_dates_sorted:
        day = masked[masked.date == d]
        if len(day) < 10:
            continue
        ic, _ = spearmanr(day["pred"], day[V2_TARGET])
        tr_ics.append(ic)
    mean_ic_tradable = float(np.mean(tr_ics)) if tr_ics else 0.0

    rebalance_dates = test_dates_sorted[::REBALANCE_EVERY]
    trades = []
    port_returns = []
    topk_returns = []
    n_skipped = 0
    candidate_counts = []
    for reb_date in rebalance_dates:
        day = test_data[test_data.date == reb_date]
        if filtering:
            day = day[tradable_mask(day, min_price or 0.0, min_adv_usd or 0.0)]
        candidate_counts.append(len(day))
        if len(day) < TOP_K:
            n_skipped += 1
            continue
        top_k = day.sort_values("pred", ascending=False).head(TOP_K)
        period_rets = []
        for _, row in top_k.iterrows():
            ticker = row["ticker"]
            t_oh = ohlcv[(ohlcv.ticker == ticker) & (ohlcv.date >= reb_date)]
            if len(t_oh) < 2:
                continue
            prices = t_oh.head(HOLD_DAYS + 1)["close"].values
            atr_pct = float(row.get("atr_pct_20d", 0.03))
            if not np.isfinite(atr_pct) or atr_pct <= 0:
                atr_pct = 0.03
            ret = _simulate_trade(prices, atr_pct, policy)
            # Round-trip transaction cost
            cost_rt = 2 * cost_bps / 1e4
            if use_spread_cost:
                spread = row.get("cs_spread_20d", np.nan)
                if np.isfinite(spread) and spread > 0:
                    cost_rt = max(cost_rt, min(float(spread), 0.10))
            ret -= cost_rt
            trades.append(ret)
            period_rets.append(ret)
            topk_returns.append(ret)
        if period_rets:
            port_returns.append(float(np.mean(period_rets)))

    if port_returns:
        # Compound non-overlapping streams, not overlapping 20d returns.
        # With HOLD_DAYS=20 and REBALANCE_EVERY=5, 4 cohorts run concurrently.
        streams = [port_returns[c::N_COHORTS] for c in range(N_COHORTS)]
        stream_cum = [float((1 + pd.Series(s)).prod()) for s in streams if s]
        total_ret = float(np.mean(stream_cum)) - 1 if stream_cum else 0.0
        eff = pd.Series([r / N_COHORTS for r in port_returns])
        sharpe = float((eff.mean() / eff.std()) * np.sqrt(252 / REBALANCE_EVERY)) if eff.std() > 0 else 0.0
        cum = (1 + eff).cumprod()
        max_dd = float(((cum - cum.cummax()) / cum.cummax()).min())
    else:
        total_ret = sharpe = max_dd = 0.0

    # Market return for this fold (median of all available stocks)
    t0, t1 = fold_bounds
    fold_oh = ohlcv[(ohlcv.date >= t0) & (ohlcv.date < t1)]
    if not fold_oh.empty:
        sp = fold_oh.drop_duplicates("ticker", keep="first").set_index("ticker")["close"]
        ep = fold_oh.drop_duplicates("ticker", keep="last").set_index("ticker")["close"]
        common = sp.index.intersection(ep.index)
        market_ret = float(((ep[common] / sp[common]) - 1).median()) if len(common) > 10 else 0.0
    else:
        market_ret = 0.0

    return {
        "mean_ic_tradable": mean_ic_tradable,
        "total_return": total_ret,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "n_trades": len(trades),
        "win_rate": float(np.mean([t > 0 for t in trades])) if trades else 0.0,
        "median_return": float(np.median(topk_returns)) if topk_returns else 0.0,
        "market_return": market_ret,
        "n_skipped_rebalances": n_skipped,
        "avg_candidates": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
    }


def run_walkforward(
    features: pd.DataFrame,
    ohlcv: pd.DataFrame,
    feat_cols: list,
    config_name: str = "v2_baseline",
    lgb_params: dict | None = None,
    objective_lambdarank: bool = False,
    verbose: bool = True,
    min_price: float | None = None,
    min_adv_usd: float | None = None,
    exit_policy: ExitPolicy | None = None,
    cost_bps: float = 0.0,
    use_spread_cost: bool = False,
    cache_dir: str | Path | None = None,
) -> RunResult:
    """Run 16-fold walk-forward; return per-fold + aggregate metrics.

    min_price/min_adv_usd filter SELECTION only (training always sees all
    rows, incl. delisted — anti-survivorship). cache_dir persists per-fold
    predictions so replay_walkforward() can re-evaluate filter/exit/cost
    variants without retraining.
    """
    params = dict(lgb_params or V2_LGB_PARAMS)
    policy = exit_policy or ExitPolicy()
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
    cache_meta = []
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

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

        # Full-cross-section IC (continuity with pre-filter benchmarks)
        daily_ics = []
        for d in sorted(test_data.date.unique()):
            day = test_data[test_data.date == d]
            if len(day) < 10:
                continue
            ic, _ = spearmanr(day["pred"], day[V2_TARGET])
            daily_ics.append(ic)
        mean_ic = float(np.mean(daily_ics)) if daily_ics else 0.0
        ic_std = float(np.std(daily_ics)) if daily_ics else 0.0
        ic_ir = mean_ic / ic_std if ic_std > 0 else 0.0
        hit_rate_ic = float(np.mean([1 for ic in daily_ics if ic > 0]) / len(daily_ics)) if daily_ics else 0.0

        period = f"{fold['test_start'].strftime('%Y-%m')}->{fold['test_end'].strftime('%Y-%m')}"
        if cache_dir is not None:
            aux = [c for c in _CACHE_AUX_COLS if c in test_data.columns]
            cached = test_data[["date", "ticker", "pred", V2_TARGET] + aux]
            cached.to_parquet(cache_dir / f"fold_{i + 1:02d}.parquet", index=False)
            cache_meta.append({
                "fold": i + 1, "period": period,
                "test_start": str(fold["test_start"]), "test_end": str(fold["test_end"]),
                "train_rows": len(train_data), "test_rows": len(test_data),
                "mean_ic": mean_ic, "ic_ir": ic_ir, "hit_rate_ic": hit_rate_ic,
            })

        ev = _evaluate_fold(
            test_data, ohlcv, (fold["test_start"], fold["test_end"]),
            min_price, min_adv_usd, policy, cost_bps, use_spread_cost,
        )

        fm = FoldMetrics(
            fold=i + 1, period=period,
            train_rows=len(train_data), test_rows=len(test_data),
            mean_ic=mean_ic, ic_ir=ic_ir, hit_rate_ic=hit_rate_ic,
            total_return=ev["total_return"], sharpe=ev["sharpe"], max_dd=ev["max_dd"],
            n_trades=ev["n_trades"], win_rate=ev["win_rate"],
            median_return=ev["median_return"], market_return=ev["market_return"],
            mean_ic_tradable=ev["mean_ic_tradable"],
            n_skipped_rebalances=ev["n_skipped_rebalances"],
            avg_candidates=ev["avg_candidates"],
        )
        result.folds.append(fm)

        if verbose:
            alpha = fm.total_return - fm.market_return
            print(f"  Fold {i+1:2d} {fm.period}: IC={mean_ic:+.4f} ICtr={fm.mean_ic_tradable:+.4f} "
                  f"Ret={fm.total_return:+7.1%} Sharpe={fm.sharpe:+5.2f} WR={fm.win_rate:.0%} "
                  f"α={alpha:+7.1%} cand={fm.avg_candidates:.0f}")

    if cache_dir is not None:
        (cache_dir / "meta.json").write_text(json.dumps(cache_meta, indent=2))

    return result


def replay_walkforward(
    cache_dir: str | Path,
    ohlcv: pd.DataFrame,
    config_name: str,
    min_price: float | None = None,
    min_adv_usd: float | None = None,
    exit_policy: ExitPolicy | None = None,
    cost_bps: float = 0.0,
    use_spread_cost: bool = False,
    verbose: bool = False,
) -> RunResult:
    """Re-evaluate cached predictions under new filter/exit/cost settings.

    Seconds instead of ~1h: no LGB training. Training-side numbers
    (mean_ic, ic_ir, hit_rate_ic) come from the cache metadata unchanged.
    """
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "meta.json").read_text())
    policy = exit_policy or ExitPolicy()
    result = RunResult(config_name=config_name, feat_cols=[], n_features=0)

    for fm_meta in meta:
        test_data = pd.read_parquet(cache_dir / f"fold_{fm_meta['fold']:02d}.parquet")
        test_data["date"] = pd.to_datetime(test_data["date"])
        ev = _evaluate_fold(
            test_data, ohlcv,
            (pd.Timestamp(fm_meta["test_start"]), pd.Timestamp(fm_meta["test_end"])),
            min_price, min_adv_usd, policy, cost_bps, use_spread_cost,
        )
        fm = FoldMetrics(
            fold=fm_meta["fold"], period=fm_meta["period"],
            train_rows=fm_meta["train_rows"], test_rows=fm_meta["test_rows"],
            mean_ic=fm_meta["mean_ic"], ic_ir=fm_meta["ic_ir"],
            hit_rate_ic=fm_meta["hit_rate_ic"],
            total_return=ev["total_return"], sharpe=ev["sharpe"], max_dd=ev["max_dd"],
            n_trades=ev["n_trades"], win_rate=ev["win_rate"],
            median_return=ev["median_return"], market_return=ev["market_return"],
            mean_ic_tradable=ev["mean_ic_tradable"],
            n_skipped_rebalances=ev["n_skipped_rebalances"],
            avg_candidates=ev["avg_candidates"],
        )
        result.folds.append(fm)
        if verbose:
            print(f"  Fold {fm.fold:2d} {fm.period}: ICtr={fm.mean_ic_tradable:+.4f} "
                  f"Ret={fm.total_return:+7.1%} Sharpe={fm.sharpe:+5.2f} WR={fm.win_rate:.0%}")

    return result


def print_comparison(baseline: RunResult, *others: RunResult) -> None:
    """Print side-by-side comparison vs baseline."""
    rows = [baseline] + list(others)
    print("\n" + "=" * 100)
    print(f"  COMPARISON  ({len(rows)} configs)")
    print("=" * 100)

    keys = ["n_features", "mean_ic", "mean_ic_tradable", "ic_std", "mean_sharpe",
            "mean_return", "median_return", "mean_alpha", "mean_win_rate",
            "selectivity_median", "folds_positive_ret", "folds_positive_alpha",
            "total_trades", "skipped_rebalances"]

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
