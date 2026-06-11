"""V3 Sprint 2 — task (d) — Regime-aware position sizing.

Re-runs the v3_hp_combo model (best so far) but applies a regime-based
exposure multiplier to the portfolio.

Position sizing rules (date-level multiplier 0.0 .. 1.0 applied to portfolio return):
  - IWM drawdown > -8%   AND   VIX z-score < 1.0:   exposure = 1.0 (full risk-on)
  - IWM drawdown in [-15%, -8%]  OR  VIX z [1.0, 2.0]:   exposure = 0.6
  - IWM drawdown < -15%  OR  VIX z > 2.0:   exposure = 0.3 (defensive)

Backtest: portfolio_period_return *= exposure_at_rebalance_date
Then Sharpe / total return / win rate are computed on the scaled returns.

Compare against v3_hp_combo (same predictions, no scaling).
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from _v3_harness import (
    V2_FEATURES_BASE, V2_EDGAR_FEATURES, V2_META_FEATURES,
    V2_LGB_PARAMS, define_folds, FoldMetrics, RunResult,
    TOP_K, HOLD_DAYS, REBALANCE_EVERY, V2_TARGET, save_result,
)

N_BINS = 16
NB = 600


def make_params():
    p = dict(V2_LGB_PARAMS)
    p.update({
        "objective": "lambdarank", "metric": "ndcg",
        "num_leaves": 31, "max_depth": 6,
        "min_child_samples": 30, "learning_rate": 0.05,
        "lambdarank_truncation_level": 8,
        "label_gain": list(range(N_BINS)),
        "reg_lambda": 5.0,
    })
    p.pop("reg_alpha", None)
    return p


def build_regime_table() -> pd.DataFrame:
    """Date-indexed regime exposure multiplier."""
    idx = pd.read_parquet("data/processed/market_indices.parquet")
    idx["date"] = pd.to_datetime(idx["date"])
    p = idx.pivot(index="date", columns="ticker", values="close").sort_index()

    iwm = p["IWM"]
    vix = p["^VIX"]
    iwm_dd = iwm / iwm.cummax() - 1
    vix_z = (vix - vix.rolling(60, min_periods=20).mean()) / vix.rolling(60, min_periods=20).std()

    df = pd.DataFrame({"iwm_dd": iwm_dd, "vix_z": vix_z}).reset_index()
    def expo(row):
        if pd.isna(row.iwm_dd):
            return 1.0
        if row.iwm_dd < -0.15 or (not pd.isna(row.vix_z) and row.vix_z > 2.0):
            return 0.3
        if row.iwm_dd < -0.08 or (not pd.isna(row.vix_z) and row.vix_z > 1.0):
            return 0.6
        return 1.0
    df["exposure"] = df.apply(expo, axis=1)
    return df[["date", "exposure"]]


def run_with_exposure(features, ohlcv, feat_cols, params, regime_tbl):
    folds = define_folds(features)
    result = RunResult(config_name="v3_regime_sized", feat_cols=feat_cols, n_features=len(feat_cols))
    expo_map = dict(zip(regime_tbl["date"], regime_tbl["exposure"]))

    for i, fold in enumerate(folds):
        m = (features.date >= fold["train_start"]) & (features.date < fold["train_end"])
        td = features[m].dropna(subset=[V2_TARGET]).sort_values("date").copy()
        td["_rel"] = td.groupby("date")[V2_TARGET].transform(
            lambda s: pd.qcut(s.rank(method="first"), N_BINS, labels=False, duplicates="drop")
        )
        td["_rel"] = td["_rel"].fillna(0).astype(int).clip(0, N_BINS - 1)
        X = td[feat_cols].fillna(0).values
        y = td["_rel"].values
        group = td.groupby("date").size().values
        ds = lgb.Dataset(X, y, group=group, feature_name=feat_cols, free_raw_data=True)
        model = lgb.train(params, ds, num_boost_round=NB,
                          callbacks=[lgb.log_evaluation(0)])
        test_m = (features.date >= fold["test_start"]) & (features.date < fold["test_end"])
        test = features[test_m].dropna(subset=[V2_TARGET]).copy()
        if test.empty:
            continue
        test["pred"] = model.predict(test[feat_cols].fillna(0).values)
        test_dates = sorted(test.date.unique())

        # IC
        ics = []
        for d in test_dates:
            day = test[test.date == d]
            if len(day) < 10:
                continue
            ic, _ = spearmanr(day["pred"], day[V2_TARGET])
            ics.append(ic)
        mean_ic = float(np.mean(ics)) if ics else 0.0
        ic_std = float(np.std(ics)) if ics else 0.0

        rebs = test_dates[::REBALANCE_EVERY]
        trades, port, topk_rets = [], [], []
        for reb in rebs:
            expo = expo_map.get(reb, 1.0)
            day = test[test.date == reb]
            if len(day) < TOP_K:
                continue
            top = day.sort_values("pred", ascending=False).head(TOP_K)
            pr = []
            for _, row in top.iterrows():
                t_oh = ohlcv[(ohlcv.ticker == row.ticker) & (ohlcv.date >= reb)]
                if len(t_oh) < 2:
                    continue
                prices = t_oh.head(HOLD_DAYS + 1)["close"].values
                vol = float(row.get("atr_pct_20d", 0.03))
                trail = np.clip(vol * 5.3, 0.10, 0.16)
                peak, ar, hit = prices[0], 0.0, False
                for k in range(1, len(prices)):
                    peak = max(peak, prices[k])
                    if (prices[k] - peak) / peak <= -trail:
                        ar = (prices[k] - prices[0]) / prices[0]
                        hit = True
                        break
                if not hit:
                    ar = (prices[-1] - prices[0]) / prices[0]
                # apply exposure
                ar_scaled = ar * expo
                trades.append({"actual_ret": ar_scaled, "raw": ar, "expo": expo})
                pr.append(ar_scaled)
                topk_rets.append(ar)  # selectivity is unscaled (model quality)
            if pr:
                port.append(float(np.mean(pr)))
        port = pd.Series(port)
        if not port.empty:
            tr = float((1 + port).prod() - 1)
            sh = float((port.mean() / port.std()) * np.sqrt(252 / REBALANCE_EVERY)) if port.std() > 0 else 0.0
            cum = (1 + port).cumprod()
            mdd = float(((cum - cum.cummax()) / cum.cummax()).min())
        else:
            tr = sh = mdd = 0.0
        wr = float(np.mean([t["actual_ret"] > 0 for t in trades])) if trades else 0.0
        smed = float(np.median(topk_rets)) if topk_rets else 0.0
        foh = ohlcv[(ohlcv.date >= fold["test_start"]) & (ohlcv.date < fold["test_end"])]
        if not foh.empty:
            sp = foh.drop_duplicates("ticker", keep="first").set_index("ticker")["close"]
            ep = foh.drop_duplicates("ticker", keep="last").set_index("ticker")["close"]
            common = sp.index.intersection(ep.index)
            mret = float(((ep[common] / sp[common]) - 1).median()) if len(common) > 10 else 0.0
        else:
            mret = 0.0
        avg_expo = float(np.mean([t["expo"] for t in trades])) if trades else 1.0
        result.folds.append(FoldMetrics(
            fold=i+1, period=f"{fold['test_start'].strftime('%Y-%m')}->{fold['test_end'].strftime('%Y-%m')}",
            train_rows=len(td), test_rows=len(test),
            mean_ic=mean_ic, ic_ir=(mean_ic/ic_std if ic_std else 0), hit_rate_ic=0,
            total_return=tr, sharpe=sh, max_dd=mdd,
            n_trades=len(trades), win_rate=wr, median_return=smed, market_return=mret,
        ))
        print(f"  Fold {i+1:2d} {fold['test_start'].strftime('%Y-%m')}: "
              f"IC={mean_ic:+.4f} Sh={sh:+5.2f} WR={wr:.0%} Ret={tr:+7.1%} avgExpo={avg_expo:.2f}")
    save_result(result)
    return result


def main() -> None:
    print("Loading...")
    features = pd.read_parquet("data/processed/features_smallcap_v3_sector.parquet")
    ohlcv = pd.read_parquet("data/processed/ohlcv_smallcap.parquet")
    features["date"] = pd.to_datetime(features["date"])
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    regime_tbl = build_regime_table()
    print(f"regime table: {regime_tbl.shape}  exposure distrib:")
    print(regime_tbl["exposure"].value_counts(normalize=True).sort_index())

    feat_cols = V2_FEATURES_BASE + V2_EDGAR_FEATURES + V2_META_FEATURES
    params = make_params()
    t0 = time.time()
    r = run_with_exposure(features, ohlcv, feat_cols, params, regime_tbl)
    agg = r.aggregate()
    print(f"\n=== v3_regime_sized (elapsed {time.time()-t0:.0f}s) ===")
    for k, v in agg.items():
        if isinstance(v, float):
            if any(t in k for t in ("rate", "return", "alpha", "selectivity")):
                print(f"  {k:<28} {v:.2%}")
            else:
                print(f"  {k:<28} {v:.4f}")
        else:
            print(f"  {k:<28} {v}")


if __name__ == "__main__":
    main()
