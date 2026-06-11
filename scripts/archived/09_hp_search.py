"""V3 Sprint 2 — task (c) — Hyperparam search for LambdaRank.

Limited search around V2 base. Test 8 configs varying:
  - num_leaves: 20 (current), 31, 63
  - min_child_samples: 60 (current), 30, 100
  - learning_rate: 0.02 (current), 0.05
  - num_boost_round: 400 (current), 600

Combinations selected for breadth without combinatorial explosion (8 runs):
  1. baseline (current candidate)
  2. leaves=31
  3. leaves=63
  4. min_child=30
  5. min_child=100
  6. lr=0.05, fewer rounds=200
  7. leaves=31 + min_child=30 + lr=0.05 + 600 rounds
  8. truncation=4 (sharper top picks)
"""
from __future__ import annotations

import json, sys, time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from _v3_harness import (
    V2_FEATURES_BASE, V2_EDGAR_FEATURES, V2_META_FEATURES,
    V2_LGB_PARAMS, V2_TARGET, define_folds, FoldMetrics, RunResult,
    TOP_K, HOLD_DAYS, REBALANCE_EVERY, save_result,
)


N_BINS = 16


def make_params(num_leaves=20, max_depth=5, min_child=60, lr=0.02,
                trunc=8, reg_lambda=5.0):
    p = dict(V2_LGB_PARAMS)
    p.update({
        "objective": "lambdarank",
        "metric": "ndcg",
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "min_child_samples": min_child,
        "learning_rate": lr,
        "lambdarank_truncation_level": trunc,
        "label_gain": list(range(N_BINS)),
        "reg_lambda": reg_lambda,
    })
    p.pop("reg_alpha", None)
    return p


def run_one(features, ohlcv, feat_cols, params, num_boost, name):
    """Inline simplified run_walkforward (avoid re-loading data)."""
    folds = define_folds(features)
    result = RunResult(config_name=name, feat_cols=feat_cols, n_features=len(feat_cols))
    t0 = time.time()
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
        model = lgb.train(params, ds, num_boost_round=num_boost,
                          callbacks=[lgb.log_evaluation(0)])

        test_m = (features.date >= fold["test_start"]) & (features.date < fold["test_end"])
        test = features[test_m].dropna(subset=[V2_TARGET]).copy()
        if test.empty:
            continue
        test["pred"] = model.predict(test[feat_cols].fillna(0).values)
        # IC
        ics = []
        test_dates = sorted(test.date.unique())
        for d in test_dates:
            day = test[test.date == d]
            if len(day) < 10:
                continue
            ic, _ = spearmanr(day["pred"], day[V2_TARGET])
            ics.append(ic)
        mean_ic = float(np.mean(ics)) if ics else 0.0
        ic_std = float(np.std(ics)) if ics else 0.0

        # Trades
        rebs = test_dates[::REBALANCE_EVERY]
        trades, port, topk_rets = [], [], []
        for reb in rebs:
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
                trades.append({"actual_ret": ar})
                pr.append(ar)
                topk_rets.append(ar)
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
        # market
        foh = ohlcv[(ohlcv.date >= fold["test_start"]) & (ohlcv.date < fold["test_end"])]
        if not foh.empty:
            sp = foh.drop_duplicates("ticker", keep="first").set_index("ticker")["close"]
            ep = foh.drop_duplicates("ticker", keep="last").set_index("ticker")["close"]
            common = sp.index.intersection(ep.index)
            mret = float(((ep[common] / sp[common]) - 1).median()) if len(common) > 10 else 0.0
        else:
            mret = 0.0
        result.folds.append(FoldMetrics(
            fold=i+1, period=f"{fold['test_start'].strftime('%Y-%m')}->{fold['test_end'].strftime('%Y-%m')}",
            train_rows=len(td), test_rows=len(test),
            mean_ic=mean_ic, ic_ir=(mean_ic/ic_std if ic_std else 0), hit_rate_ic=0,
            total_return=tr, sharpe=sh, max_dd=mdd,
            n_trades=len(trades), win_rate=wr, median_return=smed, market_return=mret,
        ))
    save_result(result)
    return result


def main() -> None:
    print("Loading...")
    features = pd.read_parquet("data/processed/features_smallcap_v3_sector.parquet")
    ohlcv = pd.read_parquet("data/processed/ohlcv_smallcap.parquet")
    features["date"] = pd.to_datetime(features["date"])
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    feat_cols = V2_FEATURES_BASE + V2_EDGAR_FEATURES + V2_META_FEATURES

    configs = [
        ("v3_hp_leaves31",   make_params(num_leaves=31, max_depth=6), 400),
        ("v3_hp_leaves63",   make_params(num_leaves=63, max_depth=7), 400),
        ("v3_hp_minc30",     make_params(min_child=30), 400),
        ("v3_hp_minc100",    make_params(min_child=100), 400),
        ("v3_hp_lr05",       make_params(lr=0.05), 200),
        ("v3_hp_trunc4",     make_params(trunc=4), 400),
        ("v3_hp_combo",      make_params(num_leaves=31, max_depth=6, min_child=30, lr=0.05), 600),
        ("v3_hp_reg2",       make_params(reg_lambda=2.0), 400),
    ]

    results = []
    print(f"Testing {len(configs)} configs...")
    for name, params, nb in configs:
        print(f"\n=== {name} ===")
        t0 = time.time()
        r = run_one(features, ohlcv, feat_cols, params, nb, name)
        agg = r.aggregate()
        print(f"  IC={agg['mean_ic']:+.4f}  Sh={agg['mean_sharpe']:.2f}  "
              f"WR={agg['mean_win_rate']:.2%}  SelMed={agg['selectivity_median']:.2%}  "
              f"+folds={agg['folds_positive_ret']}  ({time.time()-t0:.0f}s)")
        results.append((name, agg))

    # Reference
    cand = json.loads(Path("data/v3_benchmarks/v3_lambdarank.json").read_text())["aggregate"]
    print("\n\n=== HP SEARCH SUMMARY ===")
    print(f"{'Config':<22} {'IC':>8} {'Sh':>6} {'WR':>8} {'SelMed':>8} {'+folds':>8}")
    print(f"{'v3_lambdarank (REF)':<22} {cand['mean_ic']:>+8.4f} {cand['mean_sharpe']:>6.2f} "
          f"{cand['mean_win_rate']:>7.2%} {cand['selectivity_median']:>7.2%} {cand['folds_positive_ret']:>8}")
    for name, a in results:
        marker = "★" if (a['mean_sharpe'] > cand['mean_sharpe'] and a['mean_win_rate'] > cand['mean_win_rate']) else " "
        print(f"{marker}{name:<21} {a['mean_ic']:>+8.4f} {a['mean_sharpe']:>6.2f} "
              f"{a['mean_win_rate']:>7.2%} {a['selectivity_median']:>7.2%} {a['folds_positive_ret']:>8}")


if __name__ == "__main__":
    main()
