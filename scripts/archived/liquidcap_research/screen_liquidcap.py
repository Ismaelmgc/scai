"""LIQUIDCAP — purged walk-forward screen: is there ANY exploitable signal on
the PIT S&P 500 universe, and does a different model/ensemble/feature set help?

This is the leak-free foundation the June mid-cap test lacked: PIT membership
(no selection-date bias), delisted included (no survivorship), purge_days=20
(no label look-ahead), dividend-adjusted total-return bars, ~34 quarterly folds
2018-2026 (vs 16 before — twice the statistical power).

Model zoo grounded in the literature: Gu-Kelly-Xiu (RFS 2020) find shallow NNs
and boosted trees tie for best, with regularized linear capturing much of the
signal and momentum/liquidity/volatility variants dominating importance;
Grinsztajn et al. (NeurIPS 2022) show trees beat NNs on medium tabular data
(NNs hurt by uninformative features); Fischer-Krauss (EJOR 2018) show the LSTM
edge on S&P 500 constituents decayed to ~0 net of costs after 2010.

Pre-registered configs (M=8 for the Šidák haircut — includes the separate
LSTM pilot in lstm_pilot.py):
  A  lambdarank_base  LambdaRank 16-bin, 24 base features (prod-like anchor)
  B  regression_base  LGB RMSE regression, same features (simpler model)
  C  lambdarank_new   LambdaRank, base + 11 NEW features (gap/idio-vol/volume
                      -trend/range + GKX mom_12_1/mom_6_1/rev_21d/MAX/turn-vol)
  D  ensemble_ABE     per-date rank-average of A+B+E predictions (no training)
  E  ridge_new        Ridge on per-date cross-sectional ranks (GKX linear bench)
  F  mlp_new          shallow MLP (32,16) on per-date ranks (GKX NN2-style)
  G  lambdarank_fs15  LambdaRank on the top-15 features by LGB gain, selected on
                      PRE-2018 data only (predates every test window — no leak).
                      Tests "fewer, better-chosen features".
  I  lambdarank_fund  LambdaRank, base + new + 10 SEC-EDGAR fundamental ratios
                      (E/P, CFO yield, B/M, D/E, ROA, accruals, asset growth,
                      PEAD-ish earnings change — filing-date PIT, $0 cost).
                      Runs only if build_fundamentals.py output exists. CAVEAT:
                      CIK coverage of delisted members is lower than current
                      ones — judge with that asymmetry in mind.

Cost: 5 bps/side flat (all-in realistic for S&P 500 liquidity) + spread-aware
reported. Exit: prod pt40 trailing.

PROMOTION GATE to the next phase (feature engineering / sizing / leverage):
  1. best config's net return CI>0 over ALL folds (one-sample bootstrap)
  2. improvement claims vs A: paired p(<=0) after Šidák(8) < 0.05
  3. CONFIRM block (folds after 2024-07, never used to choose) return > 0
  4. IC stable over time — an IC that trends up toward the present = leftover
     universe bias, auto-reject (the June mid-cap signature).

Usage:
    PYTHONPATH=src python scripts/liquidcap/screen_liquidcap.py [--trees N]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "v3"))
sys.path.insert(0, str(ROOT / "scripts" / "liquidcap"))

import _v3_harness as h  # noqa: E402
from build_features_liquidcap import NEW_FEATURES  # noqa: E402

_spec = importlib.util.spec_from_file_location("mc", ROOT / "scripts/v3/34_mc_validate.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

DATA = ROOT / "data" / "liquidcap"
CACHE = DATA / "cache"
COST_BPS = 5.0
CONFIRM_START = pd.Timestamp("2024-07-01")
M_TRIALS = 11  # A..G + LSTM pilot + fundamentals + horizon 10d/5d (horizon_screen)
FS_CUTOFF = pd.Timestamp("2018-01-01")  # feature selection uses data before this

LAMBDARANK_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg",
    "num_leaves": 31, "max_depth": 6,
    "learning_rate": 0.05, "min_child_samples": 30,
    "subsample": 0.75, "colsample_bytree": 0.7,
    "reg_lambda": 5.0, "n_jobs": 1, "seed": 42, "verbose": -1,
}


def train(features, ohlcv, feat_cols, name, lambdarank, policy, trees):
    cache = CACHE / name
    if (cache / "meta.json").exists():
        print(f"  ({name}: cache exists, skip)")
        return
    print(f"\n=== training {name} ({len(feat_cols)} feats, trees={trees}) ===", flush=True)
    h.run_walkforward(
        features, ohlcv, feat_cols, config_name=name,
        lgb_params=LAMBDARANK_PARAMS if lambdarank else None,
        objective_lambdarank=lambdarank, n_bins=16,
        min_price=1.5, min_adv_usd=500_000, exit_policy=policy,
        cost_bps=COST_BPS, cache_dir=cache, num_boost_round=trees,
        purge_days=20, verbose=True,
    )


# ───────────────── sklearn models on per-date cross-sectional ranks ─────────
def _xs_rank(df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    """Per-date pct-rank mapped to [-0.5, 0.5], NaN -> 0 (GKX preprocessing)."""
    r = df.groupby("date")[feat_cols].rank(pct=True) - 0.5
    return r.fillna(0.0).values


def sklearn_walkforward(features, feat_cols, name, make_model) -> None:
    """Purged walk-forward for sklearn regressors; writes a replay-compatible
    prediction cache (same folds/purge as the harness)."""
    cache = CACHE / name
    if (cache / "meta.json").exists():
        print(f"  ({name}: cache exists, skip)")
        return
    cache.mkdir(parents=True, exist_ok=True)
    print(f"\n=== training {name} ({len(feat_cols)} feats, sklearn) ===", flush=True)
    folds = h.define_folds(features)
    all_dates = np.array(sorted(features["date"].unique()))
    aux = [c for c in ("atr_pct_20d", "close", "adv_usd_20d", "cs_spread_20d")
           if c in features.columns]
    meta = []
    for i, fold in enumerate(folds):
        tr = features[(features.date >= fold["train_start"])
                      & (features.date < fold["train_end"])].dropna(subset=[h.V2_TARGET])
        prior = all_dates[all_dates < fold["test_start"]]
        if len(prior) > 20:  # same 20d label purge as the harness
            tr = tr[tr["date"] < prior[-20]]
        y = tr[h.V2_TARGET].values
        lo, hi = np.quantile(y, [0.01, 0.99])  # winsorize (train-only) for NN stability
        model = make_model()
        model.fit(_xs_rank(tr, feat_cols), np.clip(y, lo, hi))

        te = features[(features.date >= fold["test_start"])
                      & (features.date < fold["test_end"])].dropna(subset=[h.V2_TARGET]).copy()
        if te.empty:
            continue
        te["pred"] = model.predict(_xs_rank(te, feat_cols))
        ics = [spearmanr(g["pred"], g[h.V2_TARGET])[0]
               for _, g in te.groupby("date") if len(g) >= 10]
        period = f"{fold['test_start']:%Y-%m}->{fold['test_end']:%Y-%m}"
        te[["date", "ticker", "pred", h.V2_TARGET] + aux].to_parquet(
            cache / f"fold_{i + 1:02d}.parquet", index=False)
        meta.append({"fold": i + 1, "period": period,
                     "test_start": str(fold["test_start"]), "test_end": str(fold["test_end"]),
                     "train_rows": len(tr), "test_rows": len(te),
                     "mean_ic": float(np.mean(ics)) if ics else 0.0,
                     "ic_ir": 0.0, "hit_rate_ic": 0.0})
        print(f"  Fold {i + 1:2d} {period}: IC={meta[-1]['mean_ic']:+.4f}", flush=True)
    (cache / "meta.json").write_text(json.dumps(meta, indent=2))


def select_features_pre2018(features, candidates: list[str], k: int = 15) -> list[str]:
    """Top-k features by LGB gain trained ONLY on pre-2018 rows — that data
    predates every test window, so the selection cannot leak."""
    import lightgbm as lgb
    tr = features[features.date < FS_CUTOFF].dropna(subset=[h.V2_TARGET])
    ds = lgb.Dataset(tr[candidates].fillna(0).values, tr[h.V2_TARGET].values,
                     feature_name=candidates, free_raw_data=True)
    params = dict(h.V2_LGB_PARAMS)
    model = lgb.train(params, ds, num_boost_round=200,
                      callbacks=[lgb.log_evaluation(0)])
    imp = pd.Series(model.feature_importance("gain"), index=candidates)
    top = imp.sort_values(ascending=False).head(k)
    print(f"  pre-2018 top-{k} by gain: {list(top.index)}")
    return list(top.index)


def build_ensemble_cache(sources: list[str], out: str) -> None:
    (CACHE / out).mkdir(parents=True, exist_ok=True)
    meta = json.loads((CACHE / sources[0] / "meta.json").read_text())
    for fm in meta:
        base = pd.read_parquet(CACHE / sources[0] / f"fold_{fm['fold']:02d}.parquet")
        ranks = [base.groupby("date")["pred"].rank(pct=True)]
        for s in sources[1:]:
            fp = CACHE / s / f"fold_{fm['fold']:02d}.parquet"
            other = pd.read_parquet(fp)[["date", "ticker", "pred"]]
            other = other.rename(columns={"pred": "pred_o"})
            m = base.merge(other, on=["date", "ticker"], how="left")
            ranks.append(m.groupby("date")["pred_o"].rank(pct=True))
        base["pred"] = pd.concat(ranks, axis=1).mean(axis=1).values
        base.to_parquet(CACHE / out / f"fold_{fm['fold']:02d}.parquet", index=False)
    (CACHE / out / "meta.json").write_text(json.dumps(meta))


def replay(name, ohlcv, policy, spread=False):
    meta = json.loads((CACHE / name / "meta.json").read_text())
    rows = []
    for fm in meta:
        td = pd.read_parquet(CACHE / name / f"fold_{fm['fold']:02d}.parquet")
        td["date"] = pd.to_datetime(td["date"])
        ev = h._evaluate_fold(
            td, ohlcv, (pd.Timestamp(fm["test_start"]), pd.Timestamp(fm["test_end"])),
            1.5, 500_000, policy, COST_BPS, spread)
        rows.append({"fold": fm["fold"], "test_start": fm["test_start"],
                     "ic_tr": ev["mean_ic_tradable"],
                     **{k: ev[k] for k in ("total_return", "sharpe", "win_rate",
                                           "n_trades", "max_dd")}})
    return pd.DataFrame(rows)


def report(name, df, base_df=None):
    bs = mc.one_sample_bootstrap(df["total_return"].values)
    mo = (1 + df["total_return"].mean()) ** (1 / 3) - 1
    conf = df[pd.to_datetime(df["test_start"]) >= CONFIRM_START]
    ic = df["ic_tr"].mean()
    # IC trend check: corr(fold index, IC) — positive trend = universe bias
    ic_trend = np.corrcoef(np.arange(len(df)), df["ic_tr"])[0, 1]
    line = (f"  {name:18} ret {bs['mean']:+6.1%}/f [{bs['ci_lo']:+.1%},{bs['ci_hi']:+.1%}]"
            f"{'*' if bs['ci_lo'] > 0 else ' '} {mo:+6.2%}/mo Sh{df['sharpe'].mean():5.2f} "
            f"IC{ic:+.4f} (trend r={ic_trend:+.2f}) conf{conf['total_return'].mean():+6.1%}/f")
    if base_df is not None:
        pb = mc.paired_bootstrap(df["total_return"].values, base_df["total_return"].values)
        ps = mc.sidak(pb["p_le0"], M_TRIALS)
        line += f"  dA{pb['mean_diff']:+.1%} pSidak={ps:.2f}"
    print(line, flush=True)
    return bs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=300)
    args = ap.parse_args()
    t0 = time.time()

    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor

    feats = pd.read_parquet(DATA / "features_liquidcap.parquet")
    feats["date"] = pd.to_datetime(feats["date"])
    ohlcv = pd.read_parquet(DATA / "ohlcv_sp500.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    ohlcv = ohlcv[ohlcv.ticker != "SPY"]

    base = [f for f in h.V2_FEATURES_BASE if f in feats.columns]
    new = [f for f in NEW_FEATURES if f in feats.columns]

    # Optional fundamentals (I): free SEC-EDGAR ratios, filing-date PIT
    fund: list[str] = []
    fund_fp = DATA / "fundamentals_daily.parquet"
    if fund_fp.exists():
        from build_fundamentals import RATIOS
        fnd = pd.read_parquet(fund_fp)
        fnd["date"] = pd.to_datetime(fnd["date"])
        feats = feats.merge(fnd, on=["date", "ticker"], how="left")
        fund = [r for r in RATIOS if r in feats.columns]
    print(f"  features: {len(base)} base + {len(new)} new + {len(fund)} fund; "
          f"rows {len(feats):,}; tickers {feats.ticker.nunique()}")

    # More folds: test starts 2018 (4y min training 2014-2018)
    h.MIN_TRAIN_END = pd.Timestamp("2018-01-01")
    policy = h.ExitPolicy(profit_target=0.40)

    train(feats, ohlcv, base, "A_lambdarank_base", True, policy, args.trees)
    train(feats, ohlcv, base, "B_regression_base", False, policy, args.trees)
    train(feats, ohlcv, base + new, "C_lambdarank_new", True, policy, args.trees)
    sklearn_walkforward(feats, base + new, "E_ridge_new", lambda: Ridge(alpha=10.0))
    sklearn_walkforward(feats, base + new, "F_mlp_new", lambda: MLPRegressor(
        hidden_layer_sizes=(32, 16), batch_size=8192, learning_rate_init=1e-3,
        alpha=1e-4, max_iter=25, early_stopping=True, n_iter_no_change=3,
        validation_fraction=0.05, random_state=42))
    fs15 = select_features_pre2018(feats, base + new, k=15)
    train(feats, ohlcv, fs15, "G_lambdarank_fs15", True, policy, args.trees)
    if fund:
        train(feats, ohlcv, base + new + fund, "I_lambdarank_fund", True,
              policy, args.trees)
    build_ensemble_cache(["A_lambdarank_base", "B_regression_base", "E_ridge_new"],
                         "D_ensemble_ABE")

    names = ["A_lambdarank_base", "B_regression_base", "C_lambdarank_new",
             "D_ensemble_ABE", "E_ridge_new", "F_mlp_new", "G_lambdarank_fs15"]
    if fund:
        names.append("I_lambdarank_fund")
    print(f"\n  === SCREEN RESULTS (purged 20d, {COST_BPS:g}bps/side flat, "
          f"pt40, top-8, Sidak M={M_TRIALS}) ===", flush=True)
    dfs = {n: replay(n, ohlcv, policy) for n in names}
    a = dfs["A_lambdarank_base"]
    for name, df in dfs.items():
        report(name, df, None if name.startswith("A") else a)

    print("\n  spread-aware (Corwin-Schultz floor):", flush=True)
    for name in names:
        report(name, replay(name, ohlcv, policy, spread=True))

    print(f"\n  Runtime {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
