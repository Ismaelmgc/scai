"""LIQUIDCAP — purged walk-forward screen: is there ANY exploitable signal on
the PIT S&P 500 universe, and does a different model/ensemble/feature set help?

This is the leak-free foundation the June mid-cap test lacked: PIT membership
(no selection-date bias), delisted included (no survivorship), purge_days=20
(no label look-ahead), dividend-adjusted total-return bars, ~34 quarterly folds
2018-2026 (vs 16 before — twice the statistical power).

Pre-registered configs (M=4 for the Šidák haircut):
  A  lambdarank_base   LambdaRank 16-bin, 24 base features (prod-like)
  B  regression_base   RMSE regression, same features (simpler model)
  C  lambdarank_new    LambdaRank, base + 6 NEW features (gap/idio-vol/volume-trend/range)
  D  ensemble_AB       per-date rank-average of A and B predictions (replay, no train)

Cost: 5 bps/side flat (all-in realistic for S&P 500 liquidity) + spread-aware
reported. Exit: prod pt40 trailing.

PROMOTION GATE to the next phase (feature engineering / sizing / leverage):
  1. best config's net return CI>0 over ALL folds (one-sample bootstrap)
  2. paired-vs-A p(<=0) after Šidák(4) < 0.05 for B/C/D claims of improvement
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


def build_ensemble_cache(a: str, b: str, out: str) -> None:
    (CACHE / out).mkdir(parents=True, exist_ok=True)
    meta = json.loads((CACHE / a / "meta.json").read_text())
    for fm in meta:
        fa = pd.read_parquet(CACHE / a / f"fold_{fm['fold']:02d}.parquet")
        fb = pd.read_parquet(CACHE / b / f"fold_{fm['fold']:02d}.parquet")
        fb = fb[["date", "ticker", "pred"]].rename(columns={"pred": "pred_b"})
        m = fa.merge(fb, on=["date", "ticker"], how="inner")
        ra = m.groupby("date")["pred"].rank(pct=True)
        rb = m.groupby("date")["pred_b"].rank(pct=True)
        m["pred"] = (ra + rb) / 2
        m.drop(columns=["pred_b"]).to_parquet(CACHE / out / f"fold_{fm['fold']:02d}.parquet",
                                              index=False)
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
        ps = mc.sidak(pb["p_le0"], 4)
        line += f"  dA{pb['mean_diff']:+.1%} pSidak={ps:.2f}"
    print(line, flush=True)
    return bs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=300)
    args = ap.parse_args()
    t0 = time.time()

    feats = pd.read_parquet(DATA / "features_liquidcap.parquet")
    feats["date"] = pd.to_datetime(feats["date"])
    ohlcv = pd.read_parquet(DATA / "ohlcv_sp500.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    ohlcv = ohlcv[ohlcv.ticker != "SPY"]

    base = [f for f in h.V2_FEATURES_BASE if f in feats.columns]
    new = [f for f in NEW_FEATURES if f in feats.columns]
    print(f"  features: {len(base)} base + {len(new)} new; "
          f"rows {len(feats):,}; tickers {feats.ticker.nunique()}")

    # More folds: test starts 2018 (4y min training 2014-2018)
    h.MIN_TRAIN_END = pd.Timestamp("2018-01-01")
    policy = h.ExitPolicy(profit_target=0.40)

    train(feats, ohlcv, base, "A_lambdarank_base", True, policy, args.trees)
    train(feats, ohlcv, base, "B_regression_base", False, policy, args.trees)
    train(feats, ohlcv, base + new, "C_lambdarank_new", True, policy, args.trees)
    build_ensemble_cache("A_lambdarank_base", "B_regression_base", "D_ensemble_AB")

    print(f"\n  === SCREEN RESULTS (purged 20d, {COST_BPS:g}bps/side flat, "
          f"pt40, top-8) ===", flush=True)
    dfs = {}
    for name in ["A_lambdarank_base", "B_regression_base", "C_lambdarank_new",
                 "D_ensemble_AB"]:
        dfs[name] = replay(name, ohlcv, policy)
    a = dfs["A_lambdarank_base"]
    for name, df in dfs.items():
        report(name, df, None if name.startswith("A") else a)

    print("\n  spread-aware (Corwin-Schultz floor):", flush=True)
    for name in dfs:
        report(name, replay(name, ohlcv, policy, spread=True))

    print(f"\n  Runtime {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
