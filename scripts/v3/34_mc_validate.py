"""Monte Carlo validation toolkit for the walk-forward harness.

Why: searching feature subsets x TOP_K over a fixed 16-fold backtest is the #1
overfitting trap — with enough trials you WILL find a winner by luck. This module
provides the statistical guardrails used by 35/36/37 (and, as a bonus, the first
honest confidence intervals on the deployed strategy):

  paired_bootstrap(a, b)  : iid resample of per-fold returns -> distribution of
                            mean(candidate) - mean(baseline); 95% CI + p(diff<=0).
  one_sample_bootstrap(x) : CI for a single strategy's mean (e.g. baseline return).
  sidak(p, M)             : multiple-testing haircut for M trials, 1-(1-p)^M.
  prob_sharpe / def_sharpe: PSR and Deflated Sharpe Ratio (Bailey-Lopez de Prado),
                            DSR uses the dispersion of the M trial Sharpes.

A candidate only "wins" if the held-out block beats baseline AND the paired
bootstrap CI excludes 0 AFTER the Sidak haircut for the number of trials.

The functions operate on plain arrays so 35/36/37 import this module via
importlib (filename starts with a digit). __main__ runs a sanity check: CIs on
the deployed baseline (top-8) and whether the round-2 top-6 edge is real.

Usage:
    PYTHONPATH=src python scripts/v3/34_mc_validate.py
"""
from __future__ import annotations

import json
import sys
from math import e, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "v3"))

from _v3_harness import (  # noqa: E402
    ExitPolicy,
    _evaluate_fold,
)

DECISION_FP = ROOT / "data" / "v3_benchmarks" / "v4_filter_decision.json"
CACHE_ROOT = ROOT / "data" / "v3_benchmarks" / "cache"
EULER = 0.5772156649015329


# ───────────────────────── bootstrap ─────────────────────────
def one_sample_bootstrap(x, B: int = 10000, seed: int = 42, alpha: float = 0.05) -> dict:
    """Bootstrap CI for the mean of a per-fold metric array."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    means = x[idx].mean(axis=1)
    return {
        "mean": float(x.mean()),
        "ci_lo": float(np.quantile(means, alpha / 2)),
        "ci_hi": float(np.quantile(means, 1 - alpha / 2)),
        "p_le0": float((means <= 0).mean()),
    }


def paired_bootstrap(a, b, B: int = 10000, seed: int = 42, alpha: float = 0.05) -> dict:
    """Paired (same resampled folds) bootstrap of mean(a) - mean(b).

    a = candidate per-fold returns, b = baseline per-fold returns (aligned folds).
    Returns CI of the difference and p(diff <= 0) = chance the candidate is NOT
    better (one-sided), which feeds the Sidak multiple-testing haircut.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert len(a) == len(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(B, len(a)))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return {
        "mean_diff": float((a - b).mean()),
        "ci_lo": float(np.quantile(diffs, alpha / 2)),
        "ci_hi": float(np.quantile(diffs, 1 - alpha / 2)),
        "p_le0": float((diffs <= 0).mean()),
    }


def sidak(p: float, m: int) -> float:
    """Multiple-testing haircut: prob. that the BEST of m trials looks this good."""
    return float(1.0 - (1.0 - p) ** max(m, 1))


# ───────────────────────── (deflated) Sharpe ─────────────────────────
def _sr_moments(returns):
    r = np.asarray(returns, dtype=float)
    n = len(r)
    sd = r.std(ddof=1)
    sr = r.mean() / sd if sd > 0 else 0.0
    skew = ((r - r.mean()) ** 3).mean() / sd ** 3 if sd > 0 else 0.0
    kurt = ((r - r.mean()) ** 4).mean() / sd ** 4 if sd > 0 else 3.0
    return sr, skew, kurt, n


def prob_sharpe(returns, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > benchmark) given non-normality."""
    sr, skew, kurt, n = _sr_moments(returns)
    denom = sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    if denom <= 0 or n < 2:
        return float("nan")
    return float(norm.cdf((sr - sr_benchmark) * sqrt(n - 1) / denom))


def expected_max_sharpe(trial_sr_std: float, m: int) -> float:
    """E[max SR] across m independent trials under the null (true SR=0)."""
    if m < 2 or trial_sr_std <= 0:
        return 0.0
    return trial_sr_std * ((1 - EULER) * norm.ppf(1 - 1 / m)
                           + EULER * norm.ppf(1 - 1 / (m * e)))


def def_sharpe(returns, trial_sr_std: float, m: int) -> float:
    """Deflated Sharpe Ratio: PSR against the SR0 expected from m trials."""
    return prob_sharpe(returns, sr_benchmark=expected_max_sharpe(trial_sr_std, m))


# ───────────────────────── replay (reusable by 35/36) ─────────────────────────
def policy_from(d: dict) -> ExitPolicy:
    fields = set(ExitPolicy.__dataclass_fields__)
    return ExitPolicy(**{k: v for k, v in d.items() if k in fields})


def replay_per_fold(cache_dir, ohlcv, mp, ma, cb, policy, k,
                    extra: pd.DataFrame | None = None,
                    overlay_col: str | None = None,
                    overlay_exclude_q: float = 0.0) -> pd.DataFrame:
    """Replay cached predictions fold-by-fold; return per-fold metrics.

    ``extra`` (date,ticker,+cols) is merged onto each fold so overlay columns not
    in the prediction cache (e.g. idio_vol_60d) become available for filtering.
    """
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "meta.json").read_text())
    rows = []
    for fm in meta:
        td = pd.read_parquet(cache_dir / f"fold_{fm['fold']:02d}.parquet")
        td["date"] = pd.to_datetime(td["date"])
        if extra is not None:
            td = td.merge(extra, on=["date", "ticker"], how="left", suffixes=("", "_x"))
        ev = _evaluate_fold(
            td, ohlcv, (pd.Timestamp(fm["test_start"]), pd.Timestamp(fm["test_end"])),
            mp, ma, policy, cb, False,
            overlay_col=overlay_col, overlay_exclude_q=overlay_exclude_q, top_k=k,
        )
        rows.append({"fold": fm["fold"], "total_return": ev["total_return"],
                     "sharpe": ev["sharpe"], "win_rate": ev["win_rate"],
                     "max_dd": ev["max_dd"], "n_trades": ev["n_trades"],
                     "market_return": ev["market_return"]})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, label: str) -> None:
    bs = one_sample_bootstrap(df["total_return"].values)
    wr = (df["win_rate"] * df["n_trades"]).sum() / df["n_trades"].sum()
    pos = int((df["total_return"] > 0).sum())
    print(f"  {label:22s} ret {bs['mean']:+6.1%} "
          f"[95% {bs['ci_lo']:+6.1%},{bs['ci_hi']:+6.1%}]  "
          f"WR {wr:4.0%}  +folds {pos}/{len(df)}  "
          f"meanSharpe {df['sharpe'].mean():+5.2f}  worstDD {df['max_dd'].min():+6.1%}")


def main() -> None:
    decision = json.loads(DECISION_FP.read_text())
    mp, ma, cb = decision["min_price"], decision["min_adv_usd"], decision["cost_bps"]
    pol = policy_from(decision["exit_policy_adaptive"])
    ohlcv = pd.read_parquet(ROOT / "data/processed/ohlcv_smallcap.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    base8 = replay_per_fold(CACHE_ROOT / "v4_nometa", ohlcv, mp, ma, cb, pol, 8)
    base6 = replay_per_fold(CACHE_ROOT / "v4_nometa", ohlcv, mp, ma, cb, pol, 6)

    print("\n  Monte Carlo sanity — deployed strategy, adaptive6_pt40, "
          f"{cb:g}bps, 10k bootstrap\n")
    summarize(base8, "baseline top-8 (prod)")
    summarize(base6, "baseline top-6")

    pb = paired_bootstrap(base6["total_return"].values, base8["total_return"].values)
    print(f"\n  top-6 vs top-8: mean diff {pb['mean_diff']:+.1%} "
          f"[95% {pb['ci_lo']:+.1%}, {pb['ci_hi']:+.1%}]  "
          f"p(diff<=0)={pb['p_le0']:.2f}")
    verdict = ("REAL (CI excludes 0)" if pb["ci_lo"] > 0
               else "NOT distinguishable from noise (CI spans 0)")
    print(f"  -> top-6 edge is {verdict}")
    print(f"  PSR baseline top-8 (vs SR=0): {prob_sharpe(base8['total_return'].values):.3f}")


if __name__ == "__main__":
    main()
