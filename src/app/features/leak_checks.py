"""Anti-leakage gate — run automatically by every production retrain before the
model is saved (and by scripts/v3/18_verify_no_leak.py on demand).

A feature that is legitimately known at T 00:00 has only a weak relation to the
20-day forward target (|Pearson| ~0-0.05 in practice); a leaked one (a label in
disguise, a future-looking merge) stands out. Checks:
  1. no label-like names (fwd_*, future_*, actual_ret*, tb_label*, *_positive)
  2. no degenerate column (< 2 distinct values)
  3. |Pearson r| with the target < 0.10 on the pooled panel (sampled)
  4. |mean per-date Spearman rho| (the IC) with the target < 0.25 (sampled dates).
     The SIGNED mean, not the median |rho|: an exposure such as beta or volatility
     has a large daily |rho| whose sign flips with the market's direction (beta
     ~0.29, vol ~0.21 on the S&P 500), while a leak is consistently signed.
     Calibrated 2026-10 on both production feature sets: legitimate features reach
     |mean IC| ≤ 0.12 (small-cap volatility, S&P 12-month momentum ~0.10); a planted
     noisy copy of the target scores 0.74. 0.25 leaves margin on both sides.
Returns a list of human-readable problems; empty = pass.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MAX_PEARSON = 0.10
MAX_MEAN_DAILY_IC = 0.25
BANNED_PREFIXES = ("fwd_", "forward_", "future_", "actual_ret", "tb_label")
BANNED_SUFFIXES = ("_positive", "_xsec_positive")


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def check_names(feats: list[str], target: str) -> list[str]:
    errors = [f"target {target!r} is in the feature list"] if target in feats else []
    for f in feats:
        if f.startswith(BANNED_PREFIXES):
            errors.append(f"{f!r}: label-like prefix")
        if f.endswith(BANNED_SUFFIXES):
            errors.append(f"{f!r}: label-like suffix (binary of a target)")
    return errors


def check_degenerate(df: pd.DataFrame, feats: list[str]) -> list[str]:
    return [f"{f}: degenerate (< 2 distinct values)" for f in feats
            if f in df.columns and _num(df[f]).dropna().nunique() < 2]


def check_pearson(df: pd.DataFrame, feats: list[str], target: str) -> list[str]:
    errors, y = [], _num(df[target])
    for f in feats:
        if f not in df.columns:
            continue
        x = _num(df[f])
        m = x.notna() & y.notna()
        if m.sum() < 1000 or x[m].nunique() < 2:
            continue
        r = float(np.corrcoef(x[m], y[m])[0, 1])
        if abs(r) > MAX_PEARSON:
            errors.append(f"{f}: |Pearson r|={abs(r):.3f} > {MAX_PEARSON} — leak suspected")
    return errors


def check_per_date_spearman(df: pd.DataFrame, feats: list[str], target: str,
                            n_dates: int = 60, seed: int = 42) -> list[str]:
    dates = pd.Series(df["date"].unique())
    sample = set(dates.sample(min(n_dates, len(dates)), random_state=seed))
    groups = [g for d, g in df[df["date"].isin(sample)].groupby("date") if len(g) >= 30]
    errors = []
    for f in feats:
        if f not in df.columns:
            continue
        ics = []
        for g in groups:
            x, y = _num(g[f]), _num(g[target])
            m = x.notna() & y.notna()
            if m.sum() >= 10 and x[m].nunique() > 1:
                rho = spearmanr(x[m], y[m])[0]
                if np.isfinite(rho):
                    ics.append(rho)
        if ics and abs(float(np.mean(ics))) > MAX_MEAN_DAILY_IC:
            errors.append(f"{f}: |mean daily IC|={abs(np.mean(ics)):.3f} "
                          f"> {MAX_MEAN_DAILY_IC} — leak suspected")
    return errors


def verify(train: pd.DataFrame, feats: list[str], target: str,
           max_rows: int = 200_000, seed: int = 42) -> list[str]:
    """All checks on the rows the model is about to be trained on."""
    errors = check_names(feats, target)
    rows = train.dropna(subset=[target])
    errors += check_degenerate(rows, feats)
    sample = rows.sample(min(max_rows, len(rows)), random_state=seed) if len(rows) else rows
    errors += check_pearson(sample, feats, target)
    errors += check_per_date_spearman(sample, feats, target)
    return errors


def assert_no_leak(train: pd.DataFrame, feats: list[str], target: str) -> None:
    """Raise (failing the run and keeping the previous model) if any check fails."""
    errors = verify(train, feats, target)
    if errors:
        raise RuntimeError("anti-leak gate FAILED — model not saved:\n  " + "\n  ".join(errors))
    print(f"  ✓ anti-leak gate passed ({len(feats)} features)")
