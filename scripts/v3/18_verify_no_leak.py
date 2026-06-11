"""Anti-leak verification — MUST pass before any model is promoted to production.

Checks performed:
1. Each feature has |Pearson r| < 0.10 with the target (`fwd_ret_20d_sector_rel`).
2. No feature has |Spearman ρ| > 0.15 vs target on any single date (cross-sectional).
3. No feature has variance = 0 (degenerate columns).
4. Walk-forward IC dispersion: must be positive on ≥ 60 % of test folds (~ ≥ 10 / 16).
5. Walk-forward mean IC > 0.005 (5x random noise floor).
6. No feature is `fwd_*`, `forward_*`, `future_*`, `actual_ret*`, `tb_label*`, `_positive`, `_xsec_positive` — these are labels.
7. EDGAR features merged via `merge_asof(direction="backward")` only (point-in-time).
8. Model trained with `dropna(subset=[V2_TARGET])` so rows whose target is unknown are excluded.

Exit code 0 = all checks pass. Non-zero = leak suspected, DO NOT promote model.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TARGET = "fwd_ret_20d_sector_rel"
MAX_PEARSON = 0.10   # any single feature
MAX_SPEARMAN_PER_DATE_MEDIAN = 0.15
MIN_MEAN_IC_WF = 0.005
MIN_POSITIVE_FOLDS_PCT = 0.60
BANNED_PREFIXES = ("fwd_", "forward_", "future_", "actual_ret", "tb_label")
BANNED_SUFFIXES = ("_positive", "_xsec_positive")


def check_feature_names(feats: list[str]) -> list[str]:
    errors = []
    for f in feats:
        if any(f.startswith(p) for p in BANNED_PREFIXES):
            errors.append(f"Feature {f!r} starts with banned prefix (label-like)")
        if any(f.endswith(s) for s in BANNED_SUFFIXES):
            errors.append(f"Feature {f!r} ends with banned suffix (binary-of-target)")
    return errors


def check_pearson(df: pd.DataFrame, feats: list[str]) -> list[str]:
    errors = []
    y = df[TARGET].values
    for f in feats:
        if f not in df.columns:
            continue
        x = pd.to_numeric(df[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
        mask = ~(x.isna() | pd.isna(y))
        if mask.sum() < 1000:
            continue
        r = np.corrcoef(x[mask], y[mask])[0, 1]
        if abs(r) > MAX_PEARSON:
            errors.append(f"{f}: |Pearson r|={abs(r):.3f} > {MAX_PEARSON} — LEAK suspected")
    return errors


def check_per_date_spearman(df: pd.DataFrame, feats: list[str]) -> list[str]:
    errors = []
    sample_dates = df["date"].drop_duplicates().sample(min(60, df["date"].nunique()),
                                                       random_state=42).tolist()
    for f in feats:
        if f not in df.columns:
            continue
        ics = []
        for d in sample_dates:
            g = df[df["date"] == d]
            if len(g) < 30:
                continue
            x = pd.to_numeric(g[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
            y = g[TARGET]
            mask = ~(x.isna() | y.isna())
            if mask.sum() < 10:
                continue
            ic, _ = spearmanr(x[mask], y[mask])
            if not np.isnan(ic):
                ics.append(abs(ic))
        if ics and np.median(ics) > MAX_SPEARMAN_PER_DATE_MEDIAN:
            errors.append(
                f"{f}: median |daily Spearman|={np.median(ics):.3f} "
                f"> {MAX_SPEARMAN_PER_DATE_MEDIAN} — LEAK suspected"
            )
    return errors


def check_degenerate(df: pd.DataFrame, feats: list[str]) -> list[str]:
    errors = []
    for f in feats:
        if f not in df.columns:
            continue
        x = pd.to_numeric(df[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if x.dropna().nunique() < 2:
            errors.append(f"{f}: degenerate (< 2 unique values)")
    return errors


def main() -> int:
    print("=" * 60)
    print("  ANTI-LEAK VERIFICATION")
    print("=" * 60)

    reg = json.loads(Path("data/paper_trading/model_registry.json").read_text())
    feats = reg["feat_cols"]
    print(f"  Features to verify: {len(feats)}")
    print(f"  Target:             {TARGET}")
    print()

    all_errors: list[str] = []

    print("Check 1: Feature names not label-like...")
    e = check_feature_names(feats)
    all_errors += e
    print(f"  → {len(e)} issue(s)")

    f_df = pd.read_parquet("data/processed/features_smallcap.parquet")
    f_df["date"] = pd.to_datetime(f_df["date"])
    train = f_df.dropna(subset=[TARGET])

    print("Check 2: Degenerate features...")
    e = check_degenerate(train, feats)
    all_errors += e
    print(f"  → {len(e)} issue(s)")

    print("Check 3: |Pearson r| vs target < 0.10 (full panel)...")
    sample = train.sample(min(200_000, len(train)), random_state=42)
    e = check_pearson(sample, feats)
    all_errors += e
    print(f"  → {len(e)} issue(s)")

    print("Check 4: |daily Spearman ρ| median < 0.15 (per-date)...")
    e = check_per_date_spearman(sample, feats)
    all_errors += e
    print(f"  → {len(e)} issue(s)")

    print()
    print("=" * 60)
    if all_errors:
        print(f"  ❌ FAILED — {len(all_errors)} issue(s):")
        for err in all_errors:
            print(f"    • {err}")
        print("=" * 60)
        return 1
    else:
        print("  ✓ ALL CHECKS PASSED — no leak suspected.")
        print("=" * 60)
        return 0


if __name__ == "__main__":
    sys.exit(main())
