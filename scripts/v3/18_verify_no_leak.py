"""Anti-leak verification on demand — the same gate every production retrain runs
automatically before saving a model (app.features.leak_checks).

    PYTHONPATH=src python scripts/v3/18_verify_no_leak.py                 # small-cap model
    PYTHONPATH=src python scripts/v3/18_verify_no_leak.py --book liquidcap
Exit code 0 = pass; 1 = leak suspected (do NOT promote the model).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "liquidcap"))

from app.features.leak_checks import verify  # noqa: E402


def smallcap() -> tuple[pd.DataFrame, list[str], str]:
    import daily_pipeline as dp
    feats = dp.V2_FEATURES + dp.V2_EDGAR_FEATURES
    df = pd.read_parquet(ROOT / "data/processed/features_smallcap.parquet")
    return df, feats, dp.V2_TARGET


def liquidcap() -> tuple[pd.DataFrame, list[str], str]:
    import daily_liquidcap as lc
    data = ROOT / "data" / "liquidcap"
    df = pd.read_parquet(data / "features_research.parquet",
                         columns=["date", "ticker", lc.TARGET] + lc.FS15)
    fnd = pd.read_parquet(data / "fundamentals_daily.parquet")
    df["date"], fnd["date"] = pd.to_datetime(df["date"]), pd.to_datetime(fnd["date"])
    return df.merge(fnd, on=["date", "ticker"], how="left"), lc.FEATURES, lc.TARGET


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", choices=["smallcap", "liquidcap"], default="smallcap")
    args = ap.parse_args()
    df, feats, target = smallcap() if args.book == "smallcap" else liquidcap()
    df["date"] = pd.to_datetime(df["date"])
    errors = verify(df, feats, target)
    print(f"{args.book}: {len(feats)} features vs {target}")
    for e in errors:
        print(f"  ✗ {e}")
    print("  ✓ no leak suspected" if not errors else f"  FAILED ({len(errors)} issue(s))")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
