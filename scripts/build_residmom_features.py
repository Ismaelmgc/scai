"""Build residual (idiosyncratic) momentum + residual short-term reversal.

Round-1 finding: raw 12-1 momentum (mom_12_1) nearly doubled the model's ranking
IC but DEGRADED the top-8 book — high-momentum small-caps cluster in the tail and
crash. The documented fix (Blitz-Hanauer-Vidojevic; Huij-Lansdorp) is RESIDUAL
momentum: the same signal computed on market-model residuals, which keeps the
return but kills the crash (~doubles the Sharpe of total-return momentum).

These are genuinely absent from the 517-col parquet: `mom_mktadj_*` (round 1) only
SUBTRACTS the index (implicit beta=1); it does not beta-adjust nor standardize by
residual vol. Here we estimate a trailing market beta per ticker, take daily
residuals e = ret - beta*mkt, and build:
  residmom_12_1 : standardized cum-residual over (t-252 .. t-21), skips last month
  residmom_6_1  : same over (t-126 .. t-21), horizon closer to our 20d target
  rev_resid_21d : NEGATIVE standardized cum-residual over the last 21d (reversal)
  rev_21d       : NEGATIVE raw 21d return (plain short-term reversal)
All standardized (mean/std of residuals) so they are cross-sectionally comparable
and the LambdaRank can split on raw values per date. PIT-safe (trailing only; the
momentum windows skip the most recent 21 sessions).

52-week-high proximity (pct_from_52w_high) and idio_vol_60d already exist in the
features parquet (unused) — they are pulled directly in scripts/v3/33_*.py, not here.

Writes data/residmom_features.parquet (date, ticker, + 4 new cols).

Usage:
    PYTHONPATH=src python scripts/build_residmom_features.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "residmom_features.parquet"

NEW_FEATURES = ["residmom_12_1", "residmom_6_1", "rev_resid_21d", "rev_21d"]

BETA_WIN, BETA_MINP = 252, 150          # trailing window for the market beta
SKIP = 21                               # skip the most recent ~1 month (12-1, 6-1)


def main() -> None:
    oh = pd.read_parquet(ROOT / "data/processed/ohlcv_smallcap.parquet")
    oh["date"] = pd.to_datetime(oh["date"])
    oh = oh.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"OHLCV: {len(oh):,} rows, {oh['ticker'].nunique()} tickers")

    spy = pd.read_parquet(ROOT / "data/processed/smallcap_spy.parquet")[["date", "close"]]
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date")
    spy["mkt"] = spy["close"].pct_change()
    oh = oh.merge(spy[["date", "mkt"]], on="date", how="left")

    g = oh.groupby("ticker", group_keys=False)
    oh["ret"] = g["close"].transform(lambda x: x.pct_change())
    oh["ret_mkt"] = oh["ret"] * oh["mkt"]
    oh["mkt2"] = oh["mkt"] ** 2

    # ── trailing market beta via rolling moments (cov/var), vectorized ──
    def roll_mean(col: str) -> pd.Series:
        return oh.groupby("ticker")[col].transform(
            lambda x: x.rolling(BETA_WIN, min_periods=BETA_MINP).mean())

    m_ret, m_mkt = roll_mean("ret"), roll_mean("mkt")
    cov = roll_mean("ret_mkt") - m_ret * m_mkt
    var = (roll_mean("mkt2") - m_mkt ** 2).replace(0, np.nan)
    beta = cov / var
    oh["e"] = oh["ret"] - beta * oh["mkt"]      # daily market-model residual

    # ── standardized residual momentum (mean/std of residuals over the window) ──
    for name, win, minp in [("residmom_12_1", 231, 120), ("residmom_6_1", 105, 60)]:
        e_sum = oh.groupby("ticker")["e"].transform(
            lambda x: x.rolling(win, min_periods=minp).sum().shift(SKIP))
        e_std = oh.groupby("ticker")["e"].transform(
            lambda x: x.rolling(win, min_periods=minp).std().shift(SKIP))
        oh[name] = (e_sum / win) / e_std.replace(0, np.nan)

    # ── residual short-term reversal (last 21d, NOT skipped) ──
    e_sum21 = oh.groupby("ticker")["e"].transform(
        lambda x: x.rolling(21, min_periods=10).sum())
    e_std21 = oh.groupby("ticker")["e"].transform(
        lambda x: x.rolling(21, min_periods=10).std())
    oh["rev_resid_21d"] = -((e_sum21 / 21) / e_std21.replace(0, np.nan))

    # ── plain short-term reversal (raw 21d return) ──
    c21 = oh.groupby("ticker")["close"].transform(lambda x: x.shift(21))
    oh["rev_21d"] = -(oh["close"] / c21 - 1)

    res = oh[["date", "ticker"] + NEW_FEATURES].replace([np.inf, -np.inf], np.nan)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT, index=False)
    print(f"Saved {len(res):,} rows x {len(NEW_FEATURES)} new features -> {OUT}")
    print("Non-null coverage:")
    for f in NEW_FEATURES:
        print(f"  {f:18s} {res[f].notna().mean():.1%}")


if __name__ == "__main__":
    main()
