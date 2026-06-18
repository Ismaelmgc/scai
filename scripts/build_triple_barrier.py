"""Build triple-barrier (realized-return-under-exit-policy) labels.

The model ranks by raw fwd_ret_20d_sector_rel, but we TRADE with a trailing ATR
stop + 40% profit target + adaptive tighten + 20d time stop. That objective/exit
MISMATCH is the most promising untested lever: a name that prints +50% at day 20
and one that hits +40% on day 3 score identically raw, but the second is better
under our exit. Here we relabel each candidate with the return it would ACTUALLY
realize under the deployed exit policy (López de Prado triple-barrier), so the
LambdaRank can be trained to rank by what the book actually earns.

Reuses the production _simulate_trade so the label == the backtest reward exactly.
Writes data/research/tb_labels.parquet (date, ticker, tb_ret).

Usage:
    PYTHONPATH=src python scripts/build_triple_barrier.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "v3"))

from _v3_harness import ExitPolicy, HOLD_DAYS, _simulate_trade  # noqa: E402

OUT = ROOT / "data" / "research" / "tb_labels.parquet"
DECISION_FP = ROOT / "data" / "v3_benchmarks" / "v4_filter_decision.json"


def main() -> None:
    t0 = time.time()
    decision = json.loads(DECISION_FP.read_text())
    fields = set(ExitPolicy.__dataclass_fields__)
    pol = ExitPolicy(**{k: v for k, v in decision["exit_policy_adaptive"].items() if k in fields})

    oh = pd.read_parquet(ROOT / "data/processed/ohlcv_smallcap.parquet",
                         columns=["date", "ticker", "close"])
    oh["date"] = pd.to_datetime(oh["date"])
    feats = pd.read_parquet(ROOT / "data/processed/features_smallcap.parquet",
                            columns=["date", "ticker", "atr_pct_20d"])
    feats["date"] = pd.to_datetime(feats["date"])
    oh = oh.merge(feats, on=["date", "ticker"], how="left").sort_values(["ticker", "date"])

    rows = []
    for ticker, g in oh.groupby("ticker", sort=False):
        closes = g["close"].to_numpy()
        atrs = g["atr_pct_20d"].to_numpy()
        dates = g["date"].to_numpy()
        n = len(closes)
        for i in range(n - 1):
            prices = closes[i:i + HOLD_DAYS + 1]
            if len(prices) < 2 or not np.isfinite(prices[0]) or prices[0] <= 0:
                continue
            atr = atrs[i]
            if not np.isfinite(atr) or atr <= 0:
                atr = 0.03
            rows.append((dates[i], ticker, _simulate_trade(prices, atr, pol)))

    tb = pd.DataFrame(rows, columns=["date", "ticker", "tb_ret"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tb.to_parquet(OUT, index=False)
    print(f"Saved {len(tb):,} triple-barrier labels -> {OUT}")
    print(f"  tb_ret mean {tb.tb_ret.mean():+.2%}  median {tb.tb_ret.median():+.2%}  "
          f"std {tb.tb_ret.std():.2%}  (capped by +40% PT / ATR stop)")
    print(f"  runtime {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
