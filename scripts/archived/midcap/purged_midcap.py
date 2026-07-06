"""Re-measure the mid-cap 20d swing (backtest_midcap) WITH purging — the June
result (+8.3%/fold, IC 0.053, CI>0) was computed on the unpurged harness and is
therefore inflated by the same 20d label look-ahead quantified in 45 for
small-caps (~-9pp/fold there). Same config as backtest_midcap.py (24 features,
LambdaRank date-relative, top-8 pt40, 15bps+spread, 250 trees), purge_days=20.

Compares purged vs the existing unpurged cache (data/midcap/cache/midcap_swing)
via the MC paired bootstrap.

Usage:
    PYTHONPATH=src python scripts/midcap/purged_midcap.py
"""
from __future__ import annotations

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

from _v3_harness import V2_FEATURES_BASE, ExitPolicy, run_walkforward  # noqa: E402

_spec = importlib.util.spec_from_file_location("mc", ROOT / "scripts/v3/34_mc_validate.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

DATA = ROOT / "data" / "midcap"
DEC = json.loads((ROOT / "data/v3_benchmarks/v4_filter_decision.json").read_text())
MP, MA, CB = DEC["min_price"], DEC["min_adv_usd"], DEC["cost_bps"]


def main() -> None:
    t0 = time.time()
    feats = pd.read_parquet(DATA / "features_midcap.parquet")
    feats["date"] = pd.to_datetime(feats["date"])
    ohlcv = pd.read_parquet(DATA / "ohlcv_midcap.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    base = [f for f in V2_FEATURES_BASE if f in feats.columns]

    pol_fields = set(ExitPolicy.__dataclass_fields__)
    policy = ExitPolicy(**{k: v for k, v in DEC["exit_policy"].items() if k in pol_fields})

    cache_pg = DATA / "cache" / "midcap_swing_purged"
    if not (cache_pg / "meta.json").exists():
        print("=== training midcap PURGED (purge_days=20, 250 trees) ===")
        run_walkforward(
            feats, ohlcv, base, config_name="midcap_swing_purged",
            cache_dir=cache_pg, objective_lambdarank=True, num_boost_round=250,
            min_price=MP, min_adv_usd=MA, exit_policy=policy, cost_bps=CB,
            use_spread_cost=True, purge_days=20, verbose=True,
        )
    else:
        print("(purged cache exists, skipping training)")

    print("\n  midcap purged vs unpurged — replay, 15bps+spread, paired bootstrap")
    for label, cache in [("unpurged", DATA / "cache" / "midcap_swing"),
                         ("purged", cache_pg)]:
        df = mc.replay_per_fold(cache, ohlcv, MP, MA, CB, policy, 8)
        mo = (1 + df["total_return"].mean()) ** (1 / 3) - 1
        mc.summarize(df, label)
        print(f"    monthly {mo:+.2%}")
        if label == "unpurged":
            up = df
        else:
            pg = df
    pb = mc.paired_bootstrap(pg["total_return"].values, up["total_return"].values)
    print(f"\n  Dret purged-unpurged {pb['mean_diff']:+.1%} "
          f"[{pb['ci_lo']:+.1%},{pb['ci_hi']:+.1%}] p(<=0)={pb['p_le0']:.3f}")
    # Purged tradable IC from the freshly written meta
    meta = json.loads((cache_pg / "meta.json").read_text())
    print(f"  purged IC (full xsec): {np.mean([f['mean_ic'] for f in meta]):+.4f}")
    print(f"  Runtime {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
