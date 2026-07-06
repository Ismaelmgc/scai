"""V4 Phase 2: exit-policy sweep on cached predictions.

Replays the v4_nometa prediction cache (created by 20_filter_sweep.py)
through a grid of exit policies under the chosen tradability filter.
No retraining — each grid point takes seconds.

Guardrails (consistency-first): a policy only beats the baseline if
    mean_return >= 0.95 x baseline   AND
    mean_sharpe >= baseline          AND
    folds_positive_ret >= baseline
WR is reported (with trade counts) but never optimized in isolation —
profit targets mechanically inflate WR while capping the right tail.

Usage:
    PYTHONPATH=src python scripts/v3/21_exit_sweep.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "v3"))

from _v3_harness import (  # noqa: E402
    ExitPolicy, replay_walkforward, save_result, print_comparison,
)

CACHE_DIR = ROOT / "data" / "v3_benchmarks" / "cache" / "v4_nometa"
DECISION_FP = ROOT / "data" / "v3_benchmarks" / "v4_filter_decision.json"


def build_grid() -> list[tuple[str, ExitPolicy]]:
    """Exit-policy grid. Names become v4_exit_<name>.json files."""
    grid: list[tuple[str, ExitPolicy]] = [
        # The live engine's two strategies, under honest filter+costs:
        ("baseline_trail", ExitPolicy()),
        ("adaptive6_d5", ExitPolicy(adaptive_tighten=0.06)),
        # Adaptive variants
        ("adaptive8_d5", ExitPolicy(adaptive_tighten=0.08)),
        ("adaptive6_d3", ExitPolicy(adaptive_tighten=0.06, adaptive_after_days=3)),
        ("adaptive6_d8", ExitPolicy(adaptive_tighten=0.06, adaptive_after_days=8)),
        # Profit targets (reported for completeness; WR-inflating)
        ("pt15", ExitPolicy(profit_target=0.15)),
        ("pt25", ExitPolicy(profit_target=0.25)),
        ("pt40", ExitPolicy(profit_target=0.40)),
        ("adaptive6_pt25", ExitPolicy(adaptive_tighten=0.06, profit_target=0.25)),
        ("adaptive6_pt40", ExitPolicy(adaptive_tighten=0.06, profit_target=0.40)),
        # Breakeven stops
        ("be5", ExitPolicy(breakeven_after=0.05)),
        ("be8", ExitPolicy(breakeven_after=0.08)),
        ("be10", ExitPolicy(breakeven_after=0.10)),
        ("adaptive6_be8", ExitPolicy(adaptive_tighten=0.06, breakeven_after=0.08)),
        # Time stops
        ("ts10", ExitPolicy(time_stop=10)),
        ("ts15", ExitPolicy(time_stop=15)),
        ("adaptive6_ts15", ExitPolicy(adaptive_tighten=0.06, time_stop=15)),
        # Wider/tighter trails
        ("trail_8_12", ExitPolicy(trail_min=0.08, trail_max=0.12)),
        ("trail_12_20", ExitPolicy(trail_min=0.12, trail_max=0.20)),
    ]
    return grid


def main() -> None:
    t0 = time.time()
    decision = json.loads(DECISION_FP.read_text())
    min_price, min_adv, cost_bps = (decision["min_price"],
                                    decision["min_adv_usd"],
                                    decision["cost_bps"])
    print(f"Filter: price>=${min_price:g}, ADV>=${min_adv:,}, cost={cost_bps:g}bps/side")

    ohlcv = pd.read_parquet(ROOT / "data/processed/ohlcv_smallcap.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    baseline = replay_walkforward(
        CACHE_DIR, ohlcv, config_name="v4_filt_baseline",
        min_price=min_price, min_adv_usd=min_adv, cost_bps=cost_bps,
    )
    base_agg = baseline.aggregate()
    base_pos = int(base_agg["folds_positive_ret"].split("/")[0])
    print(f"Baseline: ret={base_agg['mean_return']:+.1%} sharpe={base_agg['mean_sharpe']:+.2f} "
          f"WR={base_agg['mean_win_rate']:.1%} +folds={base_agg['folds_positive_ret']}")

    results = []
    for name, policy in build_grid():
        res = replay_walkforward(
            CACHE_DIR, ohlcv, config_name=f"v4_exit_{name}",
            min_price=min_price, min_adv_usd=min_adv,
            exit_policy=policy, cost_bps=cost_bps,
        )
        save_result(res)
        agg = res.aggregate()
        n_pos = int(agg["folds_positive_ret"].split("/")[0])
        passes = (
            agg["mean_return"] >= 0.95 * base_agg["mean_return"]
            and agg["mean_sharpe"] >= base_agg["mean_sharpe"]
            and n_pos >= base_pos
        )
        results.append((name, policy, res, agg, passes))
        flag = "PASS" if passes else "    "
        print(f"  {flag} {name:18s} ({policy.describe():40s}) "
              f"ret={agg['mean_return']:+7.1%} sharpe={agg['mean_sharpe']:+5.2f} "
              f"WR={agg['mean_win_rate']:5.1%} +folds={agg['folds_positive_ret']} "
              f"maxDD={min(f.max_dd for f in res.folds):+.1%}")

    passing = [r for r in results if r[4]]
    if passing:
        # Among guardrail-passers, prefer highest Sharpe (consistency),
        # tie-break on WR as the user's aspiration.
        passing.sort(key=lambda r: (round(r[3]["mean_sharpe"], 2), r[3]["mean_win_rate"]),
                     reverse=True)
        w_name, w_policy, w_res, w_agg, _ = passing[0]
        print(f"\nWINNER: {w_name} ({w_policy.describe()})")
        print(f"  ret={w_agg['mean_return']:+.1%} sharpe={w_agg['mean_sharpe']:+.2f} "
              f"WR={w_agg['mean_win_rate']:.1%} vs baseline WR={base_agg['mean_win_rate']:.1%}")
        print_comparison(baseline, w_res)
        decision["exit_policy"] = {
            "name": w_name,
            "trail_mult": w_policy.trail_mult,
            "trail_min": w_policy.trail_min, "trail_max": w_policy.trail_max,
            "adaptive_tighten": w_policy.adaptive_tighten,
            "adaptive_after_days": w_policy.adaptive_after_days,
            "profit_target": w_policy.profit_target,
            "breakeven_after": w_policy.breakeven_after,
            "time_stop": w_policy.time_stop,
        }
        DECISION_FP.write_text(json.dumps(decision, indent=2))
        print(f"Exit decision appended to {DECISION_FP}")
    else:
        print("\nNo policy passed the guardrails — keep the baseline trailing stop.")

    print(f"Total runtime: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
