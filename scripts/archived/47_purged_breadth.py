"""V4.1 experiment — BREADTH (portfolio width) on the PURGED prediction cache.

Hypothesis: IR ≈ IC·√breadth. If the purged IC decays slowly with rank, widening
top-8 → top-10/12/16 adds independent bets: per-name alpha falls but portfolio
variance falls faster → Sharpe up, and capacity/impact per name improves. The
honest counter-force: deeper picks are less liquid (wider spreads) and lower
ranked, so the spread-aware cost model must confirm whatever flat 15bps says.

Replay-only (no retraining): _evaluate_fold(top_k=K) on v4_purge_purged, prod
exit pt40, both cost models.

GATE (all required, judged at flat 15bps AND spread-aware):
  1. paired bootstrap vs K=8: p(diff<=0) after Šidák (SESSION_TRIALS) < 0.05
  2. SELECT (folds 1-12) / CONFIRM (folds 13-16): confirm delta > 0
  3. DSR of the candidate's fold returns > 0.95
Sharpe delta is reported (the mechanism predicts it rises) but the gate is on
returns — a Sharpe-only "win" with lower return would need a leverage story
that small-cap margin costs don't support.

Usage:
    PYTHONPATH=src python scripts/v3/47_purged_breadth.py
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

from _v3_harness import _evaluate_fold  # noqa: E402

_spec = importlib.util.spec_from_file_location("mc", ROOT / "scripts/v3/34_mc_validate.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

CACHE = ROOT / "data" / "v3_benchmarks" / "cache" / "v4_purge_purged"
DEC = json.loads((ROOT / "data/v3_benchmarks/v4_filter_decision.json").read_text())
MP, MA, CB = DEC["min_price"], DEC["min_adv_usd"], DEC["cost_bps"]
SESSION_TRIALS = 7
SELECT_FOLDS = 12
KS = [8, 10, 12, 16]  # 8 = baseline


def replay_k(meta, ohlcv, pol, k, spread):
    rows = []
    for fm in meta:
        td = pd.read_parquet(CACHE / f"fold_{fm['fold']:02d}.parquet")
        td["date"] = pd.to_datetime(td["date"])
        ev = _evaluate_fold(
            td, ohlcv,
            (pd.Timestamp(fm["test_start"]), pd.Timestamp(fm["test_end"])),
            MP, MA, pol, CB, spread, top_k=k)
        rows.append({"fold": fm["fold"], **{key: ev[key] for key in
                     ("total_return", "sharpe", "win_rate", "n_trades",
                      "max_dd", "avg_candidates")}})
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    pol = mc.policy_from(DEC["exit_policy"])
    ohlcv = pd.read_parquet(ROOT / "data/processed/ohlcv_smallcap.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    meta = json.loads((CACHE / "meta.json").read_text())

    print(f"\n  47 — breadth on PURGED cache (exit pt40, Sidak M={SESSION_TRIALS})")
    gate_by_cost = {}
    for label, spread in [("flat 15bps", False), ("spread-aware", True)]:
        print(f"\n  --- cost {label} ---")
        dfs = {k: replay_k(meta, ohlcv, pol, k, spread) for k in KS}
        base = dfs[8]
        cand = float(base["avg_candidates"].mean())
        mo = (1 + base["total_return"].mean()) ** (1 / 3) - 1
        print(f"  K= 8 (base)  {base['total_return'].mean():+7.1%}/f {mo:+6.2%}/mo "
              f"Sh{base['sharpe'].mean():5.2f}  (avg candidates/rebalance {cand:.0f})")
        stats = {}
        for k in KS[1:]:
            df = dfs[k]
            pb = mc.paired_bootstrap(df["total_return"].values, base["total_return"].values)
            p_sidak = mc.sidak(pb["p_le0"], SESSION_TRIALS)
            sel = (df["total_return"][:SELECT_FOLDS] - base["total_return"][:SELECT_FOLDS]).mean()
            conf = (df["total_return"][SELECT_FOLDS:] - base["total_return"][SELECT_FOLDS:]).mean()
            mo = (1 + df["total_return"].mean()) ** (1 / 3) - 1
            dsh = df["sharpe"].mean() - base["sharpe"].mean()
            stats[k] = (pb, p_sidak, sel, conf)
            print(f"  K={k:2d}         {df['total_return'].mean():+7.1%}/f {mo:+6.2%}/mo "
                  f"Sh{df['sharpe'].mean():5.2f} (dSh{dsh:+5.2f})  "
                  f"d{pb['mean_diff']:+.1%}[{pb['ci_lo']:+.1%},{pb['ci_hi']:+.1%}] "
                  f"pSidak={p_sidak:.2f} sel{sel:+.1%} conf{conf:+.1%}")
        best = max(stats, key=lambda k: stats[k][2])  # best SELECT delta
        pb, p_sidak, sel, conf = stats[best]
        trial_srs = [dfs[k]["total_return"].mean() / dfs[k]["total_return"].std() for k in KS]
        dsr = mc.def_sharpe(dfs[best]["total_return"].values,
                            float(np.std(trial_srs)), SESSION_TRIALS)
        ok = p_sidak < 0.05 and conf > 0 and dsr > 0.95
        gate_by_cost[label] = ok
        print(f"  SELECT winner K={best}: pSidak={p_sidak:.3f} conf{conf:+.1%} "
              f"DSR={dsr:.3f} -> {'pass' if ok else 'fail'}")

    ok = all(gate_by_cost.values())
    print(f"\n  GATE (both cost models): "
          f"{'PROMOTE' if ok else 'REJECT — wider K does not beat top-8 after the haircut'}")
    print(f"  Runtime {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
