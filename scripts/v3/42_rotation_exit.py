"""V4 experiment: score-rotation exit — cut held losers whose model conviction
decays and redeploy the slot into the best fresh signal.

Motivation (from live paper trading): with a fixed 20d hold + trailing stop, a
slowly-bleeding name that never trips the trailing stop occupies a slot for ~20d,
blocking capital that a fresh top-ranked signal could use. The production backtest
harness can't test this — it simulates INDEPENDENT overlapping 20d cohorts, not a
real 8-slot portfolio. So this is a proper slot-based portfolio simulator.

Baseline (production-like): 8 equal-weight slots, enter top-8 by model score,
hold with the trailing/PT/(adaptive)/20d exits, rebalance every 5d refilling only
FREED slots. No rank exit.

Rotation: same, plus a hysteresis rank exit — at each rebalance, drop any held
name whose score-rank has fallen out of the top-N_hold tradable (N_hold >= K), and
refill freed slots with the best non-held names. N_hold is the stickiness band:
N_hold=K → full rotation, larger → stickier.

Predictions/prices/atr/spread come from the cached daily predictions of the
production 28-feature model (v4_fs_base28). Costs = 15bps round-trip or the
measured spread, whichever larger. MC paired bootstrap of per-fold return vs the
matching baseline.

Usage:
    PYTHONPATH=src python scripts/v3/42_rotation_exit.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "v3"))

_spec = importlib.util.spec_from_file_location("mc", ROOT / "scripts/v3/34_mc_validate.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

CACHE = ROOT / "data" / "v3_benchmarks" / "cache" / "v4_fs_base28"
DECISION = json.loads((ROOT / "data/v3_benchmarks/v4_filter_decision.json").read_text())
MP, MA, CB = DECISION["min_price"], DECISION["min_adv_usd"], DECISION["cost_bps"]
TOP_K, REBALANCE_EVERY = 8, 5


@dataclass
class Policy:
    trail_mult: float = 5.3
    trail_min: float = 0.10
    trail_max: float = 0.16
    adaptive_tighten: float | None = None
    adaptive_after_days: int = 5
    profit_target: float | None = 0.40
    time_stop: int = 20


def pol(d: dict) -> Policy:
    f = {k: d[k] for k in Policy.__dataclass_fields__ if k in d}
    return Policy(**f)


def load_fold(i: int):
    """Per-date dict: date -> DataFrame indexed by ticker (pred, close, atr, adv, spread)."""
    df = pd.read_parquet(CACHE / f"fold_{i:02d}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"atr_pct_20d": "atr", "adv_usd_20d": "adv", "cs_spread_20d": "spread"})
    by_date = {}
    for d, g in df.groupby("date"):
        g = g.dropna(subset=["close"]).drop_duplicates("ticker").set_index("ticker")
        by_date[d] = g
    return sorted(by_date), by_date


def _roundtrip_cost(spread: float) -> float:
    c = 2 * CB / 1e4
    if np.isfinite(spread) and spread > 0:
        c = max(c, min(float(spread), 0.10))
    return c


def simulate_fold(dates, by_date, policy: Policy, n_hold: int | None,
                  rank_every: int = REBALANCE_EVERY) -> dict:
    slots: dict[str, dict] = {}
    daily_port, closed, held_days = [], [], []
    n_entries = 0

    def tradable_ranked(g: pd.DataFrame) -> list[str]:
        t = g[(g["close"] >= MP) & (g["adv"] >= MA)]
        return t.sort_values("pred", ascending=False).index.tolist()

    for i, d in enumerate(dates):
        g = by_date[d]
        ranked = tradable_ranked(g)
        rank_of = {tk: r for r, tk in enumerate(ranked, 1)}
        slot_rets, cost_today = [], 0.0

        # 1) mark-to-market + exits for held positions
        for tk in list(slots):
            pos = slots[tk]
            px = float(g.at[tk, "close"]) if tk in g.index else pos["last"]
            slot_rets.append(px / pos["last"] - 1 if pos["last"] else 0.0)
            pos["last"] = px
            pos["peak"] = max(pos["peak"], px)
            pos["days"] += 1
            entry = pos["entry"]
            reason = None
            if policy.profit_target and px >= entry * (1 + policy.profit_target):
                reason = "pt"
            eff = pos["trail"]
            if (policy.adaptive_tighten and pos["days"] > policy.adaptive_after_days
                    and px > entry):
                eff = min(eff, policy.adaptive_tighten)
            if reason is None and (px - pos["peak"]) / pos["peak"] <= -eff:
                reason = "trail"
            if reason is None and pos["days"] >= policy.time_stop:
                reason = "time"
            if (reason is None and n_hold is not None and i % rank_every == 0
                    and rank_of.get(tk, 10**9) > n_hold):
                reason = "rank"
            if reason:
                cost_today += _roundtrip_cost(pos["spread"])
                closed.append(px / entry - 1 - _roundtrip_cost(pos["spread"]))
                held_days.append(pos["days"])
                del slots[tk]

        # portfolio daily return: mean over K slots (empty = cash = 0), minus
        # round-trip costs realized today (allocated 1/K per closed slot).
        daily_port.append((sum(slot_rets) - cost_today) / TOP_K)

        # 2) refill freed slots on rebalance days with best non-held names
        if i % REBALANCE_EVERY == 0 and len(slots) < TOP_K:
            for tk in ranked:
                if len(slots) >= TOP_K:
                    break
                if tk in slots:
                    continue
                row = g.loc[tk]
                atr = float(row["atr"]) if np.isfinite(row["atr"]) and row["atr"] > 0 else 0.03
                slots[tk] = {
                    "entry": float(row["close"]), "last": float(row["close"]),
                    "peak": float(row["close"]), "days": 0,
                    "trail": float(np.clip(atr * policy.trail_mult, policy.trail_min, policy.trail_max)),
                    "spread": float(row["spread"]),
                }
                n_entries += 1

    # liquidate open positions at fold end (charge round-trip)
    for pos in slots.values():
        closed.append(pos["last"] / pos["entry"] - 1 - _roundtrip_cost(pos["spread"]))
        held_days.append(pos["days"])

    s = pd.Series(daily_port)
    eq = (1 + s).cumprod()
    total = float(eq.iloc[-1]) - 1 if len(s) else 0.0
    sharpe = float(s.mean() / s.std() * np.sqrt(252)) if s.std() > 0 else 0.0
    maxdd = float(((eq - eq.cummax()) / eq.cummax()).min()) if len(s) else 0.0
    return {
        "total_return": total, "sharpe": sharpe, "max_dd": maxdd,
        "wr": float(np.mean([c > 0 for c in closed])) if closed else 0.0,
        "n_trades": len(closed),
        "avg_hold": float(np.mean(held_days)) if held_days else 0.0,
    }


def run(policy: Policy, n_hold: int | None, rank_every: int = REBALANCE_EVERY) -> pd.DataFrame:
    rows = []
    for i in range(1, 17):
        dates, by_date = load_fold(i)
        if not dates:
            continue
        rows.append({"fold": i, **simulate_fold(dates, by_date, policy, n_hold, rank_every)})
    return pd.DataFrame(rows)


def wr(df):
    return (df["wr"] * df["n_trades"]).sum() / df["n_trades"].sum()


def pline(name: str, df: pd.DataFrame, base: pd.DataFrame | None = None) -> None:
    cols = (f"  {name:22} {df['total_return'].mean():+8.1%} {df['sharpe'].mean():6.2f} "
            f"{wr(df):5.0%} {df['max_dd'].mean():6.1%} {df['avg_hold'].mean():5.1f} "
            f"{int(df['n_trades'].sum()):>5}")
    if base is not None:
        pb = mc.paired_bootstrap(df["total_return"].values, base["total_return"].values)
        flag = "  <-CI>0" if pb["ci_lo"] > 0 else ""
        cols += f"  {pb['mean_diff']:+.1%} [{pb['ci_lo']:+.1%},{pb['ci_hi']:+.1%}]{flag}"
    print(cols)


def main() -> None:
    t0 = time.time()
    base_pol = pol(DECISION["exit_policy"])              # pt40
    adapt_pol = pol(DECISION["exit_policy_adaptive"])    # adaptive6_pt40

    print(f"\n  Score-rotation exit — slot portfolio (K={TOP_K}, reb={REBALANCE_EVERY}d, "
          f"{CB:g}bps+spread, filter ${MP}/{MA/1e3:.0f}K)\n")
    print(f"  {'variant':22} {'ret/fold':>8} {'Sharpe':>6} {'WR':>5} {'maxDD':>6} "
          f"{'hold':>5} {'trd':>5} {'diff vs base [95% CI]':>24}")

    base = run(base_pol, None)
    pline("BASE pt40", base)
    for name, nh, rev in [("ROT n_hold=8", 8, REBALANCE_EVERY),
                          ("ROT n_hold=12", 12, REBALANCE_EVERY),
                          ("ROT n_hold=16", 16, REBALANCE_EVERY),
                          ("ROT n_hold=24", 24, REBALANCE_EVERY),
                          ("ROT n_hold=12 daily", 12, 1)]:
        pline(name, run(base_pol, nh, rev), base)

    ad = run(adapt_pol, None)
    print()
    pline("BASE adaptive6_pt40", ad)
    pline("ROT adapt n_hold=16", run(adapt_pol, 16), ad)

    print(f"\n  CI>0 = rotation beats its baseline.  Runtime {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
