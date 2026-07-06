"""V4.1 experiment — re-validate SIZING levers on the PURGED prediction cache.

Why: 43_capital_deployment found conviction sizing (linear ∝z) was the only
winning lever (+4.4%/mo, CI>0) — but it was selected on the UNPURGED
v4_fs_base28 cache, i.e. on predictions inflated by the 20d label leak (45).
A sizing rule that leans harder into the model's confidence can only help if
the confidence is real; on leaked predictions the confidence was partly the
future itself. This re-runs the same slot simulation on v4_purge_purged.

Variants (M counted in the session-wide Šidák haircut):
  conv exp(z)   : weight ∝ exp(pred z-score at entry)
  conv linear   : weight ∝ max(z, 0.1)
  inverse-vol   : weight ∝ 1/ATR%(entry) — risk parity across slots (new)

GATE (all three required to promote):
  1. paired bootstrap (16 folds) vs equal-weight: p(diff<=0) after Šidák
     haircut for SESSION_TRIALS < 0.05
  2. SELECT/CONFIRM: variant chosen on folds 1-12 must also beat equal-weight
     on folds 13-16 (never used for selection)
  3. Deflated Sharpe Ratio of the candidate's fold returns > 0.95, deflating
     by the dispersion of trial Sharpes across the session's variants

Usage:
    PYTHONPATH=src python scripts/v3/46_purged_sizing.py
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

CACHE = ROOT / "data" / "v3_benchmarks" / "cache" / "v4_purge_purged"
DEC = json.loads((ROOT / "data/v3_benchmarks/v4_filter_decision.json").read_text())
MP, MA, CB = DEC["min_price"], DEC["min_adv_usd"], DEC["cost_bps"]
TOP_K, REBALANCE_EVERY = 8, 5
# Total configurations tried in this session's search (46: 3 sizing, 47: 3
# breadth, 48: 1 horizon) — the Šidák haircut must count ALL of them, not
# just the ones inside this script.
SESSION_TRIALS = 7
SELECT_FOLDS = 12  # folds 1-12 select, 13-16 confirm


@dataclass
class Policy:
    trail_mult: float = 5.3
    trail_min: float = 0.10
    trail_max: float = 0.16
    adaptive_tighten: float | None = None
    adaptive_after_days: int = 5
    profit_target: float | None = 0.40
    time_stop: int = 20


def load_fold(i: int):
    df = pd.read_parquet(CACHE / f"fold_{i:02d}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"atr_pct_20d": "atr", "adv_usd_20d": "adv",
                            "cs_spread_20d": "spread"})
    by_date = {}
    for d, g in df.groupby("date"):
        g = g.dropna(subset=["close"]).drop_duplicates("ticker").set_index("ticker")
        tr = g[(g["close"] >= MP) & (g["adv"] >= MA)]
        mu, sd = tr["pred"].mean(), tr["pred"].std()
        g["z"] = (g["pred"] - mu) / sd if sd > 0 else 0.0
        by_date[d] = g
    return sorted(by_date), by_date


def _cost(spread):
    c = 2 * CB / 1e4
    if np.isfinite(spread) and spread > 0:
        c = max(c, min(float(spread), 0.10))
    return c


def _conv(row, scheme):
    if scheme == "conv":
        return float(np.exp(np.clip(row["z"], -2, 10)))
    if scheme == "convlin":
        return max(float(row["z"]), 0.1)
    if scheme == "invvol":
        atr = float(row["atr"]) if np.isfinite(row["atr"]) and row["atr"] > 0 else 0.03
        return 1.0 / max(atr, 0.01)
    return 1.0


def simulate_fold(dates, by_date, policy, scheme="equal"):
    slots: dict[str, dict] = {}
    daily, closed = [], []
    for i, d in enumerate(dates):
        g = by_date[d]
        ranked = g[(g["close"] >= MP) & (g["adv"] >= MA)].sort_values(
            "pred", ascending=False)
        held = list(slots)
        if held:
            conv = np.array([slots[tk]["conv"] for tk in held])
            w = conv / conv.sum() * (len(held) / TOP_K)
        else:
            w = np.array([])
        day_ret, cost_today = 0.0, 0.0
        for idx, tk in enumerate(held):
            pos = slots[tk]
            px = float(g.at[tk, "close"]) if tk in g.index else pos["last"]
            day_ret += w[idx] * (px / pos["last"] - 1 if pos["last"] else 0.0)
            pos["last"] = px
            pos["peak"] = max(pos["peak"], px)
            pos["days"] += 1
            entry, reason = pos["entry"], None
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
            if reason:
                cost_today += w[idx] * _cost(pos["spread"])
                closed.append(px / entry - 1 - _cost(pos["spread"]))
                del slots[tk]
        daily.append(day_ret - cost_today)

        if i % REBALANCE_EVERY == 0 and len(slots) < TOP_K:
            for tk, row in ranked.iterrows():
                if len(slots) >= TOP_K:
                    break
                if tk in slots:
                    continue
                atr = float(row["atr"]) if np.isfinite(row["atr"]) and row["atr"] > 0 else 0.03
                slots[tk] = {"entry": float(row["close"]), "last": float(row["close"]),
                             "peak": float(row["close"]), "days": 0,
                             "conv": _conv(row, scheme),
                             "trail": float(np.clip(atr * policy.trail_mult,
                                                    policy.trail_min, policy.trail_max)),
                             "spread": float(row["spread"])}
    for pos in slots.values():
        closed.append(pos["last"] / pos["entry"] - 1 - _cost(pos["spread"]))
    return np.array(daily), closed


def fold_metrics(daily):
    s = pd.Series(daily)
    eq = (1 + s).cumprod()
    total = float(eq.iloc[-1]) - 1 if len(s) else 0.0
    sharpe = float(s.mean() / s.std() * np.sqrt(252)) if s.std() > 0 else 0.0
    maxdd = float(((eq - eq.cummax()) / eq.cummax()).min()) if len(s) else 0.0
    return total, sharpe, maxdd


def run(policy, scheme):
    rows, wrs = [], []
    for i in range(1, 17):
        dates, by_date = load_fold(i)
        daily, closed = simulate_fold(dates, by_date, policy, scheme)
        tot, sh, dd = fold_metrics(daily)
        rows.append({"fold": i, "total_return": tot, "sharpe": sh, "max_dd": dd})
        if closed:
            wrs.append((np.mean([c > 0 for c in closed]), len(closed)))
    n = sum(w[1] for w in wrs)
    wr = sum(w[0] * w[1] for w in wrs) / n if n else 0.0
    return pd.DataFrame(rows), wr


def main():
    t0 = time.time()
    p = Policy(**{k: DEC["exit_policy"][k] for k in Policy.__dataclass_fields__
                  if k in DEC["exit_policy"]})
    print(f"\n  46 — sizing on PURGED cache (K={TOP_K}, {CB:g}bps+spread, "
          f"Sidak M={SESSION_TRIALS})\n")
    base, base_wr = run(p, "equal")
    mo = (1 + base["total_return"].mean()) ** (1 / 3) - 1
    print(f"  {'BASE equal-weight':20} {base['total_return'].mean():+7.1%}/f "
          f"{mo:+6.2%}/mo Sh{base['sharpe'].mean():5.2f} WR{base_wr:4.0%}")

    results = {}
    trial_sharpes = [base["total_return"].mean() / base["total_return"].std()]
    for name, scheme in [("conv exp(z)", "conv"), ("conv linear", "convlin"),
                         ("inverse-vol", "invvol")]:
        df, wr = run(p, scheme)
        results[name] = df
        pb = mc.paired_bootstrap(df["total_return"].values, base["total_return"].values)
        p_sidak = mc.sidak(pb["p_le0"], SESSION_TRIALS)
        sel = (df["total_return"][:SELECT_FOLDS] - base["total_return"][:SELECT_FOLDS]).mean()
        conf = (df["total_return"][SELECT_FOLDS:] - base["total_return"][SELECT_FOLDS:]).mean()
        mo = (1 + df["total_return"].mean()) ** (1 / 3) - 1
        trial_sharpes.append(df["total_return"].mean() / df["total_return"].std())
        print(f"  {name:20} {df['total_return'].mean():+7.1%}/f {mo:+6.2%}/mo "
              f"Sh{df['sharpe'].mean():5.2f} WR{wr:4.0%}  "
              f"d{pb['mean_diff']:+.1%}[{pb['ci_lo']:+.1%},{pb['ci_hi']:+.1%}] "
              f"pSidak={p_sidak:.2f} sel{sel:+.1%} conf{conf:+.1%}")

    # Gate: pick best on SELECT folds, then judge on the full battery
    best = max(results, key=lambda k: (results[k]["total_return"][:SELECT_FOLDS]
                                       - base["total_return"][:SELECT_FOLDS]).mean())
    df = results[best]
    pb = mc.paired_bootstrap(df["total_return"].values, base["total_return"].values)
    p_sidak = mc.sidak(pb["p_le0"], SESSION_TRIALS)
    conf = (df["total_return"][SELECT_FOLDS:] - base["total_return"][SELECT_FOLDS:]).mean()
    dsr = mc.def_sharpe(df["total_return"].values,
                        float(np.std(trial_sharpes)), SESSION_TRIALS)
    ok = p_sidak < 0.05 and conf > 0 and dsr > 0.95
    print(f"\n  SELECT(1-{SELECT_FOLDS}) winner: '{best}'")
    print(f"  gate: pSidak={p_sidak:.3f}(<0.05) confirm(13-16)d={conf:+.1%}(>0) "
          f"DSR={dsr:.3f}(>0.95)")
    print(f"  -> {'PROMOTE' if ok else 'REJECT — not distinguishable from equal-weight after the haircut'}")
    print(f"\n  Runtime {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
