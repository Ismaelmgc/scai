"""V4 experiment — CAPITAL DEPLOYMENT levers (not signal): the model is at its
signal ceiling (IC ~0.06), so the untapped axis for higher monthly return is HOW
we deploy capital, not what we predict.

Three levers on a slot portfolio (reuses the v4_fs_base28 daily-prediction cache;
15bps+spread costs; MC paired bootstrap vs the equal-weight baseline):

  #2 Conviction sizing : weight held names by their entry cross-sectional pred
                         z-score — exp(z) (strong) or ∝z (linear) — vs equal.
  #3 Exposure / regime : hold cash on rebalances when the universe is risk-off
                         (median trailing-10d return of tradable names < θ). A
                         cross-sectional z-gate is useless (the top-K always have
                         high z), so the regime signal is the universe trend.
  #1 Leverage          : pure analysis — scale the best daily-return series by L,
                         subtract margin borrow, recompute monthly return / Sharpe
                         / maxDD + the underlying drop that triggers a margin call.
                         Leverage adds NO alpha — it scales return AND drawdown.

Usage:
    PYTHONPATH=src python scripts/v3/43_capital_deployment.py
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
DEC = json.loads((ROOT / "data/v3_benchmarks/v4_filter_decision.json").read_text())
MP, MA, CB = DEC["min_price"], DEC["min_adv_usd"], DEC["cost_bps"]
TOP_K, REBALANCE_EVERY = 8, 5
BORROW_APR = 0.065


@dataclass
class Policy:
    trail_mult: float = 5.3
    trail_min: float = 0.10
    trail_max: float = 0.16
    adaptive_tighten: float | None = None
    adaptive_after_days: int = 5
    profit_target: float | None = 0.40
    time_stop: int = 20


def pol(d):
    return Policy(**{k: d[k] for k in Policy.__dataclass_fields__ if k in d})


def load_fold(i: int):
    df = pd.read_parquet(CACHE / f"fold_{i:02d}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"atr_pct_20d": "atr", "adv_usd_20d": "adv", "cs_spread_20d": "spread"})
    # regime proxy: median trailing-10d return of the (roughly) tradable universe
    piv = df.pivot_table(index="date", columns="ticker", values="close")
    reg = piv.pct_change(10).median(axis=1)
    by_date = {}
    for d, g in df.groupby("date"):
        g = g.dropna(subset=["close"]).drop_duplicates("ticker").set_index("ticker")
        tr = g[(g["close"] >= MP) & (g["adv"] >= MA)]
        mu, sd = tr["pred"].mean(), tr["pred"].std()
        g["z"] = (g["pred"] - mu) / sd if sd > 0 else 0.0
        by_date[d] = g
    dates = sorted(by_date)
    regime = {d: (float(reg.loc[d]) if d in reg.index and np.isfinite(reg.loc[d]) else 0.0)
              for d in dates}
    return dates, by_date, regime


def _cost(spread):
    c = 2 * CB / 1e4
    if np.isfinite(spread) and spread > 0:
        c = max(c, min(float(spread), 0.10))
    return c


def _conv(z, scheme):
    if scheme == "conv":
        return float(np.exp(np.clip(z, -2, 10)))
    if scheme == "convlin":
        return max(float(z), 0.1)
    return 1.0


def simulate_fold(dates, by_date, regime, policy, scheme="equal", gate_reg=None):
    slots: dict[str, dict] = {}
    daily, closed = [], []
    for i, d in enumerate(dates):
        g = by_date[d]
        ranked = g[(g["close"] >= MP) & (g["adv"] >= MA)].sort_values("pred", ascending=False)

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

        risk_off = gate_reg is not None and regime[d] < gate_reg
        if i % REBALANCE_EVERY == 0 and len(slots) < TOP_K and not risk_off:
            for tk, row in ranked.iterrows():
                if len(slots) >= TOP_K:
                    break
                if tk in slots:
                    continue
                atr = float(row["atr"]) if np.isfinite(row["atr"]) and row["atr"] > 0 else 0.03
                slots[tk] = {"entry": float(row["close"]), "last": float(row["close"]),
                             "peak": float(row["close"]), "days": 0,
                             "conv": _conv(row["z"], scheme),
                             "trail": float(np.clip(atr * policy.trail_mult,
                                                    policy.trail_min, policy.trail_max)),
                             "spread": float(row["spread"])}
    for pos in slots.values():
        closed.append(pos["last"] / pos["entry"] - 1 - _cost(pos["spread"]))
    return np.array(daily), closed


def metrics(daily, lev=1.0):
    r = lev * daily - BORROW_APR / 252 * (lev - 1)
    s = pd.Series(r)
    eq = (1 + s).cumprod()
    total = float(eq.iloc[-1]) - 1 if len(s) else 0.0
    sharpe = float(s.mean() / s.std() * np.sqrt(252)) if s.std() > 0 else 0.0
    maxdd = float(((eq - eq.cummax()) / eq.cummax()).min()) if len(s) else 0.0
    return total, sharpe, maxdd


def run(policy, scheme="equal", gate_reg=None):
    dailies, wrs = [], []
    for i in range(1, 17):
        dates, by_date, regime = load_fold(i)
        if not dates:
            continue
        daily, closed = simulate_fold(dates, by_date, regime, policy, scheme, gate_reg)
        dailies.append(daily)
        if closed:
            wrs.append((np.mean([c > 0 for c in closed]), len(closed)))
    return dailies, wrs


def agg(dailies, lev=1.0):
    rows = [metrics(d, lev) for d in dailies]
    return (np.array([r[0] for r in rows]), np.array([r[1] for r in rows]),
            np.array([r[2] for r in rows]))


def wravg(wrs):
    n = sum(w[1] for w in wrs)
    return sum(w[0] * w[1] for w in wrs) / n if n else 0.0


def line(name, dailies, wrs, base_tot=None):
    tot, sh, dd = agg(dailies)
    mo = (1 + tot.mean()) ** (1 / 3) - 1
    out = (f"  {name:24} {tot.mean():+7.1%}/f {mo:+6.1%}/mo Sh{sh.mean():5.2f} "
           f"WR{wravg(wrs):4.0%} DD{dd.mean():6.1%}")
    if base_tot is not None:
        pb = mc.paired_bootstrap(tot, base_tot)
        flag = " <-CI>0" if pb["ci_lo"] > 0 else ""
        out += f"  d{pb['mean_diff']:+.1%}[{pb['ci_lo']:+.1%},{pb['ci_hi']:+.1%}]{flag}"
    print(out)
    return tot


def main():
    t0 = time.time()
    p = pol(DEC["exit_policy"])
    print(f"\n  Capital deployment — slot portfolio (K={TOP_K}, {CB:g}bps+spread)\n")
    print("  #2 sizing + #3 regime gate (unlevered):")
    base_d, base_wr = run(p, "equal")
    base_tot = line("BASE equal-weight", base_d, base_wr)
    variants = {}
    for name, sc, gr in [("conviction exp(z)", "conv", None),
                         ("conviction linear", "convlin", None),
                         ("regime gate >0", "equal", 0.0),
                         ("regime gate >-2%", "equal", -0.02),
                         ("convlin + regime>0", "convlin", 0.0)]:
        d, w = run(p, sc, gr)
        line(name, d, w, base_tot)
        variants[name] = d

    cand = {"equal": base_d, **{k: variants[k] for k in variants}}
    best = max(cand, key=lambda k: agg(cand[k])[0].mean())
    bd = cand[best]
    print(f"\n  #1 LEVERAGE on best = '{best}' (borrow {BORROW_APR:.1%} APR on the levered part):")
    print(f"  {'lev':>4} {'ret/fold':>9} {'ret/mo':>7} {'Sharpe':>7} {'maxDD':>7} "
          f"{'margin call if mkt drops':>26}")
    for lev in [1.0, 1.25, 1.5, 2.0, 2.5]:
        tot, sh, dd = agg(bd, lev)
        mo = (1 + tot.mean()) ** (1 / 3) - 1

        def call_drop(m):
            denom = lev - m * lev
            return max(0.0, min(1.0, (1 - m * lev) / denom)) if denom else 1.0
        cd = "—" if lev == 1 else f"{call_drop(0.25):.0%}@m25 / {call_drop(0.40):.0%}@m40"
        print(f"  {lev:>4.2f} {tot.mean():+8.1%} {mo:+7.1%} {sh.mean():6.2f} "
              f"{dd.mean():7.1%} {cd:>20}")

    print("\n  sizing/regime: CI>0 vs equal = real gain.  leverage scales return AND "
          f"drawdown (no alpha).  Runtime {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
