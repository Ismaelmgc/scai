"""V4 final report: honest metrics with uncertainty, old fiction vs reality.

Produces reports/v4_final_report.md with, per configuration:
- Win rate with trade count and Wilson 95% CI
- Monthly return (fold return de-annualized: (1+r)^(1/3)-1, folds ~3 months)
- Sharpe, max drawdown, folds positive
- Alpha vs median small-cap (harness) and vs SPY buy & hold per fold period

Usage:
    PYTHONPATH=src python scripts/v3/24_v4_report.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "data" / "v3_benchmarks"
CACHE_META = BENCH / "cache" / "v4_nometa" / "meta.json"

# (config json, label, comment)
CONFIGS = [
    ("no_meta", "V3 era (leaky-era harness)", "pre-fix params; reference only"),
    ("v4_nofilt_nometa", "Old fiction (no filter, no cost)", "what V3 believed"),
    ("v4_filt_baseline", "Honest baseline (filter + 15bps)", "anchor"),
    ("v4_exit_pt40", "V4 Baseline strategy (pt40)", "deployed: baseline portfolio"),
    ("v4_exit_adaptive6_pt40", "V4 Adaptive strategy", "deployed: adaptive portfolio"),
    ("v4_filt_baseline_spreadcost", "Conservative bound (spread costs)", "stress costing"),
]


def wilson_ci(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def spy_fold_returns() -> dict[str, float]:
    """SPY buy & hold return per fold period (from cache fold boundaries)."""
    meta = json.loads(CACHE_META.read_text())
    spy = pd.read_parquet(ROOT / "data/processed/smallcap_spy.parquet")
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date")
    out = {}
    for m in meta:
        t0, t1 = pd.Timestamp(m["test_start"]), pd.Timestamp(m["test_end"])
        win = spy[(spy.date >= t0) & (spy.date < t1)]
        out[m["period"]] = float(win.iloc[-1].close / win.iloc[0].close - 1) if len(win) > 1 else 0.0
    return out


def describe(payload: dict, spy_rets: dict[str, float]) -> dict:
    agg = payload["aggregate"]
    folds = payload["folds"]
    n_trades = agg["total_trades"]
    wr = agg["mean_win_rate"]
    lo, hi = wilson_ci(wr * n_trades, n_trades)
    monthly = (1 + agg["mean_return"]) ** (1 / 3) - 1
    spy_alpha = [f["total_return"] - spy_rets.get(f["period"], 0.0) for f in folds]
    max_dd = min(f["max_dd"] for f in folds)
    return {
        "wr": wr, "wr_lo": lo, "wr_hi": hi, "n_trades": n_trades,
        "monthly": monthly, "mean_return": agg["mean_return"],
        "sharpe": agg["mean_sharpe"], "max_dd": max_dd,
        "alpha_mkt": agg["mean_alpha"],
        "alpha_spy": sum(spy_alpha) / len(spy_alpha) if spy_alpha else 0.0,
        "folds_pos": agg["folds_positive_ret"],
        "ic_tradable": agg.get("mean_ic_tradable", agg.get("mean_ic", 0.0)),
    }


def main() -> None:
    spy_rets = spy_fold_returns()
    rows = []
    for cfg, label, comment in CONFIGS:
        fp = BENCH / f"{cfg}.json"
        if not fp.exists():
            print(f"  (skipping {cfg} — no benchmark file)")
            continue
        d = describe(json.loads(fp.read_text()), spy_rets)
        d.update({"config": cfg, "label": label, "comment": comment})
        rows.append(d)

    lines = [
        "# SCAI V4 — Final Validation Report",
        "",
        f"_Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}. 16-fold walk-forward, "
        "2022-06 → 2026-06, out-of-sample. Filter: price ≥ $1.50, ADV20 ≥ $500K, "
        "cost 15 bps/side unless noted._",
        "",
        "| Config | WR (95% CI) | Trades | Monthly ret | Fold ret | Sharpe | Max DD | α vs mkt | α vs SPY | +Folds |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for d in rows:
        lines.append(
            f"| **{d['label']}** | {d['wr']:.1%} ({d['wr_lo']:.0%}–{d['wr_hi']:.0%}) "
            f"| {d['n_trades']:,} | {d['monthly']:+.1%} | {d['mean_return']:+.1%} "
            f"| {d['sharpe']:+.2f} | {d['max_dd']:+.1%} | {d['alpha_mkt']:+.1%} "
            f"| {d['alpha_spy']:+.1%} | {d['folds_pos']} |"
        )
    lines += [
        "",
        "## Honest reading",
        "",
        "- **The old numbers were fiction.** The unfiltered backtest earned a large "
        "share of its return in stocks that could not actually be bought (sub-penny, "
        "$11–$153/day volume). Live trading proved it: 22 trades, all stopped out, "
        "portfolios −7% / +4% while the backtest promised +18%/fold.",
        "- **What V4 actually changes**: tradability gate (selection only), 40% "
        "profit target on both strategies, adaptive tighten kept on the adaptive "
        "portfolio. Features and model unchanged — every candidate batch "
        "(rank transforms, microstructure, fundamentals) failed promotion criteria.",
        "- **WR > 70% verdict**: not reachable without sacrificing returns. The "
        "honest envelope is ~51% (baseline) to ~60% (adaptive) with these CIs. "
        "Anyone promising 70%+ on 20-day small-cap holds is selling leakage.",
        "- **Expected live performance**: between the deployed-strategy rows and "
        "the conservative bound. Paper-trade ≥ 30 closed trades before trusting "
        "any live number.",
        "",
        "## Caveats",
        "",
        "- 2021–2026 sample is mostly bull market; fold 2 (2022 bear) and fold 11 "
        "(2025 correction) are the realistic stress cases — both negative.",
        "- Backtest fills at close with flat costs; real small-cap fills will be "
        "worse on thin names even post-filter.",
        "- 16 folds ≈ 4 years. Fold-level means carry wide error bars themselves.",
    ]
    out = ROOT / "reports" / "v4_final_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
