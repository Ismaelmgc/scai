"""V3 Sprint 1 — comparison summary across all benchmarks run."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))


CONFIGS = [
    ("v2_baseline", "V2 baseline (regression + raw sector)"),
    ("v3_sector_fix", "V3 + sector enrichment"),
    ("v3_regime", "V3 + sector + 22 regime features"),
    ("v3_lambdarank", "V3 + sector + LambdaRank objective"),
]


def main() -> None:
    rows = []
    for name, desc in CONFIGS:
        fp = Path(f"data/v3_benchmarks/{name}.json")
        if not fp.exists():
            print(f"MISSING: {name}")
            continue
        data = json.loads(fp.read_text())
        agg = data["aggregate"]
        rows.append((name, desc, agg))

    keys = [
        ("n_features", "n_feat"),
        ("mean_ic", "IC"),
        ("ic_std", "IC σ"),
        ("mean_sharpe", "Sharpe"),
        ("mean_win_rate", "WinRate"),
        ("selectivity_median", "SelMed"),
        ("folds_positive_ret", "+folds"),
        ("mean_return", "MeanRet"),
        ("median_return", "MedRet"),
    ]

    header_w = 38
    col_w = 12
    print("\n" + "=" * (header_w + col_w * len(rows)))
    print("  V3 SPRINT 1 — BENCHMARK COMPARISON")
    print("=" * (header_w + col_w * len(rows)))
    print(f"{'Metric':<{header_w}}" + "".join(f"{n[:col_w-1]:>{col_w}}" for n, _, _ in rows))
    print("-" * (header_w + col_w * len(rows)))
    for k, label in keys:
        line = f"{label:<{header_w}}"
        for _, _, agg in rows:
            v = agg.get(k, "")
            if isinstance(v, float):
                if any(t in k for t in ("rate", "return", "selectivity")):
                    line += f"{v:>{col_w}.2%}"
                else:
                    line += f"{v:>{col_w}.4f}"
            else:
                line += f"{str(v):>{col_w}}"
        print(line)
    print("=" * (header_w + col_w * len(rows)))
    print("\nDescriptions:")
    for n, d, _ in rows:
        print(f"  {n:<22} → {d}")

    # Diff vs baseline
    base = rows[0][2]
    print("\n\nDeltas vs V2 baseline:")
    for n, _, a in rows[1:]:
        d_ic = a["mean_ic"] - base["mean_ic"]
        d_wr = a["mean_win_rate"] - base["mean_win_rate"]
        d_sh = a["mean_sharpe"] - base["mean_sharpe"]
        d_sel = a["selectivity_median"] - base["selectivity_median"]
        print(f"  {n:<22}  ΔIC={d_ic:+.4f}  ΔWR={d_wr:+.2%}  ΔSh={d_sh:+.2f}  ΔSelMed={d_sel:+.2%}")


if __name__ == "__main__":
    main()
