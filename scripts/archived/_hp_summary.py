"""Compare HP search results vs current V3 candidate (lambdarank)."""
import json
from pathlib import Path

BASELINE = "v3_lambdarank"
configs = [BASELINE] + [
    "v3_hp_leaves31", "v3_hp_leaves63",
    "v3_hp_minc30", "v3_hp_minc100",
    "v3_hp_lr05", "v3_hp_trunc4",
    "v3_hp_combo", "v3_hp_reg2",
]

rows = []
for c in configs:
    fp = Path(f"data/v3_benchmarks/{c}.json")
    if not fp.exists():
        continue
    a = json.loads(fp.read_text())["aggregate"]
    rows.append((c, a))

print(f"{'config':<22}{'IC':>9}{'Sh':>7}{'WR':>8}{'SelMed':>9}{'+folds':>8}{'MeanR':>10}")
print("-" * 73)
for name, a in rows:
    print(f"{name:<22}{a['mean_ic']:>9.4f}{a['mean_sharpe']:>7.2f}"
          f"{a['mean_win_rate']:>8.2%}{a['selectivity_median']:>9.2%}"
          f"{a['folds_positive_ret']:>8}{a['mean_return']:>10.1%}")

base = rows[0][1]
print(f"\nΔ vs {BASELINE}:")
for name, a in rows[1:]:
    score = (a['mean_sharpe']-base['mean_sharpe'])*0.4 + \
            (a['mean_win_rate']-base['mean_win_rate'])*100*0.3 + \
            (a['selectivity_median']-base['selectivity_median'])*100*0.3
    print(f"  {name:<22} ΔSh={a['mean_sharpe']-base['mean_sharpe']:+.2f}"
          f"  ΔWR={a['mean_win_rate']-base['mean_win_rate']:+.2%}"
          f"  ΔSelMed={a['selectivity_median']-base['selectivity_median']:+.2%}"
          f"  ΔIC={a['mean_ic']-base['mean_ic']:+.4f}"
          f"  score={score:+.2f}")
