"""Aggregate existing WF benchmark folds into period-level summaries.

Compares both strategies against:
- Universo mediana (equal-weight small-cap median)
- IWM (Russell 2000 ETF — natural small-cap benchmark)
- SPY (S&P 500 ETF — general market)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

b = json.load(open(ROOT / "data/v3_benchmarks/v3_wr_baseline_8.json"))
a = json.load(open(ROOT / "data/v3_benchmarks/v3_wr_adapt_stop.json"))

bf = b["folds"]
af = a["folds"]


# ── Download IWM + SPY data and compute per-fold returns ──────────────

def _parse_fold_dates(period_str: str) -> tuple[str, str]:
    """Parse '2022-06->2022-09' into start/end date strings."""
    start, end = period_str.split("->")
    return start.strip() + "-01", end.strip() + "-01"


def _download_index_returns() -> dict[str, list[float]]:
    """Download IWM + SPY via yfinance and compute return per fold."""
    import yfinance as yf

    # Get date range from folds
    first_start, _ = _parse_fold_dates(bf[0]["period"])
    _, last_end = _parse_fold_dates(bf[-1]["period"])

    # Download enough margin
    start = pd.Timestamp(first_start) - pd.DateOffset(days=5)
    end = pd.Timestamp(last_end) + pd.DateOffset(days=5)

    result = {}
    for ticker in ["IWM", "SPY"]:
        print(f"  Downloading {ticker} ({start.date()} → {end.date()})...")
        data = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                           end=end.strftime("%Y-%m-%d"), progress=False)
        if data.empty:
            print(f"  ⚠ No data for {ticker}")
            result[ticker] = [0.0] * len(bf)
            continue

        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.index = pd.to_datetime(data.index)
        close = data["Close"].sort_index()

        fold_returns = []
        for fold in bf:
            fold_start, fold_end = _parse_fold_dates(fold["period"])
            fs = pd.Timestamp(fold_start)
            fe = pd.Timestamp(fold_end)
            mask = close[(close.index >= fs) & (close.index < fe)]
            if len(mask) >= 2:
                ret = float(mask.iloc[-1] / mask.iloc[0] - 1)
            else:
                ret = 0.0
            fold_returns.append(ret)
        result[ticker] = fold_returns

    return result


print("Downloading index data...")
idx_returns = _download_index_returns()
print()

# Each fold is ~63 trading days (~3 calendar months), except last fold (~1 month)
# Folds by index:
#  0: 2022-06→2022-09   4: 2023-06→2023-09   8:  2024-06→2024-09  12: 2025-07→2025-10
#  1: 2022-09→2022-12   5: 2023-09→2023-12   9:  2024-09→2025-01  13: 2025-10→2026-01
#  2: 2022-12→2023-03   6: 2023-12→2024-03  10:  2025-01→2025-04  14: 2026-01→2026-04
#  3: 2023-03→2023-06   7: 2024-03→2024-06  11:  2025-04→2025-07  15: 2026-04→2026-05

periods = [
    ("Últ. 6 meses  (2025-10 → 2026-05)", 13, 16),
    ("Último año    (2025-04 → 2026-05)", 11, 16),
    ("Últimos 2 años (2024-06 → 2026-05)", 8, 16),
    ("Últimos 3 años (2023-06 → 2026-05)", 4, 16),
    ("Full          (2022-06 → 2026-05)", 0, 16),
]


def aggregate(folds, start, end):
    sel = folds[start:end]
    cum = np.prod([1 + f["total_return"] for f in sel]) - 1
    # months: 3 months per fold, except last fold if n_trades <= 16 → ~1 month
    months = sum(3 for f in sel[:-1]) + (1 if sel[-1]["n_trades"] <= 16 else 3)
    years = months / 12
    ann = (1 + cum) ** (1 / years) - 1 if years > 0 else cum
    tot_trades = sum(f["n_trades"] for f in sel)
    wr = sum(f["win_rate"] * f["n_trades"] for f in sel) / tot_trades
    sharpes = [f["sharpe"] for f in sel if f["sharpe"] != 0]
    sharpe_avg = np.mean(sharpes) if sharpes else 0
    median_avg = np.mean([f["median_return"] for f in sel])
    # Count positive folds
    pos_folds = sum(1 for f in sel if f["total_return"] > 0)
    return {
        "cum": cum, "ann": ann, "wr": wr, "sharpe": sharpe_avg,
        "median_trade": median_avg, "trades": tot_trades,
        "months": months, "pos_folds": pos_folds, "total_folds": len(sel),
    }


def aggregate_mkt(folds, start, end):
    sel = folds[start:end]
    cum = np.prod([1 + f["market_return"] for f in sel]) - 1
    months = sum(3 for f in sel[:-1]) + (1 if sel[-1]["n_trades"] <= 16 else 3)
    years = months / 12
    ann = (1 + cum) ** (1 / years) - 1 if years > 0 else cum
    return {"cum": cum, "ann": ann}


def aggregate_idx(returns_list, start, end):
    sel = returns_list[start:end]
    cum = np.prod([1 + r for r in sel]) - 1
    n_folds = end - start
    # Approximate months same as folds
    months = sum(3 for _ in sel[:-1]) + (1 if n_folds > 0 and bf[end-1]["n_trades"] <= 16 else 3)
    years = months / 12
    ann = (1 + cum) ** (1 / years) - 1 if years > 0 else cum
    return {"cum": cum, "ann": ann}


print()
print("=" * 130)
print("  RENDIMIENTO POR PERÍODO — Baseline vs Adaptive Stop vs Índices de Mercado")
print("  Walk-forward out-of-sample, TOP-8 picks, rebalance cada 5 días, trailing stop ATR-adaptativo")
print("=" * 130)
print()

for name, s, e in periods:
    rb = aggregate(bf, s, e)
    ra = aggregate(af, s, e)
    rm = aggregate_mkt(bf, s, e)
    ri = aggregate_idx(idx_returns["IWM"], s, e)
    rs = aggregate_idx(idx_returns["SPY"], s, e)

    print(f"  ┌─ {name}")
    print(f"  │  {rb['months']} meses, {rb['trades']} trades, {rb['total_folds']} folds")
    print(f"  │")
    print(f"  │  {'':30s} {'BASELINE':>12s}  {'ADAPTIVE':>12s}  {'Univ.Med':>12s}  {'IWM':>10s}  {'SPY':>10s}")
    print(f"  │  {'─'*30} {'─'*12}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*10}")
    print(f"  │  {'Retorno acumulado':<30s} {rb['cum']:>+11.0%}  {ra['cum']:>+11.0%}  {rm['cum']:>+11.0%}  {ri['cum']:>+9.1%}  {rs['cum']:>+9.1%}")
    print(f"  │  {'Retorno anualizado':<30s} {rb['ann']:>+11.0%}  {ra['ann']:>+11.0%}  {rm['ann']:>+11.0%}  {ri['ann']:>+9.1%}  {rs['ann']:>+9.1%}")
    print(f"  │  {'Sharpe medio (por fold)':<30s} {rb['sharpe']:>11.2f}  {ra['sharpe']:>11.2f}  {'—':>12s}  {'—':>10s}  {'—':>10s}")
    print(f"  │  {'Win Rate':<30s} {rb['wr']:>11.1%}  {ra['wr']:>11.1%}  {'—':>12s}  {'—':>10s}  {'—':>10s}")
    print(f"  │  {'Mediana retorno por trade':<30s} {rb['median_trade']:>+11.2%}  {ra['median_trade']:>+11.2%}  {'—':>12s}  {'—':>10s}  {'—':>10s}")
    print(f"  │  {'Folds positivos':<30s} {rb['pos_folds']}/{rb['total_folds']:>10d}  {ra['pos_folds']}/{ra['total_folds']:>10d}  {'—':>12s}  {'—':>10s}  {'—':>10s}")
    alpha_b_iwm = rb["cum"] - ri["cum"]
    alpha_a_iwm = ra["cum"] - ri["cum"]
    alpha_b_spy = rb["cum"] - rs["cum"]
    alpha_a_spy = ra["cum"] - rs["cum"]
    print(f"  │  {'Alpha vs IWM':<30s} {alpha_b_iwm:>+11.0%}  {alpha_a_iwm:>+11.0%}  {'—':>12s}  {'—':>10s}  {'—':>10s}")
    print(f"  │  {'Alpha vs SPY':<30s} {alpha_b_spy:>+11.0%}  {alpha_a_spy:>+11.0%}  {'—':>12s}  {'—':>10s}  {'—':>10s}")
    print(f"  └{'─'*128}")
    print()

# Per-year breakdown
print("=" * 130)
print("  DESGLOSE POR AÑO CALENDARIO")
print("=" * 130)
print()

year_ranges = [
    ("2022 (H2)", 0, 2),
    ("2023",      2, 6),
    ("2024",      6, 10),
    ("2025",     10, 14),
    ("2026 (parcial)", 14, 16),
]

print(f"  {'AÑO':<18s} │ {'BASE ret':>10s} {'BASE WR':>8s} {'BASE Sh':>8s} │ {'ADAPT ret':>10s} {'ADAPT WR':>8s} {'ADAPT Sh':>8s} │ {'Univ.':>7s} {'IWM':>7s} {'SPY':>7s}")
print(f"  {'─'*18} │ {'─'*10} {'─'*8} {'─'*8} │ {'─'*10} {'─'*8} {'─'*8} │ {'─'*7} {'─'*7} {'─'*7}")

for name, s, e in year_ranges:
    rb = aggregate(bf, s, e)
    ra = aggregate(af, s, e)
    rm = aggregate_mkt(bf, s, e)
    ri = aggregate_idx(idx_returns["IWM"], s, e)
    rs = aggregate_idx(idx_returns["SPY"], s, e)
    print(f"  {name:<18s} │ {rb['cum']:>+9.0%} {rb['wr']:>7.1%} {rb['sharpe']:>8.2f} │ {ra['cum']:>+9.0%} {ra['wr']:>7.1%} {ra['sharpe']:>8.2f} │ {rm['cum']:>+6.0%} {ri['cum']:>+6.1%} {rs['cum']:>+6.1%}")

print()

# Fold-by-fold detail
print("=" * 140)
print("  DETALLE POR FOLD (trimestral)")
print("=" * 140)
print()
print(f"  {'#':>2s} {'Período':<18s} │ {'BASE ret':>9s} {'WR':>6s} {'Sharpe':>7s} {'SelMed':>7s} │ {'ADAPT ret':>9s} {'WR':>6s} {'Sharpe':>7s} {'SelMed':>7s} │ {'Univ':>7s} {'IWM':>7s} {'SPY':>7s}")
print(f"  {'─'*2} {'─'*18} │ {'─'*9} {'─'*6} {'─'*7} {'─'*7} │ {'─'*9} {'─'*6} {'─'*7} {'─'*7} │ {'─'*7} {'─'*7} {'─'*7}")

for i in range(16):
    fb = bf[i]
    fa = af[i]
    ri = idx_returns["IWM"][i] if i < len(idx_returns["IWM"]) else 0.0
    rs = idx_returns["SPY"][i] if i < len(idx_returns["SPY"]) else 0.0
    print(f"  {i+1:>2d} {fb['period']:<18s} │ {fb['total_return']:>+8.0%} {fb['win_rate']:>6.1%} {fb['sharpe']:>7.2f} {fb['median_return']:>+7.2%} │ {fa['total_return']:>+8.0%} {fa['win_rate']:>6.1%} {fa['sharpe']:>7.2f} {fa['median_return']:>+7.2%} │ {fb['market_return']:>+7.2%} {ri:>+7.2%} {rs:>+7.2%}")

print()
print("  Nota: Cada fold = ~63 días trading (~3 meses). Retornos compuestos de rebalanceos cada 5 días.")
print("  SelMed = mediana del retorno real de las acciones seleccionadas (selectividad del modelo).")
print("  Univ = mediana del retorno de TODAS las acciones en el universo (benchmark equal-weight).")
print("  IWM = Russell 2000 ETF, SPY = S&P 500 ETF (retorno total en el mismo período).")
print()

# Summary statistics
all_b_wr = [f["win_rate"] for f in bf]
all_a_wr = [f["win_rate"] for f in af]
all_b_ret = [f["total_return"] for f in bf]
all_a_ret = [f["total_return"] for f in af]

print("=" * 110)
print("  RESUMEN ESTADÍSTICO")
print("=" * 110)
print()
print(f"  {'Métrica':<35s} {'BASELINE':>12s}  {'ADAPTIVE':>12s}")
print(f"  {'─'*35} {'─'*12}  {'─'*12}")
print(f"  {'Win Rate media (across folds)':<35s} {np.mean(all_b_wr):>11.1%}  {np.mean(all_a_wr):>11.1%}")
print(f"  {'Win Rate mínima (peor fold)':<35s} {np.min(all_b_wr):>11.1%}  {np.min(all_a_wr):>11.1%}")
print(f"  {'Win Rate máxima (mejor fold)':<35s} {np.max(all_b_wr):>11.1%}  {np.max(all_a_wr):>11.1%}")
print(f"  {'Folds positivos':<35s} {sum(1 for r in all_b_ret if r > 0)}/{len(all_b_ret):>10d}  {sum(1 for r in all_a_ret if r > 0)}/{len(all_a_ret):>10d}")
print(f"  {'Peor fold retorno':<35s} {np.min(all_b_ret):>+11.0%}  {np.min(all_a_ret):>+11.0%}")
print(f"  {'Mejor fold retorno':<35s} {np.max(all_b_ret):>+11.0%}  {np.max(all_a_ret):>+11.0%}")
print(f"  {'Sharpe medio':<35s} {np.mean([f['sharpe'] for f in bf]):>12.2f}  {np.mean([f['sharpe'] for f in af]):>12.2f}")
print()
