"""Live generalization monitor — does the model still rank correctly in production?

The weekly retrain keeps the model FRESH but never validates it: there is no
holdout in the pipeline, so a model that stops generalizing would be refit and
shipped with no alarm. The only true out-of-sample test is what actually happened
to the signals we generated. This job measures that.

For every scored name (the full ~320/day cross-section stored in Supabase), it
computes the realised 20-trading-day forward return from the committed OHLCV and
then reports, over a rolling window of matured dates:

  - Live IC: mean per-date Spearman(score, fwd_ret_20d). The honest live version
    of the backtest's tradable-IC (~+0.008). <= 0 means the ranking is dead.
  - Selection value: avg 20d return of TRADED vs NOT-traded signals. Traded
    should beat not-traded; if not, the gate+ranking add nothing.
  - Traded hit rate / avg return vs the backtest band (WR 51-60%).

It also backfills actual_ret_20d into Supabase (so the dashboard's "Retorno real"
column stops showing "pendiente" forever — the daily pipeline never re-pushes it).

Verdict drives the exit code: 2 = DEGRADED (a scheduled run goes red → GitHub
emails you = the alarm), 0 = OK or still WARMING UP. Warm-up is expected until
~20 trading days after the 2026-06-11 reset (first real read ≈ mid-July 2026).

Usage:
    PYTHONPATH=src python scripts/monitor_live_ic.py
    PYTHONPATH=src python scripts/monitor_live_ic.py --no-write   # don't touch Supabase
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from app.data import supabase_store  # noqa: E402

HORIZON_DAYS = 20          # forward-return horizon (matches V2_HOLD_DAYS / the target)
WINDOW_DAYS = 180          # rolling calendar window of signal dates to consider
MIN_NAMES_PER_DATE = 20    # need a real cross-section for a per-date IC to mean anything
MIN_DATES = 8              # below this many matured dates we only report, never alarm
MIN_TRADED = 30            # below this many matured TRADED signals, WR/selection are
                           # noise (n=8 -> WR CI ≈ [10%,74%]) → informational, never alarm

# Backtest reference band (16-fold WF, tradability filter, honest costs).
REF_IC = 0.008
REF_WR_LO, REF_WR_HI = 0.51, 0.60

STRATEGIES = ("baseline", "adaptive")


def _fwd_returns(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """(ticker, date) -> realised HORIZON-trading-day forward return.

    Positional shift(-H) per ticker = close H trading days later, exactly like
    SignalTracker.backfill_outcomes. NaN where H days don't exist yet (not matured).
    """
    o = ohlcv[["date", "ticker", "close"]].copy()
    o["date"] = pd.to_datetime(o["date"])
    o = o.sort_values(["ticker", "date"])
    fwd_close = o.groupby("ticker")["close"].shift(-HORIZON_DAYS)
    o["fwd_ret_20d"] = fwd_close / o["close"] - 1
    return o[["ticker", "date", "fwd_ret_20d"]]


def _load_signals(strategy: str) -> pd.DataFrame:
    since = (date.today() - timedelta(days=WINDOW_DAYS)).isoformat()
    rows = supabase_store.read_signals_since(strategy, since)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["signal_date"])
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["was_traded"] = df["was_traded"].astype(bool)
    return df


def _per_date_ic(matured: pd.DataFrame) -> tuple[float, int, float]:
    """Mean / count / std of per-date Spearman(score, fwd_ret_20d)."""
    ics = []
    for _, g in matured.groupby("signal_date"):
        if len(g) < MIN_NAMES_PER_DATE or g["score"].nunique() < 2:
            continue
        ic, _ = spearmanr(g["score"], g["fwd_ret_20d"])
        if not np.isnan(ic):
            ics.append(ic)
    if not ics:
        return float("nan"), 0, float("nan")
    return float(np.mean(ics)), len(ics), float(np.std(ics))


def _analyse(strategy: str, fwd: pd.DataFrame, write: bool) -> int:
    sig = _load_signals(strategy)
    print(f"\n── {strategy.upper()} ─────────────────────────────")
    if sig.empty:
        print("  Sin señales en Supabase.")
        return 0

    merged = sig.merge(fwd, on=["ticker", "date"], how="left")
    matured = merged[merged["fwd_ret_20d"].notna()].copy()
    n_dates_total = sig["signal_date"].nunique()
    n_dates_matured = matured["signal_date"].nunique()
    print(f"  Señales: {len(sig)} en {n_dates_total} fechas | "
          f"maduradas (≥{HORIZON_DAYS} sesiones): {len(matured)} en {n_dates_matured} fechas")

    # Bonus: push realised returns back to Supabase so the dashboard column works.
    if write and not matured.empty:
        outcomes = [{"signal_date": r["signal_date"], "ticker": r["ticker"],
                     "actual_ret_20d": round(float(r["fwd_ret_20d"]), 6)}
                    for _, r in matured.iterrows()]
        try:
            supabase_store.update_signal_outcomes(strategy, outcomes)
            print(f"  ✓ actual_ret_20d backfilleado en Supabase: {len(outcomes)} filas")
        except Exception as e:  # best-effort, never fail the monitor on a write
            print(f"  ⚠ No se pudo backfillear Supabase: {e}")

    if n_dates_matured < MIN_DATES:
        first = (sig["date"].min() + pd.Timedelta(days=int(HORIZON_DAYS * 1.5))).date()
        print(f"  ⏳ WARM-UP: {n_dates_matured}/{MIN_DATES} fechas maduradas. "
              f"Lectura fiable a partir de ~{first}. Sin veredicto todavía.")
        return 0

    live_ic, n_ic, ic_std = _per_date_ic(matured)
    traded = matured[matured["was_traded"]]
    missed = matured[~matured["was_traded"]]
    t_avg = float(traded["fwd_ret_20d"].mean()) if len(traded) else float("nan")
    t_wr = float((traded["fwd_ret_20d"] > 0).mean()) if len(traded) else float("nan")
    m_avg = float(missed["fwd_ret_20d"].mean()) if len(missed) else float("nan")
    spread = t_avg - m_avg

    print(f"  Live IC (Spearman/fecha): {live_ic:+.4f} ± {ic_std:.4f}  "
          f"({n_ic} fechas)   [backtest ref ≈ {REF_IC:+.4f}]")
    print(f"  Traded   : {len(traded):4d} señales | ret20d medio {t_avg:+.2%} | "
          f"WR {t_wr:.0%}  [ref {REF_WR_LO:.0%}-{REF_WR_HI:.0%}]")
    print(f"  No-traded: {len(missed):4d} señales | ret20d medio {m_avg:+.2%}")
    print(f"  Valor de selección (traded − no-traded): {spread:+.2%}")

    # ── Verdict ──
    # IC is measured over the FULL cross-section (thousands of matured signals /
    # ≥8 dates) → it can alarm. The traded-based checks (WR / return / selection
    # spread) run on the ~8-name rolling book; below MIN_TRADED they are pure
    # noise, so they stay INFORMATIONAL and never turn the run red.
    fails, warns = [], []
    if live_ic <= 0:
        fails.append(f"IC en vivo ≤ 0 ({live_ic:+.4f}): el ranking no generaliza")
    elif live_ic < REF_IC * 0.5:
        warns.append(f"IC en vivo ({live_ic:+.4f}) < mitad del backtest ({REF_IC:+.4f})")

    n_traded = len(traded)
    if n_traded >= MIN_TRADED:
        if not np.isnan(t_avg) and t_avg <= 0:
            fails.append(f"retorno medio de los traded ≤ 0 ({t_avg:+.2%})")
        if not np.isnan(t_wr) and t_wr < 0.45:
            fails.append(f"WR de traded < 45% ({t_wr:.0%})")
        elif not np.isnan(t_wr) and t_wr < REF_WR_LO:
            warns.append(f"WR de traded ({t_wr:.0%}) por debajo de la banda backtest")
        if not np.isnan(spread) and spread <= 0:
            warns.append("la selección no bate a las no-traded (spread ≤ 0)")
    else:
        warns.append(f"muestra traded pequeña (n={n_traded}<{MIN_TRADED}): "
                     f"WR {t_wr:.0%} / selección {spread:+.2%} informativos, "
                     f"no disparan alarma (ruido a este tamaño)")

    if fails:
        print("  🔴 DEGRADADO:")
        for f in fails + warns:
            print(f"     - {f}")
        print("     → NO es 'reentrenar otra vez' (misma receta). Investigar / parar / "
              "re-validar con el harness.")
        return 2
    if warns:
        print("  🟡 ATENCIÓN:")
        for w in warns:
            print(f"     - {w}")
        return 0
    print("  🟢 OK: el modelo sigue generalizando en producción.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Live generalization / IC monitor")
    ap.add_argument("--no-write", action="store_true",
                    help="Don't backfill actual_ret_20d into Supabase")
    args = ap.parse_args()

    print("═══ SCAI — Monitor de generalización en vivo ═══")
    if not supabase_store.is_configured():
        print("Supabase no configurado — abortando.")
        sys.exit(0)

    ohlcv = pd.read_parquet(ROOT / "data/processed/ohlcv_smallcap.parquet")
    fwd = _fwd_returns(ohlcv)
    print(f"OHLCV hasta {pd.to_datetime(ohlcv['date']).max().date()} | "
          f"horizonte {HORIZON_DAYS} sesiones | ventana {WINDOW_DAYS}d")

    worst = 0
    for strat in STRATEGIES:
        worst = max(worst, _analyse(strat, fwd, write=not args.no_write))
    sys.exit(worst)


if __name__ == "__main__":
    main()
