"""ILLIQUID shadow logger — out-of-sample test of the small-cap illiquidity edge
(see [[smallcap-illiquidity-edge]]) WITHOUT touching the live paper books.

In-sample (16 mined folds) the edge is promising but unconfirmed (α +0.64%/mo vs
IWM @ $5k under realistic impact, but CI spans 0). The only honest arbiter is
forward OOS data. Run daily, it refreshes the illiquid-variant top-8 every 5
TRADING days (matching the backtest's 20d-hold / 5d-rebalance overlapping
cohorts, NOT a fresh unrelated 8 each day), using the SAME production model +
features as daily_pipeline, then scores matured cohorts' realised 20d return vs
IWM. Isolated: own log, no Supabase book, no cash engine — it measures pure
SELECTION alpha (at the strategy's cadence), which is exactly the doubtful part.

Selection = production tradability gate → bottom ADV tercile → drop widest-20%
cs_spread_20d → top-8 by model score. All from daily OHLC-derived features (no
quote data needed: cs_spread_20d is the Corwin-Schultz estimate).

    PYTHONPATH=src python scripts/illiquid_shadow.py          # log today + score matured
    PYTHONPATH=src python scripts/illiquid_shadow.py --score  # score only
Run daily AFTER the daily pipeline refreshes features (features_smallcap.parquet).
"""
from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.features.tradability import tradable_mask  # noqa: E402

# reuse the EXACT production feature lists + model path from the daily pipeline
_spec = importlib.util.spec_from_file_location("dp", ROOT / "scripts/daily_pipeline.py")
dp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(dp)
FEAT = [f for f in (dp.V2_FEATURES + dp.V2_EDGAR_FEATURES + dp.V2_META_FEATURES)]
TOP_K = dp.V2_TOP_K

FEATURES_FP = ROOT / "data/processed/features_smallcap.parquet"
OHLCV_FP = ROOT / "data/processed/ohlcv_smallcap.parquet"
MODEL_FP = ROOT / str(dp.MODEL_PATH)
LOG_FP = ROOT / "data/paper_trading/illiquid_shadow/picks.parquet"
BENCH_FP = ROOT / "data/processed/bench_iwm_spy.parquet"
HOLD_DAYS = 20
REBALANCE_EVERY = 5    # refresh the top-8 every 5 TRADING days, like the backtest
ADV_TERCILE = 1.0 / 3.0
SPREAD_TRIM_Q = 0.80   # keep spread <= 80th pct among low-ADV (drop widest 20%)


def illiquid_picks(day: pd.DataFrame, model) -> pd.DataFrame:
    """Apply the illiquid selection to one date's rows; return the top-8 picks."""
    day = day[tradable_mask(day)].copy()
    if len(day) < TOP_K:
        return pd.DataFrame()
    cols = [f for f in FEAT if f in day.columns]
    day["score"] = model.predict(day[cols].fillna(0).values)
    # bottom ADV tercile (the illiquidity premium lives here)
    low = day[day["adv_usd_20d"].rank(pct=True) <= ADV_TERCILE]
    if len(low) < TOP_K:
        return pd.DataFrame()
    # drop the widest-spread 20% within the low-ADV set
    thr = low["cs_spread_20d"].quantile(SPREAD_TRIM_Q)
    low = low[low["cs_spread_20d"].fillna(low["cs_spread_20d"].median()) <= thr]
    return low.sort_values("score", ascending=False).head(TOP_K)


def do_log() -> None:
    feat = pd.read_parquet(FEATURES_FP)
    feat["date"] = pd.to_datetime(feat["date"])
    latest = feat["date"].max()
    with open(MODEL_FP, "rb") as f:
        model = pickle.load(f)  # noqa: S301
    picks = illiquid_picks(feat[feat["date"] == latest], model)
    if picks.empty:
        print(f"  [{latest.date()}] no valid illiquid selection (too few candidates)")
        return
    LOG_FP.parent.mkdir(parents=True, exist_ok=True)
    log = pd.read_parquet(LOG_FP) if LOG_FP.exists() else pd.DataFrame()
    # Rebalance cadence: the job runs daily, but the strategy refreshes the top-8
    # every REBALANCE_EVERY *trading* days (20d hold / 5d rebalance = overlapping
    # cohorts, NOT a fresh unrelated 8 each day). Skip until 5 trading days have
    # passed since the last cohort so the shadow log mirrors the backtest.
    if not log.empty:
        last = pd.to_datetime(log["pick_date"]).max()
        cal = np.sort(feat["date"].unique())
        n_since = int(((cal > np.datetime64(last)) & (cal <= np.datetime64(latest))).sum())
        if n_since < REBALANCE_EVERY:
            print(f"  [{latest.date()}] {n_since} trading day(s) since last cohort "
                  f"{last.date()} (rebalance every {REBALANCE_EVERY}) — skip")
            return
    rows = picks[["ticker", "score", "adv_usd_20d", "cs_spread_20d", "close"]].copy()
    rows.insert(0, "pick_date", str(latest.date()))
    rows = rows.rename(columns={"close": "entry_close"})
    out = pd.concat([log, rows], ignore_index=True)
    out.to_parquet(LOG_FP, index=False)
    print(f"  [{latest.date()}] logged {len(rows)} illiquid picks: "
          f"{', '.join(rows['ticker'])}")
    print(f"    median spread {rows['cs_spread_20d'].median():.2%}, "
          f"median ADV ${rows['adv_usd_20d'].median()/1e6:.1f}M")


def _iwm(start, end) -> pd.DataFrame:
    b = pd.read_parquet(BENCH_FP) if BENCH_FP.exists() else pd.DataFrame(columns=["date", "IWM"])
    if not b.empty:
        b["date"] = pd.to_datetime(b["date"])
    if b.empty or b["date"].max() < end:
        import yfinance as yf
        d = yf.download("IWM", start=str((start - pd.Timedelta(days=5)).date()),
                        end=str((end + pd.Timedelta(days=5)).date()), auto_adjust=True, progress=False)
        s = d["Close"]; s = s.iloc[:, 0] if hasattr(s, "columns") else s
        b = s.rename("IWM").reset_index(); b.columns = ["date", "IWM"]; b["date"] = pd.to_datetime(b["date"])
        b.to_parquet(BENCH_FP, index=False)
    return b


def do_score() -> None:
    if not LOG_FP.exists():
        print("  no picks logged yet"); return
    log = pd.read_parquet(LOG_FP); log["pick_date"] = pd.to_datetime(log["pick_date"])
    ohlcv = pd.read_parquet(OHLCV_FP, columns=["date", "ticker", "close"])
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    dmax = ohlcv["date"].max()
    bench = _iwm(log["pick_date"].min(), dmax)

    rows = []
    for pdate, g in log.groupby("pick_date"):
        fwd = ohlcv[ohlcv["date"] > pdate]["date"].drop_duplicates().sort_values()
        if len(fwd) < HOLD_DAYS:
            continue  # not matured
        exit_date = fwd.iloc[HOLD_DAYS - 1]
        rets = []
        for _, p in g.iterrows():
            s = ohlcv[(ohlcv["ticker"] == p["ticker"]) & (ohlcv["date"] > pdate) &
                      (ohlcv["date"] <= exit_date)].sort_values("date")["close"]
            if len(s) >= 2 and p["entry_close"] > 0:
                rets.append(float(s.iloc[-1] / p["entry_close"] - 1))
        if not rets:
            continue
        bw = bench[(bench["date"] > pdate) & (bench["date"] <= exit_date)].dropna(subset=["IWM"])
        iwm = float(bw["IWM"].iloc[-1] / bw["IWM"].iloc[0] - 1) if len(bw) >= 2 else np.nan
        rows.append({"pick_date": pdate.date(), "strat_20d": float(np.mean(rets)),
                     "iwm_20d": iwm, "alpha": float(np.mean(rets)) - iwm, "n": len(rets)})

    if not rows:
        print("  no matured cohorts yet (need 20 trading days forward per pick_date)"); return
    r = pd.DataFrame(rows)
    print(f"\n  === ILLIQUID SHADOW — OOS forward track record ({len(r)} matured cohorts) ===")
    for _, x in r.iterrows():
        print(f"    {x['pick_date']}: strat {x['strat_20d']:+6.2%}  IWM {x['iwm_20d']:+6.2%}  "
              f"α {x['alpha']:+6.2%}  (n={int(x['n'])})")
    print(f"  MEAN 20d: strat {r['strat_20d'].mean():+.2%}  IWM {r['iwm_20d'].mean():+.2%}  "
          f"α {r['alpha'].mean():+.2%}   cohorts α>0: {int((r['alpha']>0).sum())}/{len(r)}")
    print("  (raw selection alpha, no cost/stops — the doubtful part. Accumulates forward.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true", help="score matured cohorts only")
    args = ap.parse_args()
    if not args.score:
        do_log()
    do_score()


if __name__ == "__main__":
    main()
