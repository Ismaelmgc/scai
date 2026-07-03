"""LIQUIDCAP — LSTM pilot: does a sequence model on raw price/volume dynamics
extract signal that the engineered tabular features miss?

Literature prior (be honest about expectations): Fischer & Krauss (EJOR 2018)
ran exactly this — LSTM on S&P 500 constituents, daily bars. 0.46%/day gross
1992-2009, but the edge collapses to ~0 NET OF COSTS after 2010, and most of
the signal was short-term reversal in disguise. Grinsztajn et al. (NeurIPS
2022): NNs underperform trees on medium tabular data. So the null hypothesis
is a null result; this pilot exists to MEASURE it on our PIT universe rather
than argue about it. Counted as trial #8 in the session's Šidák haircut.

Design (CPU-feasible pilot, not a full run):
  input   per (stock, day): last 60 days of [per-sequence z-scored daily
          return, log relative volume vs 60d mean]
  model   LSTM(hidden=32) -> Linear(1), MSE on the winsorized 20d
          date-relative forward return
  folds   semi-annual 2018-2026 (~17), purge 20d, training days strided
          (--stride 3) to fit CPU; 3 epochs, batch 2048
  output  prediction cache compatible with screen_liquidcap.replay -> same
          gates (CI>0, paired-vs-A Šidák(8), confirm block, IC-trend check)

If (and only if) the pilot clears the gates, scale to full folds/stride 1.

Usage:
    PYTHONPATH=src python scripts/liquidcap/lstm_pilot.py [--stride 3] [--epochs 3]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "v3"))

import _v3_harness as h  # noqa: E402

DATA = ROOT / "data" / "liquidcap"
CACHE = DATA / "cache" / "H_lstm_pilot"
SEQ_LEN = 60
TARGET = "fwd_ret_20d_sector_rel"


def build_sequences(panel: pd.DataFrame):
    """Per-ticker aligned arrays of (z-scored ret, log rel volume) + date index."""
    panel = panel.sort_values(["ticker", "date"])
    out = {}
    for tkr, g in panel.groupby("ticker"):
        ret = g["close"].pct_change().values
        vol = g["volume"].astype(float).values
        vol_ma = pd.Series(vol).rolling(60, min_periods=20).mean().values
        rel = np.log(np.where(vol_ma > 0, vol / vol_ma, 1.0).clip(0.05, 20.0))
        out[tkr] = (g["date"].values, ret, rel)
    return out


def make_xy(seqs, wanted: pd.DataFrame):
    """Materialize (N, SEQ_LEN, 2) inputs for the (date,ticker) rows in wanted."""
    X, keep = [], []
    for tkr, grp in wanted.groupby("ticker"):
        if tkr not in seqs:
            continue
        dates, ret, rel = seqs[tkr]
        pos = {d: i for i, d in enumerate(dates)}
        for row_i, d in zip(grp.index, grp["date"].values):
            i = pos.get(d)
            if i is None or i < SEQ_LEN:
                continue
            r = ret[i - SEQ_LEN + 1: i + 1]
            v = rel[i - SEQ_LEN + 1: i + 1]
            if np.isnan(r).any() or np.isnan(v).any():
                continue
            sd = r.std()
            X.append(np.stack([(r - r.mean()) / (sd if sd > 0 else 1.0), v], axis=1))
            keep.append(row_i)
    return np.asarray(X, dtype=np.float32), wanted.loc[keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=3, help="train-day subsampling")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()

    import torch
    from torch import nn
    torch.manual_seed(42)
    torch.set_num_threads(8)

    feats = pd.read_parquet(DATA / "features_liquidcap.parquet",
                            columns=["date", "ticker", TARGET, "atr_pct_20d",
                                     "close", "adv_usd_20d", "cs_spread_20d"])
    feats["date"] = pd.to_datetime(feats["date"])
    panel = pd.read_parquet(DATA / "ohlcv_sp500.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel.ticker != "SPY"]
    seqs = build_sequences(panel)

    # Semi-annual folds 2018+ (pilot compute budget)
    h.MIN_TRAIN_END = pd.Timestamp("2018-01-01")
    h.FOLD_DAYS = 126
    folds = h.define_folds(feats)
    all_dates = np.array(sorted(feats["date"].unique()))
    CACHE.mkdir(parents=True, exist_ok=True)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(2, 32, batch_first=True)
            self.head = nn.Linear(32, 1)

        def forward(self, x):
            _, (hn, _) = self.lstm(x)
            return self.head(hn[-1]).squeeze(-1)

    meta = []
    for i, fold in enumerate(folds):
        tr = feats[(feats.date >= fold["train_start"])
                   & (feats.date < fold["train_end"])].dropna(subset=[TARGET])
        prior = all_dates[all_dates < fold["test_start"]]
        if len(prior) > 20:
            tr = tr[tr["date"] < prior[-20]]
        tr_days = sorted(tr["date"].unique())[::args.stride]
        tr = tr[tr["date"].isin(tr_days)]
        Xtr, tr = make_xy(seqs, tr)
        y = tr[TARGET].values
        lo, hi = np.quantile(y, [0.01, 0.99])
        ytr = torch.tensor(np.clip(y, lo, hi), dtype=torch.float32)
        Xtr_t = torch.tensor(Xtr)

        net = Net()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        lossf = nn.MSELoss()
        n = len(Xtr_t)
        for ep in range(args.epochs):
            perm = torch.randperm(n)
            tot = 0.0
            for b in range(0, n, 2048):
                idx = perm[b:b + 2048]
                opt.zero_grad()
                out = net(Xtr_t[idx])
                loss = lossf(out, ytr[idx])
                loss.backward()
                opt.step()
                tot += float(loss) * len(idx)
            print(f"  fold {i+1} ep{ep+1} mse={tot/n:.6f} (n={n:,})", flush=True)

        te = feats[(feats.date >= fold["test_start"])
                   & (feats.date < fold["test_end"])].dropna(subset=[TARGET]).copy()
        Xte, te = make_xy(seqs, te)
        if len(te) == 0:
            continue
        with torch.no_grad():
            te["pred"] = net(torch.tensor(Xte)).numpy()
        ics = [spearmanr(g["pred"], g[TARGET])[0]
               for _, g in te.groupby("date") if len(g) >= 10]
        period = f"{fold['test_start']:%Y-%m}->{fold['test_end']:%Y-%m}"
        te.to_parquet(CACHE / f"fold_{i + 1:02d}.parquet", index=False)
        meta.append({"fold": i + 1, "period": period,
                     "test_start": str(fold["test_start"]), "test_end": str(fold["test_end"]),
                     "train_rows": len(tr), "test_rows": len(te),
                     "mean_ic": float(np.mean(ics)) if ics else 0.0,
                     "ic_ir": 0.0, "hit_rate_ic": 0.0})
        print(f"  Fold {i+1:2d} {period}: IC={meta[-1]['mean_ic']:+.4f}", flush=True)

    (CACHE / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n  cache -> {CACHE}")
    print("  evaluate with screen_liquidcap.replay('H_lstm_pilot', ...) against A")
    print(f"  Runtime {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
