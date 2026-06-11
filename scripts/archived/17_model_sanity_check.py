"""Sanity check production model: feature importance, OOS IC, leak heuristics."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def main() -> None:
    with open("data/models/smallcap_v3_lambdarank.pkl", "rb") as fh:
        m = pickle.load(fh)
    reg = json.loads(Path("data/paper_trading/model_registry.json").read_text())
    feats = reg["feat_cols"]

    f = pd.read_parquet("data/processed/features_smallcap.parquet")
    f["date"] = pd.to_datetime(f["date"])

    # ─── 1. OOS IC (last 6 months, in-sample for prod model — sanity only) ───
    recent = f[f["date"] >= "2025-11-01"].dropna(subset=["fwd_ret_20d_sector_rel"]).copy()
    recent["pred"] = m.predict(recent[feats].fillna(0).values)

    ics = []
    for _, g in recent.groupby("date"):
        if len(g) >= 10:
            ic, _ = spearmanr(g["pred"], g["fwd_ret_20d_sector_rel"])
            if not np.isnan(ic):
                ics.append(ic)
    print(f"IS-recent IC: mean={np.mean(ics):+.4f}  std={np.std(ics):.4f}  "
          f"days={len(ics)}  +days={sum(1 for x in ics if x > 0)}/{len(ics)}")

    # ─── 2. Top-K vs market median (2026 Q1, in-sample but ok proxy) ───
    oos = f[(f["date"] >= "2026-01-01") & (f["date"] < "2026-04-22")].dropna(
        subset=["fwd_ret_20d"]
    ).copy()
    oos["pred"] = m.predict(oos[feats].fillna(0).values)
    top_rets, mkt_rets = [], []
    for _, g in oos.groupby("date"):
        if len(g) < 50:
            continue
        top = g.nlargest(8, "pred")
        top_rets.append(top["fwd_ret_20d"].mean())
        mkt_rets.append(g["fwd_ret_20d"].median())
    if top_rets:
        print(f"2026Q1 top-8 mean 20d ret: {np.mean(top_rets):+.2%}  "
              f"vs market median: {np.mean(mkt_rets):+.2%}  "
              f"α: {np.mean(top_rets) - np.mean(mkt_rets):+.2%}  days={len(top_rets)}")

    # ─── 3. Leak heuristic: any feature with |Pearson r| > 0.05 vs target ───
    print()
    print("Feature ↔ target Pearson correlation (looking for any |r| > 0.05 = warning):")
    train = f.dropna(subset=["fwd_ret_20d_sector_rel"]).sample(
        min(100_000, len(f)), random_state=42
    )
    y = train["fwd_ret_20d_sector_rel"].values
    rows = []
    for fc in feats:
        if fc not in train.columns:
            continue
        x = pd.to_numeric(train[fc], errors="coerce").fillna(0).values
        if np.std(x) == 0:
            continue
        r = np.corrcoef(x, y)[0, 1]
        rows.append((fc, r))
    rows.sort(key=lambda x: -abs(x[1]))
    for fc, r in rows:
        flag = "  ⚠ HIGH" if abs(r) > 0.5 else ("  ?" if abs(r) > 0.05 else "")
        print(f"  {fc:25s}: {r:+.4f}{flag}")


if __name__ == "__main__":
    main()
