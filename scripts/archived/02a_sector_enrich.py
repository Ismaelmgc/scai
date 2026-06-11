"""V3 Step S1.2 — Fix sector classification.

Problem: 54% of feature rows have sector='Unknown' because 766/1042 universe
tickers have sic_code='N/A' (mostly delisted). This contaminates:
  - target: fwd_ret_20d_sector_rel (sector_avg dominated by Unknown bucket)
  - features: sector_ret_60d, ret_vs_sector_60d, sector_momentum_rank_*

Fix:
  1. Enrich universe via yfinance for N/A tickers
  2. Build enriched universe parquet
  3. Recompute sector column + sector-relative target + sector features in features parquet
  4. Save to data/processed/features_smallcap_v3_sector.parquet
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.app.features.sector import sic_to_sector

YF_TO_GICS = {
    "Healthcare": "Healthcare",
    "Financial Services": "Financials",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Technology": "Technology",
    "Industrials": "Industrials",
    "Communication Services": "Communication Services",
    "Real Estate": "Real Estate",
    "Energy": "Energy",
    "Utilities": "Utilities",
    "Basic Materials": "Materials",
}


def enrich_universe() -> pd.DataFrame:
    """Fetch sector from yfinance for tickers missing SIC mapping."""
    cache_fp = Path("data/processed/sector_enrichment_yf.parquet")
    universe = pd.read_parquet("data/processed/smallcap_universe.parquet")
    universe["sector_sic"] = universe["sic_code"].apply(sic_to_sector)

    need = universe[universe["sector_sic"] == "Unknown"]["ticker"].tolist()
    print(f"Tickers needing Yahoo enrichment: {len(need)}")

    if cache_fp.exists():
        cache = pd.read_parquet(cache_fp)
        print(f"Cache has {len(cache)} entries")
    else:
        cache = pd.DataFrame(columns=["ticker", "yf_sector"])

    already = set(cache["ticker"])
    to_fetch = [t for t in need if t not in already]
    print(f"Need to fetch from Yahoo: {len(to_fetch)}")

    if to_fetch:
        import yfinance as yf
        rows = []
        for i, t in enumerate(to_fetch):
            try:
                info = yf.Ticker(t).info
                sec = info.get("sector") or info.get("sectorKey") or None
            except Exception:
                sec = None
            rows.append({"ticker": t, "yf_sector": sec})
            if (i + 1) % 50 == 0:
                print(f"  fetched {i+1}/{len(to_fetch)}", flush=True)
                # periodic save
                tmp = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
                tmp.to_parquet(cache_fp, index=False)
            time.sleep(0.15)
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
        cache.to_parquet(cache_fp, index=False)

    # Merge: use SIC where present, else YF
    cache["yf_sector_gics"] = cache["yf_sector"].map(YF_TO_GICS).fillna("Unknown")
    universe = universe.merge(cache[["ticker", "yf_sector_gics"]], on="ticker", how="left")
    universe["sector"] = np.where(
        universe["sector_sic"] != "Unknown",
        universe["sector_sic"],
        universe["yf_sector_gics"].fillna("Unknown"),
    )
    print("\nFinal sector distribution:")
    print(universe["sector"].value_counts())
    return universe[["ticker", "sector"]]


def rebuild_features(sector_map: pd.DataFrame) -> Path:
    """Update sector + sector-relative target + sector features."""
    print("\nLoading features...")
    f = pd.read_parquet("data/processed/features_smallcap.parquet")
    f["date"] = pd.to_datetime(f["date"])
    f = f.drop(columns=["sector"]).merge(sector_map, on="ticker", how="left")
    f["sector"] = f["sector"].fillna("Unknown")
    new_dist = f["sector"].value_counts(normalize=True).head(15)
    print("Row-level sector distribution (new):")
    print(new_dist)

    # Recompute sector-relative target (fwd_ret_20d_sector_rel) — same logic as pipeline.py
    print("Recomputing fwd_ret_20d_sector_rel...")
    sector_avg = f.groupby(["date", "sector"])["fwd_ret_20d"].transform("mean")
    f["fwd_ret_20d_sector_rel"] = f["fwd_ret_20d"] - sector_avg

    # Recompute sector-relative features
    print("Recomputing sector_ret_60d / ret_vs_sector_60d...")
    f = f.sort_values(["ticker", "date"]).reset_index(drop=True)
    f["ret_1d"] = f.groupby("ticker")["close"].pct_change() if "close" in f.columns else f["ret_1d"]
    # Per-date sector mean of trailing 60d return
    f["ret_60d"] = f.groupby("ticker")["close"].pct_change(60) if "close" in f.columns else f["ret_60d"]
    sec_ret60 = f.groupby(["date", "sector"])["ret_60d"].transform("mean")
    f["sector_ret_60d"] = sec_ret60
    f["ret_vs_sector_60d"] = f["ret_60d"] - sec_ret60

    out = Path("data/processed/features_smallcap_v3_sector.parquet")
    f.to_parquet(out, index=False)
    print(f"Saved {out}  shape={f.shape}")
    return out


def main() -> None:
    sector_map = enrich_universe()
    out = rebuild_features(sector_map)
    sector_map.to_parquet("data/processed/universe_sector_v3.parquet", index=False)
    print(f"\nAll done. Use {out} in benchmark step.")


if __name__ == "__main__":
    main()
