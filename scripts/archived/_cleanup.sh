#!/usr/bin/env bash
# V3 cleanup script
set -e
cd "$(dirname "$0")/../.."

echo "=== Removing unused V3 exploration scripts ==="
rm -f scripts/v3/02_sector_bench.py
rm -f scripts/v3/03_regime_bench.py
rm -f scripts/v3/03b_market_features.py
rm -f scripts/v3/04_lambdarank_bench.py
rm -f scripts/v3/05_candidate_s1.py
rm -f scripts/v3/06_edgar_sector.py
rm -f scripts/v3/07_feature_select.py
rm -f scripts/v3/08_lean_bench.py
rm -f scripts/v3/10_multi_horizon.py
rm -f scripts/v3/_diag.py
rm -f scripts/v3/_inspect.py
rm -f scripts/v3/_validate_sector.py
rm -rf scripts/v3/__pycache__

echo "=== Remaining V3 scripts ==="
ls scripts/v3/

echo "=== Removing unused heavy data files ==="
rm -f data/processed/features_smallcap_v3_regime.parquet
# NOTE: features_smallcap.parquet is the V3 production features file (renamed from v3_sector). DO NOT delete.
rm -f data/processed/signals_smallcap_2026.parquet
rm -f data/processed/signals_smallcap_2026_holdout.parquet
rm -f data/processed/ohlcv_smallcap_yahoo.parquet
rm -f data/processed/smallcap_news.parquet

echo "=== Remaining processed/ ==="
du -sh data/processed/* | sort -h
echo "=== Total data size ==="
du -sh data/
