# SCAI – Small Cap AI Trading Platform

> AI-powered platform for selecting, scoring, and backtesting US small-cap equities.

---

## Version Status

**V2 — CLOSED (2026-05-20)** — Frozen for rollback reference.
- LightGBM regressor, 26+5 features, Walk-forward: Sharpe 2.94, WR 44.88%
- Preserved: `data/models/smallcap_v2_secrel20d.pkl`

**V3 — PRODUCTION (2026-05-21)** — LambdaRank model deployed, dual paper trading active.

| Metric | V2 baseline | **V3 Production** | Δ vs V2 |
|---|---|---|---|
| Objective | MSE Regression | **LambdaRank** | rank-based |
| Features | 26+5=31 | **26+2+5=33** | +2 EDGAR |
| Mean Sharpe | 2.94 | **4.62** (Baseline) / **4.23** (Adaptive) | **+1.68 / +1.29** |
| Win Rate | 44.88% | **54.0%** (Baseline) / **64.4%** (Adaptive) | **+9.1pp / +19.5pp** |
| Selectivity median | **-2.42%** | **+2.37%** / **+4.71%** | **fixed** |
| Folds positive ret | 13/16 | **16/16** / 15/16 | +3 / +2 |
| Walk-forward trades | 1,600 | 1,600 | identical folds |

**V3 production config:**
- Model: LightGBM LambdaRank, 600 trees, `num_leaves=31, max_depth=6, min_child_samples=30, lr=0.05`
- Target: `fwd_ret_20d_sector_rel`, binned to 16 relevance levels, `lambdarank_truncation_level=8`
- Features: 33 (26 base + 2 EDGAR (`dilution_pct`, `current_ratio`) + 5 meta-learning)
- Backtest: TOP_K=8, hold 20d, rebalance 5d, ATR-clipped trailing stop [10%, 16%]
- Dual paper trading (since 2026-05-19, €1,000 each):
  - **Baseline**: standard trailing stop
  - **Adaptive Stop**: tighten to 6% after day 5 if position profitable (WR +10.3pp)

**V3 research (kept vs discarded):**
- ✅ LambdaRank objective (truncation_level=8, 16 bins) — replaces MSE regression
- ✅ Sector enrichment via yfinance — reduced Unknown rows 54% → 43.6%
- ✅ HP tuning winner (`v3_hp_combo`): SelMed crossed to positive (+0.47%)
- ✅ Adaptive stop: WR +10.3pp (52.75% → 63.06%), Sharpe 4.23
- ❌ Market-regime features — IC dropped, folds 14→12
- ❌ Feature pruning — low-gain features contribute under LambdaRank
- ❌ Multi-horizon target (5d+10d+20d) — short horizons predict reversals
- ❌ Score gate (σ filter) — zero effect with LambdaRank rankings

**Artifacts:**
- Benchmark harness: [scripts/v3/_v3_harness.py](scripts/v3/_v3_harness.py)
- Per-config JSONs: [data/v3_benchmarks/](data/v3_benchmarks/)
- WR benchmarks: [scripts/v3/14_wr_benchmarks.py](scripts/v3/14_wr_benchmarks.py)
- Period summary: [scripts/v3/15_period_summary.py](scripts/v3/15_period_summary.py)

---

## Overview

Para lanzar: `scai run` | Para dashboard: `scai web` (http://localhost:8501) | Para monitor intradía: `scai monitor`

SCAI is a modular platform that:

1. **Discovers** small-cap universe dynamically via Polygon.io
2. **Builds** 33 production features (technical, fundamental, meta-learning)
3. **Ranks** stocks using LightGBM LambdaRank (sector-relative returns)
4. **Selects** TOP-8 BUY signals via equal-weight ranking
5. **Paper trades** two strategies in parallel (Baseline + Adaptive Stop)
6. **Monitors** positions with ATR-adaptive trailing stops (daily + optional intraday)

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────────┐
│  DATA LAYER │───▶│FEATURE LAYER│───▶│ MODEL LAYER │───▶│  EXECUTION   │
│             │    │             │    │             │    │              │
│• Polygon.io │    │• Price act. │    │• LightGBM   │    │• TOP-8 rank  │
│• Yahoo Fin  │    │• Volatility │    │  LambdaRank │    │• Trail stops │
│• SEC EDGAR  │    │• Liquidity  │    │• 33 features│    │• ATR [10-16%]│
│• FRED       │    │• Momentum   │    │• 16 bins    │    │• Dual paper  │
│             │    │• Sector     │    │• 600 trees  │    │  trading     │
│• Parquet    │    │• EDGAR fund.│    │             │    │              │
│• DuckDB     │    │• Meta-learn │    │• Walk-fwd   │    │              │
└─────────────┘    └─────────────┘    └─────────────┘    └──────┬───────┘
                                                                │
                                       ┌─────────────┐    ┌─────▼────────┐
                                       │  DASHBOARD  │◀───│ PAPER TRADE  │
                                       │  • FastAPI  │    │  • Baseline  │
                                       │  • Chart.js │    │  • Adaptive  │
                                       │  • Dual tab │    │  • JSON state│
                                       └─────────────┘    └──────────────┘
```

## Key Principles

- **No temporal leakage**: All features and labels are strictly point-in-time (`as_of()`, `lag_safe_merge()`)
- **No random splits**: Only walk-forward cross-validation (16 folds)
- **No survivorship bias**: Universe includes delisted tickers
- **Realistic execution**: Slippage, spread proxy, ATR-adaptive trailing stops
- **OTC excluded** by default
- **Reproducible**: Global seed control (`SCAI_SEED=42`)

## Quick Start

### 1. Installation

```bash
git clone <repo-url> && cd SCAI
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configuration

```bash
cp .env.example .env
# Set: SCAI_POLYGON_API_KEY=your_key_here
```

### 3. Daily Pipeline (Production)

```bash
# Via CLI (recommended)
scai run

# Or manually
DYLD_LIBRARY_PATH=.local/lib PYTHONPATH=src python scripts/daily_pipeline.py
```

### 4. Dashboard

```bash
scai web    # → http://localhost:8501
```

### 5. Intraday Monitor

```bash
scai monitor
```

### 6. Analysis Pipeline (non-production)

```bash
DYLD_LIBRARY_PATH=.local/lib PYTHONPATH=src python scripts/run_smallcap_pipeline.py --skip-download --eval-holdout
```

### 7. Tests

```bash
PYTHONPATH=src pytest tests/unit -v --tb=short
```

## Web Dashboard

The dashboard (`scai web`, port 8501) shows:

- **Dual strategy tabs**: Baseline vs Adaptive Stop
- **Portfolio equity curve** per strategy (Chart.js)
- **Open positions** with entry price, current P&L, trail level
- **Signal history** with BUY signals and outcomes
- **Data stats**: OHLCV coverage, universe size, model info

## Project Structure

```
SCAI/
├── pyproject.toml          # Dependencies and project config
├── .env                    # API keys (SCAI_POLYGON_API_KEY)
├── PROJECT.md              # Full technical documentation
├── scripts/
│   ├── daily_pipeline.py   # ★ Production: V3 + dual paper trading
│   ├── intraday_monitor.py # Intraday trailing stop check
│   ├── run_smallcap_pipeline.py  # Analysis pipeline
│   ├── v3/                 # V3 research & benchmarks
│   └── archived/           # Historical research scripts
├── src/app/
│   ├── cli/main.py         # scai run | web | monitor
│   ├── config/             # Pydantic settings
│   ├── data/
│   │   ├── massive/        # Polygon.io client
│   │   ├── store/          # ParquetStore + DuckDB
│   │   └── free_sources/   # Yahoo, EDGAR, FINRA, FRED, Nasdaq
│   ├── features/           # Feature engineering (33 production)
│   │   ├── pipeline.py     # build_feature_matrix()
│   │   └── meta_features.py # Error-aware meta-learning
│   ├── models/
│   │   ├── multi_model.py  # LGB+XGB+CB ensemble (analysis)
│   │   └── tabular.py      # Single LGB model
│   ├── backtest/           # Walk-forward backtester
│   ├── paper_trading.py    # Dual strategy engine
│   ├── web/
│   │   ├── server.py       # FastAPI dashboard (port 8501)
│   │   └── templates/      # Jinja2 dark theme, dual tabs
│   └── utils/              # Logging, seeds, point-in-time
├── data/
│   ├── processed/          # OHLCV + features (parquet)
│   ├── models/             # V3 LambdaRank model (.pkl)
│   ├── paper_trading/      # Baseline + adaptive/ portfolios
│   └── v3_benchmarks/      # Walk-forward benchmark JSONs
└── tests/
    ├── unit/
    └── integration/
```

## Configuration

All settings can be overridden via environment variables (prefix `SCAI_`):

```bash
export SCAI_SEED=42
export SCAI_ENV=development
export SCAI_POLYGON_API_KEY=...  # in .env
```

See `src/app/config/__init__.py` for the full list.

## Running Tests

```bash
PYTHONPATH=src pytest tests/unit -v --tb=short
```

## License

MIT
- **Sector ETFs trend**: XLF, XLE, XLV, XLK, XLI etc. — confirm sector momentum

**All these are free on Yahoo. ~10 new features, no new infra needed.**

#### 2. FIX SECTOR CLASSIFICATION
Currently `assign_sectors()` exists but the universe parquet lacks the column.
- Re-fetch SIC codes from EDGAR or use FMP/Yahoo Finance sector
- Build proper GICS sector mapping
- Verify `sector_ret_60d` and `ret_vs_sector_60d` features are actually informative (currently degraded)

#### 3. STRENGTHEN SCORE CALIBRATION (model architecture)
Single LGB regressor is too noisy. Options to test:
- **Quantile regression** (predict P25/P50/P75 instead of mean) — better tail estimation
- **Classification head**: P(top-decile-next-20d) instead of regression
- **Ensemble** (LGB + XGBoost + CatBoost averaged or stacked) — already have `MultiModelEnsemble`
- **Two-stage model**: Stage 1 = regime filter (will the small-cap basket trend up?); Stage 2 = stock selection
- **Loss function**: try ranking loss (LambdaRank/LambdaMART) instead of MSE — we only care about top-K ordering

#### 4. ADD QUALITY/MICROSTRUCTURE FEATURES
Real edge in small-caps comes from microstructure that retail can't price:
- **Short interest** (FINRA): % float short, days-to-cover, short squeezes are HUGE in small-caps
- **Insider transactions** (EDGAR Form 4): cluster buys are predictive
- **Institutional ownership change** (13F): smart money flows
- **Borrow rate / hard-to-borrow**: signals scarcity
- **Trading halts / LULD bands**: volatility regime markers
- **Bid-ask spread (real, not proxy)**: requires intraday data
- **Earnings dates**: blackout windows pre/post earnings reduce noise

#### 5. ALTERNATIVE TARGETS / HORIZONS
Test:
- **Multi-horizon ensemble**: 5d + 10d + 20d targets averaged
- **Risk-adjusted target**: `fwd_ret_20d / atr_20d` (Sharpe-like)
- **Probabilistic target**: P(ret > +5% AND no -10% DD in next 20d)
- **Triple-barrier method** (Marcos López de Prado): label = first barrier hit (PT/SL/timeout)

#### 6. RISK & POSITION MANAGEMENT
Current: equal-weight 12.5% × 8 with simple trailing stop.
- **Volatility-targeted sizing**: each position contributes equal risk (1/N risk budget)
- **Correlation-aware sizing**: penalize signals from correlated stocks (cluster on sector/factor)
- **Dynamic top-K**: trade fewer positions when confidence (model score spread) is low
- **Regime-based exposure**: when VIX > 25 or breadth weak, reduce gross exposure to 50%
- **Profit-take rules**: lock half at +20%, let runners go

#### 7. ALTERNATIVE DATA (after free-data exhausted)
- **News sentiment** (free: Yahoo News RSS, Reuters; paid: Polygon News, RavenPack)
- **Reddit/StockTwits mentions** (free APIs available, very noisy but predictive in small-caps)
- **Google Trends** (free, ticker symbol searches)
- **Options activity** (unusual call/put ratio, IV percentile) — most small-caps have no options

#### 8. EXECUTION REALISM
Current backtest assumes execution at next-day open with 15bps cost.
- **Liquidity penalty**: penalize stocks with low ADV (price impact = sqrt(participation))
- **Slippage by volatility**: slippage = f(spread, vol, participation)
- **Borrow availability** (for shorts in future): check Interactive Brokers borrow list
- **Capacity test**: would strategy work with €10K, €100K, €1M?

### Recommended V3 First-Sprint Tasks (quick wins)

1. **Add IWM/SPY/VIX features** (1-2 days, just Yahoo download + features)
2. **Fix sector column in universe** (1 day, EDGAR SIC mapping)
3. **Retrain V2 with these 12+ new features → measure IC delta**
4. **Test LambdaRank objective vs regression** (1 day, swap objective in LGB)

These four together should move IC from 0.053 → 0.07-0.10 and stabilize bear-regime performance. Once measured, decide on architecture changes (#3).

### Anti-pattern: things NOT to do in V3
- Don't add more features blindly (>50 already, marginal value diminishing)
- Don't use deep learning yet (data too small, 945K rows is barely enough for LGB)
- Don't optimize hyperparameters without first fixing structural issues
- Don't expand universe further (1042 tickers is plenty; quality > quantity)

