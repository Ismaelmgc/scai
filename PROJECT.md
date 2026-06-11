# SCAI — Small Cap AI Trading Platform

> Documentación completa del proyecto para contexto de sesiones de IA.
> Última actualización: 2026-05-21

---

## 1. Visión General

SCAI es una plataforma de trading algorítmico basada en machine learning para acciones **US small-cap** ($50M–$2B de capitalización). El sistema:

1. **Descubre** dinámicamente un universo de acciones small-cap via Polygon.io API
2. **Descarga** datos OHLCV, corporate actions y fundamentales (incremental)
3. **Construye** una matriz de ~150+ features técnicos, fundamentales y de microestructura
4. **Entrena** un modelo LightGBM LambdaRank sobre sector-relative returns (20d)
5. **Genera señales** BUY rankeando por score LambdaRank, top-8 equal-weight
6. **Paper trades** con trailing stops ATR-adaptativos (dos estrategias en paralelo)
7. **Presenta** resultados en un dashboard web FastAPI + Chart.js con tabs por estrategia

**Capital actual**: €1,000 (paper trading, desde 2026-05-19)
**Modelo producción**: V3 LambdaRank (33 features, 600 trees)

---

## 2. Stack Tecnológico

| Componente | Tecnología | Versión |
|---|---|---|
| Runtime | Python | 3.11.8 |
| OS | macOS arm64 (Apple Silicon) | — |
| Entorno | venv en `.venv/` | — |
| Data API | Polygon.io (free plan) | REST v3 |
| ML | LightGBM, XGBoost, scikit-learn | 4.1+, >=2.0, >=1.3 |
| DataFrame | pandas + pyarrow | 2.1+ |
| Storage | Parquet (ParquetStore) + DuckDB | — |
| Web | FastAPI + Jinja2 + Chart.js | 0.136+ |
| HTTP | httpx (async-capable, sync used) | >=0.25 |
| Config | pydantic-settings | — |
| Logging | structlog | >=23.2 |
| Testing | pytest + pytest-cov | >=7.4 |
| Build | hatchling | — |

---

## 3. Estructura del Proyecto

```
SCAI/
├── .env                          # API keys (SCAI_POLYGON_API_KEY=...)
├── Makefile                      # Comandos: install, test, run-api, lint, etc.
├── pyproject.toml                # Dependencias y configuración del proyecto
├── PROJECT.md                    # ← ESTE ARCHIVO
│
├── configs/
│   └── com.scai.daily.plist      # macOS LaunchAgent (daily automation)
│
├── scripts/
│   ├── run_smallcap_pipeline.py  # Pipeline análisis + backtest (~2000 líneas)
│   ├── daily_pipeline.py         # ★ PRODUCCIÓN: V3 LambdaRank + dual paper trading
│   ├── intraday_monitor.py       # Monitor intradía trailing stops (Polygon snapshots)
│   ├── v3/                       # V3 research & benchmarks
│   │   ├── _v3_harness.py        # Shared 16-fold walk-forward harness
│   │   ├── 14_wr_benchmarks.py   # Win rate improvement benchmarks (6 configs)
│   │   ├── 15_period_summary.py  # Period-by-period analysis script
│   │   └── ...                   # Sprint 1-2 research scripts
│   └── archived/                 # Scripts de diagnóstico y sweeps (no producción)
│
├── src/app/                      # Código fuente principal
│   ├── config/__init__.py        # Settings (pydantic-settings, .env)
│   ├── utils/
│   │   ├── __init__.py           # Logging, seed, helpers
│   │   └── point_in_time.py      # as_of(), lag_safe_merge() — anti-leakage
│   │
│   ├── data/
│   │   ├── massive/              # Cliente Polygon.io (rate limiting, retries)
│   │   ├── store/parquet_store.py  # ParquetStore: read/write/upsert/query via DuckDB
│   │   └── providers/factory.py  # Provider factory
│   │
│   ├── features/                 # Feature engineering (~150+ features)
│   │   ├── pipeline.py           # Orquestador: build_feature_matrix() + labels
│   │   ├── price_action.py       # Retornos, gaps, overnight/intraday, reversals
│   │   ├── momentum.py           # SMA/EMA, MACD, RSI, Bollinger, ADX, Ichimoku
│   │   ├── volatility.py         # Vol realizada, ATR, beta, vol idiosincrática
│   │   ├── liquidity.py          # Dollar volume, Amihud, spread proxy, capacidad
│   │   ├── market_regime.py      # Régimen de mercado (SPY)
│   │   ├── microstructure.py     # VWAP deviation, OBV, Corwin-Schultz
│   │   ├── sector.py             # SIC→sector, retornos relativos, rotación
│   │   ├── cross_sectional.py    # Rank percentil, z-score por fecha
│   │   ├── fundamentals.py       # Márgenes, ROE, apalancamiento desde XBRL
│   │   ├── alpha_features.py     # Autocorrelación, skewness, vol-of-vol
│   │   └── sentiment.py          # Per-ticker sentiment (Polygon News API)
│   │
│   ├── models/                   # Modelos ML
│   │   ├── multi_model.py        # MultiModelEnsemble: LGB+XGB+CB → average/Ridge
│   │   ├── tabular.py            # TabularModel: single LGB con stacking opcional
│   │   └── feature_selection.py  # Gain importance + correlation pruning
│   │
│   ├── backtest/__init__.py      # Backtester: ejecución realista con stops
│   ├── paper_trading.py          # Paper trading engine: JSON state, trailing stops, adaptive stop
│   ├── reporting/__init__.py     # Text + HTML reports
│   │
│   ├── cli/
│   │   └── main.py               # CLI: scai run | scai web | scai monitor
│   │
│   └── web/
│       ├── server.py             # FastAPI dashboard (puerto 8501), dual strategy tabs
│       └── templates/dashboard.html  # Dark theme, tabs Baseline vs Adaptive Stop
│
├── data/
│   ├── processed/                # Datos procesados (parquet)
│   ├── models/                   # smallcap_v3_lambdarank.pkl (producción)
│   ├── paper_trading/            # Baseline portfolio, signals, logs
│   │   └── adaptive/             # Adaptive stop portfolio (separate state)
│   └── v3_benchmarks/            # Walk-forward benchmark JSONs (16 configs)
│
├── tests/
│   ├── unit/                     # 8 archivos: features, models, config, store, massive, etc.
│   └── integration/              # Pipeline end-to-end
│
└── reports/                      # Resultados de análisis (.md, .json)
```

---

## 4. Modelo V3 (Producción)

### 4.1 Configuración Validada

Resultado de Walk-Forward leak-free (16 folds, 2022-06 → 2026-05, out-of-sample, ver § 10 Anti-Leak Protocol):

| Métrica | Valor |
|---|---|
| **Modelo** | LightGBM LambdaRank |
| **Target** | `fwd_ret_20d_sector_rel` (sector-relative 20d return, binned to 16 levels) |
| **Features** | **28** (26 base + 2 EDGAR — **0 meta**, ver § 4.6) |
| **Mean IC (WF)** | +0.0115 (cross-sectional Spearman) |
| **Mean Sharpe (WF)** | aprox. +3 (validado por fold, ver `data/v3_benchmarks/`) |
| **Folds con IC > 0** | 14/16 |
| **Mean Return / fold** | +26.0 % (no_meta, leak-free) |
| **Trades validados** | ~ 1,600 (out-of-sample, ~4 años) |
| **Trees** | 600 |

> **Importante**: Cifras anteriores (Sharpe 4.62, ret +24,106 %, 16/16 folds positivos) eran resultado de un **data leak** en meta-features y de compounding incorrecto sobre cohortes solapadas. Quedaron descartadas el 2026-05-22 tras la auditoría descrita en § 10.

### 4.2 Features V3 (28 columnas)

```python
# 26 base features (trailing rolling per-ticker o cross-sectional same-date)
V3_FEATURES_BASE = [
    'max_dd_60d', 'vol_of_vol_60d', 'ret_kurtosis_60d', 'avg_trade_size_20d',
    'obv', 'obv_vs_sma_60d', 'amihud_60d', 'downside_vol_60d', 'ema_26',
    'spread_proxy_20d', 'ret_skew_60d', 'ret_252d', 'sma_200', 'sector_ret_60d',
    'realized_vol_120d', 'macd_hist', 'pct_from_52w_low', 'adv_60d',
    'ret_vs_sector_60d', 'vol_of_vol_ratio', 'price_roc_smooth_120d',
    'vwap_dev_avg_20d', 'reversal_20v60', 'vol_regime_change', 'beta_60d',
    'macd_signal',
]
# 2 EDGAR fundamentals (point-in-time via merge_asof backward sobre filing_date)
V3_EDGAR_FEATURES = ['dilution_pct', 'current_ratio']
# 0 meta-learning features (RETIRADAS 2026-05-22 — A/B sin valor cuando leak-free)
V3_META_FEATURES: list[str] = []
```

**Importancia (gain) de los top-5**: `avg_trade_size_20d` (17.3%), `amihud_60d` (8.2%), `realized_vol_120d` (4.9%), `ema_26` (4.9%), `spread_proxy_20d` (4.8%). Las 28 features tienen gain > 0 (ninguna inerte).

### 4.3 Hiperparámetros

```python
V3_LGB_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg",
    "num_leaves": 31, "max_depth": 6,
    "learning_rate": 0.05, "min_child_samples": 30,
    "subsample": 0.75, "colsample_bytree": 0.7,
    "reg_lambda": 5.0,
    "lambdarank_truncation_level": 8,
    "label_gain": list(range(16)),  # 16 relevance bins
    "n_jobs": 1, "seed": 42, "verbose": -1,
}
N_ROUNDS = 600
```

### 4.4 Señales y Ejecución

- **Tradability gate (v3.2, 2026-06-11)**: antes del ranking se excluyen tickers no operables — `close < $1.50`, `ADV20 < $500K` o datos stale (> 4 días). Módulo compartido `src/app/features/tradability.py`, usado idénticamente por producción (`daily_pipeline.py`) y el harness (`_v3_harness.py`). El filtro aplica SOLO a la selección; el entrenamiento mantiene deslistados (anti-survivorship). Motivo: el paper trading en vivo compró zombies sub-penny (SRNE @ $0.0006, $153/día de volumen) porque nada re-validaba la operabilidad tras el snapshot del universo. Umbrales validados con sweep 3×3 (`scripts/v3/20_filter_sweep.py`); baseline honesto: `data/v3_benchmarks/v4_filt_baseline.json` (+12.7%/fold, Sharpe 2.52, WR 51.3%, 14/16 folds, coste 15bps/lado).
- **Ranking**: por score LambdaRank (mayor = mejor)
- **Top-K**: 8 posiciones
- **Sizing**: equal-weight (12.5% cada una)
- **Rebalanceo**: cada 5 días hábiles
- **Holding period**: 20 días
- **Trailing stops**: ATR-adaptativos, clipped [10%, 16%]
- **Cooldown**: 5 días de bloqueo tras salida por trailing stop (evita re-entry inmediato)

### 4.5 Dual Strategy Paper Trading

Dos estrategias se ejecutan en paralelo con portfolios separados:

| Estrategia | Trailing Stop | WR (backtest) | Sharpe | Portfolio |
|---|---|---|---|---|
| **Baseline** | ATR × 5.3, clipped [10%, 16%] | 54.0% | 4.62 | `data/paper_trading/portfolio.json` |
| **Adaptive Stop** | Igual, pero tighten a 6% tras día 5 si profitable | 64.4% | 4.23 | `data/paper_trading/adaptive/portfolio.json` |

Paper trading activo desde **2026-05-19**. Meta: 3+ meses (~200 trades) para validación.

### 4.6 Evolución V2 → V3

| Cambio | Resultado | Evidencia |
|---|---|---|
| **LambdaRank** reemplaza regressor | Sharpe +0.77, SelMed cruzó a positivo | WF 16 folds: 2.94 → 3.71 |
| **Sector enrichment** (yfinance) | Unknown rows 54% → 43.6% | Mejores sector_ret_60d / ret_vs_sector_60d |
| **HP tuning** (leaves=31, depth=6, lr=0.05, mc=30, 600 trees) | SelMed +0.47% (primera vez positivo) | v3_hp_combo ganador de 7 configs |
| **Adaptive stop** (tighten 6% after day 5) | WR +10.3pp (52.75% → 63.06%) | WR benchmark 6 configs |
| Market-regime features | ❌ Descartado | IC drop, folds positivos 14→12 |
| Feature pruning | ❌ Descartado | Low-gain features aportan bajo LambdaRank |
| Multi-horizon target | ❌ Descartado | Short horizons predicen reversals |
| Score gate (σ filter) | ❌ Descartado | Zero effect con LambdaRank rankings |

---

## 5. Pipeline Principal (`scripts/run_smallcap_pipeline.py`)

### 5.1 Flujo de Ejecución

```
STEP 1 → Descubrimiento de universo (discover_universe)
STEP 2 → Descarga OHLCV incremental (download_ohlcv)
STEP 2b → Filtro de calidad post-OHLCV (filter_universe_quality)
STEP 3 → Corporate actions + fundamentales + SPY
STEP 4 → Construcción de features (build_feature_matrix + lags)
STEP 5 → Entrenamiento de modelos (train_models)
STEP 6 → Señales + backtest validación
STEP 7 → Evaluación holdout (solo con --eval-holdout)
```

### 5.2 Argumentos CLI

```bash
python scripts/run_smallcap_pipeline.py \
  --train-start 2020-01-01 \
  --predict-from 2026-01-02 \
  --holdout-from 2026-04-01 \
  --predict-to 2026-05-06 \
  --max-tickers 400 \
  --top 15 \
  --eval-holdout \
  --skip-download
```

---

## 6. Daily Pipeline (`scripts/daily_pipeline.py`)

Pipeline de producción diario con el modelo V3 LambdaRank.

### 6.1 Flujo

```
STEP 1 → Descarga incremental OHLCV + SPY
STEP 2 → Rebuild features (incluye meta-features)
STEP 3 → Train (LGB LambdaRank, 600 trees) o Load modelo V3
STEP 4 → Generar señales (rank por score LambdaRank, top-8 BUY)
STEP 5 → Paper trading dual:
         5a → Baseline (trail ATR [10-16%])
         5b → Adaptive Stop (tighten 6% after day 5 if profitable)
```

Modelo: `data/models/smallcap_v3_lambdarank.pkl`

### 6.2 Uso

```bash
# Ejecución diaria normal (CLI)
scai run

# Equivalente manual
DYLD_LIBRARY_PATH=.local/lib PYTHONPATH=src python scripts/daily_pipeline.py

# Dashboard web
scai web    # → http://localhost:8501

# Monitor intradía de trailing stops
scai monitor

# Forzar reentrenamiento
DYLD_LIBRARY_PATH=.local/lib PYTHONPATH=src python scripts/daily_pipeline.py --force-retrain
```

### 6.3 CLI (`scai`)

Instalado via `pip install -e .` (entry point en `pyproject.toml`):

| Comando | Descripción |
|---|---|
| `scai run` | Ejecuta daily pipeline completo |
| `scai web` | Lanza dashboard FastAPI (puerto 8501) |
| `scai monitor` | Check intradía trailing stops via Polygon snapshots |

### 6.4 Automatización (macOS)

```bash
launchctl load ~/Library/LaunchAgents/com.scai.daily.plist  # L-V 22:00 UTC
```

---

## 7. Sistema de Datos

### 7.1 Polygon.io API

**Plan**: Free (5 calls/min, ~2 años de histórico)

**Cliente** (`src/app/data/massive/client.py`):
- Rate limiter: 5 calls/min, retries con exponential backoff
- Paginación automática, timeout 30s

### 7.2 ParquetStore

- Ruta base: `data/processed/`
- APIs: `write()`, `upsert()`, `read_as_of()`, `query()` (DuckDB)

### 7.3 Datos Actuales

| Dominio | Filas | Tickers | Rango |
|---|---|---|---|
| `ohlcv_smallcap` | 830,068 | 1,048 | 2021-05-20 → 2026-05-20 |
| `features_smallcap` | 830,068 | 1,048 | 1,256 fechas |
| `smallcap_universe` | ~324 | 324 | snapshot actual |

---

## 8. Feature Engineering (~150+ features)

### 8.1 Categorías

| Módulo | Features | Descripción |
|---|---|---|
| `price_action.py` | ~15 | Returns multi-horizonte, gaps, overnight/intraday |
| `momentum.py` | ~30+ | SMA/EMA, MACD, RSI, Bollinger, ADX, Ichimoku |
| `volatility.py` | ~15 | Vol realizada, ATR, downside deviation, beta |
| `liquidity.py` | ~15 | Dollar volume, Amihud, spread proxy, capacidad |
| `microstructure.py` | ~12 | VWAP deviation, OBV, Corwin-Schultz |
| `sector.py` | ~10 | Sector assignment, retornos relativos, rotación |
| `cross_sectional.py` | ~15 | Rank percentil, z-score, sector-relative |
| `fundamentals.py` | ~20 | Márgenes, ROE, apalancamiento (XBRL) |
| `alpha_features.py` | ~15 | Autocorrelación, skewness, vol-of-vol |
| `sentiment.py` | ~6 | Per-ticker news sentiment (Polygon News) |
| Lag features (pipeline) | ~25 | Lagged returns, momentum acceleration |

### 8.2 Labels (Targets)

| Target | Tipo | Uso |
|---|---|---|
| `fwd_ret_Xd` | Regression | Forward return a X días |
| `fwd_ret_Xd_sector_rel` | Regression | **V2 target** (sector-relative return) |
| `fwd_ret_Xd_positive` | Classification | Análisis (no producción) |

### 8.3 Feature Selection (V2)

Selección estable multi-seed:
1. Entrenar LGB con 3 seeds × 3 targets (secrel_5d, 10d, 20d)
2. Rank importance por gain en cada run
3. Mean rank across 9 runs → top-30
4. Correlation pruning (|corr| > 0.85)
5. Resultado: **26 features finales**

---

## 9. Backtest y Paper Trading

### 9.1 Configuración

```python
initial_capital = 1000      # €1,000
max_positions = 8
holding_period = 20 days
trailing_stop = 10-16%      # ATR-adaptive, clipped
commission = 5 bps/side
slippage = 10 bps
```

### 9.2 Paper Trading Engine (Dual Strategy)

- **Baseline**: `data/paper_trading/portfolio.json` — trailing stop ATR × 5.3, clipped [10%, 16%]
- **Adaptive Stop**: `data/paper_trading/adaptive/portfolio.json` — tighten trail to 6% after day 5 if position profitable
- Señal al close(t) → ejecución al open(t+1)
- **Cooldown 5d**: tras trailing stop exit, ticker bloqueado 5 días hábiles (+42pp en backtest)
- Trailing stops monitoreados diariamente + opción intradía (`scai monitor`)
- Trade log en `daily_log.jsonl` + signal history en parquet
- Dashboard web con tabs por estrategia (`scai web`)
- Paper trading activo desde **2026-05-19** (€1,000 cada estrategia)

### 9.3 Walk-Forward Benchmark Results (16 folds, out-of-sample, **leak-free**)

Resultados honestos tras corrección de leak en meta-features y de compounding solapado (ver § 10).

| Configuración | Mean IC | +Folds | Mean Return / fold | Mean WR | Notas |
|---|---|---|---|---|---|
| **no_meta (PRODUCCIÓN)** | +0.0115 | 14/16 | +26.0 % | ~48 % | Modelo actual `smallcap_v3_lambdarank.pkl` |
| meta_30d | +0.0048 | 13/16 | +20.0 % | ~47 % | Probado, descartado |
| meta_45d | +0.0080 | 14/16 | +18.2 % | ~47 % | Probado, descartado |
| meta_60d | +0.0087 | 14/16 | +19.7 % | ~48 % | Probado, descartado |

> Una "fold ret" cubre ~3 meses (FOLD_DAYS = 63). El compounding suma N_COHORTS = 4 streams no solapadas (HOLD_DAYS // REBALANCE_EVERY).
> Benchmark data: `data/v3_benchmarks/no_meta.json`, `meta_30d.json`, `meta_45d.json`, `meta_60d.json`
> Mercado (IWM Russell 2000 / SPY) en el mismo periodo: +52 % / +82 % acumulado (46 meses), ver `scripts/v3/15_period_summary.py`.

---

## 10. Anti-Leak Protocol (OBLIGATORIO antes de promover un modelo)

El 2026-05-22 se detectó un data leak grave en V3: meta-features filtraban valores futuros (signal whose 20d outcome reaching into the future), inflando los retornos validados a niveles imposibles (+24,000 % vs mercado +1 %). Se retiraron las meta-features y se estableció el siguiente protocolo. **No saltarlo nunca**.

### 10.1 Reglas de feature design

1. **Features = información disponible a las T 00:00**. Cualquier valor que dependa de precios, volúmenes o eventos en `(T, T+H]` es un label, no un feature.
2. **Rolling temporal**: SIEMPRE per-ticker (`groupby('ticker').transform(rolling(...))`). Nunca `groupby('sector')` con rolling temporal (mezcla órdenes de filas entre tickers).
3. **Cross-sectional**: SIEMPRE same-date (`groupby('date')...` o `groupby(['date','sector'])...`). Nunca mezclar fechas.
4. **Datos externos (EDGAR, fundamentales, noticias)**: SIEMPRE `merge_asof(direction='backward')` sobre `filing_date` (fecha de publicación real, no `period_end`). Lag mínimo = 1 día hábil si la fuente publica intradía.
5. **Features derivadas de signal history**: el `signal_date` de cada registro histórico debe cumplir `signal_date + horizon + buffer < current_date` (para `horizon=20d`, exigir `signal_date < current_date − 30 días calendario`).
6. **Prefijos/sufijos prohibidos** en `feat_cols`: `fwd_*`, `forward_*`, `future_*`, `actual_ret*`, `tb_label*`, `*_positive`, `*_xsec_positive`. Estos son labels.
7. **El target nunca aparece** en `feat_cols`. Verificar con `assert TARGET not in feat_cols`.

### 10.2 Gate automático (`scripts/v3/18_verify_no_leak.py`)

Ejecutado automáticamente por `scripts/v3/12_train_v3_production.py` **antes** de guardar el modelo. Aborta el entrenamiento si cualquier check falla.

| Check | Umbral | Razón |
|---|---|---|
| Nombres prohibidos | 0 features con prefijo/sufijo de label | Detecta inclusión accidental de un target |
| Features degeneradas | 0 features con < 2 valores únicos | Detecta columnas vacías o constantes |
| `|Pearson r|` panel | < 0.10 vs target | Features sin leak tienen `|r|` en `[0, 0.05]` |
| Median `|Spearman ρ|` per-date | < 0.15 | Captura leaks que solo se ven cross-sectionally |

Umbrales validados en producción actual (28 features): max `|r| = 0.034` (`avg_trade_size_20d`), todas las demás < 0.025.

### 10.3 Validación walk-forward

Cualquier feature/cambio nuevo debe pasar por `scripts/v3/_v3_harness.py` (16 folds, embargo implícito por `dropna(subset=[target])`) y mejorar los criterios:

- IC medio sin caer (baseline actual: +0.0115)
- Folds con IC > 0 sin caer (baseline: 14/16)
- Mean return / fold sin caer (baseline: +26 %)
- Compounding correcto (split en `N_COHORTS = HOLD_DAYS // REBALANCE_EVERY` streams no solapadas, ver `_v3_harness.py`)

### 10.4 Anti-bias clásico (ya implementado)

1. **Survivorship bias**: catálogo incluye 5,980 tickers delisted, excluye SPACs (SIC 6770).
2. **Look-ahead bias**: `as_of()`, `lag_safe_merge()`, fundamentals por `filing_date`.
3. **Data quality**: min 250 days, price > $1.50, ADV20 > $300K.
4. **Overfitting**: Walk-Forward CV (16 folds), feature importance gain > 0 obligatoria.

---

## 11. Limitaciones Conocidas

1. **Datos históricos limitados**: ~5 años (2021-2026), mayormente bull market
2. **WR Baseline < 55%** en períodos bear (fold 5: 37.5%, fold 11: 34.6%)
3. **Capital pequeño** (€1,000): comisiones mínimas limitan sizing real
4. **Market cap snapshot**: API devuelve market cap actual, no point-in-time
5. **Retornos backtest inflados**: compounding teórico sin costes. En real con €1K serán mucho menores
6. **Paper trading recién iniciado**: solo desde 2026-05-19 (2 días). Mínimo 3 meses para validar

---

## 12. Comandos Frecuentes

```bash
# IMPORTANTE: siempre incluir DYLD_LIBRARY_PATH para ML
# Pipeline diario (producción)
scai run
# o manualmente:
DYLD_LIBRARY_PATH=.local/lib PYTHONPATH=src python scripts/daily_pipeline.py

# Dashboard web
scai web    # → http://localhost:8501

# Monitor intradía
scai monitor

# Pipeline de análisis (no producción)
DYLD_LIBRARY_PATH=.local/lib PYTHONPATH=src python scripts/run_smallcap_pipeline.py --skip-download

# Tests
PYTHONPATH=src pytest tests/unit -v --tb=short
```

---

## 13. Variables de Entorno

```bash
SCAI_POLYGON_API_KEY=...    # en .env
SCAI_ENV=development
SCAI_SEED=42
```

---

## 14. Documentación de Estrategia (`docs/strategy.html`)

HTML standalone que documenta la estrategia completa: universo, features, modelo, señales, ejecución.

**Mantenimiento**: Regenerar `docs/strategy.html` cada vez que cambie alguno de:
- Parámetros del modelo (TOP_K, holding, LGB params)
- Lista de features (V2_FEATURES)
- Feature importance (tras reentrenamiento significativo)
- Configuración de trailing stops o position sizing
- Resultados de validación (walk-forward CV)

Esto garantiza que el HTML refleja siempre el estado actual del modelo en producción.

---

## 15. Free Data Sources (`src/app/data/free_sources/`)

Conectores para fuentes de datos gratuitas que complementan Polygon/Massive:

| Conector | Función | Uso |
|----------|---------|-----|
| `yahoo.py` | OHLCV histórico (5+ años) | Extensión temporal, backfill |
| `sec_edgar.py` | Fundamentales XBRL (SEC filings) | Features: cash_runway, dilution, revenue_growth |
| `finra.py` | Short sale volume (RegSHO) | Features: short_ratio, spike, pressure |
| `fred.py` | Macro (tasas, VIX, spreads) | Features: yield_curve, vix_regime, credit_stress |
| `nasdaq_trader.py` | Directorio de símbolos | Filtrado: ETFs, test issues, financial status |

**Hallazgos clave** (2026-05-19):
- Yahoo quality score 0.72 vs Massive (diff por dividend adjustment, no error de datos)
- 2 de 26 features V2 (`avg_trade_size_20d`, `vwap_dev_avg_20d`) no disponibles en Yahoo
- Hybrid (Yahoo + Massive) mejora top-8 return +40% pero reduce IC -38%
- Ver `reports/free_data_sources_plan.md` para evaluación completa

---

## 16. Reglas de Desarrollo

### 14.1 Reglas Críticas

1. **SIEMPRE** incluir `DYLD_LIBRARY_PATH=.local/lib` al ejecutar scripts ML. Sin esto → segfault por libomp.
2. **PYTHONPATH=src** necesario para imports (`from app.config import ...`).
3. **n_jobs=1** en todos los modelos ML (evita conflicto libomp multi-threading).
4. **Anti-leakage**: usar `as_of()` y `lag_safe_merge()`. Nunca usar datos futuros.

### 14.2 Estándares de Código

1. **No crear archivos innecesarios**: extender código existente en vez de crear nuevos módulos.
2. **No dejar código muerto**: eliminar funciones no llamadas, imports no usados, clases sin referencia.
3. **No crear abstracciones prematuras**: no crear wrappers, factories o helpers "por si acaso".
4. **No duplicar configuración**: parámetros de modelo definidos UNA vez (en `daily_pipeline.py` para V3).
5. **No crear archivos YAML que no se cargan**: si los params están hardcoded, no crear un YAML decorativo.
6. **Tests deben importar módulos que existen**: al eliminar un módulo, eliminar sus tests.
7. **Scripts de investigación** van en `scripts/archived/` cuando no se usan activamente.
8. **Docstrings solo donde aportan**: no añadir docstrings a funciones con nombre autoexplicativo.
9. **Cada función debe tener al menos un caller**: si nadie la llama, eliminarla.
10. **Feature selection estable**: usar multi-seed para evitar features inestables.

### 14.3 Estructura de Nuevas Features

Al implementar una feature nueva en el pipeline:

1. Validar con investigación (script en `scripts/`) antes de integrar
2. Si el resultado es positivo, integrar en el archivo correspondiente de `src/app/`
3. Archivar el script de investigación en `scripts/archived/`
4. Actualizar `V2_FEATURES` en `daily_pipeline.py` si se añade un feature
5. Actualizar esta documentación
6. Eliminar código que el nuevo feature reemplace

### 14.4 Checklist Pre-Commit

- [ ] ¿Hay imports no usados?
- [ ] ¿Hay funciones definidas pero no llamadas?
- [ ] ¿Hay módulos que ya no importa nadie?
- [ ] ¿Se actualizó PROJECT.md si cambió la estructura?
- [ ] ¿Los tests pasan? (`PYTHONPATH=src pytest tests/unit -v`)

---

## 17. Historial de Versiones

| Versión | Cambio | Resultado |
|---|---|---|
| v1.0 | Pipeline base: LightGBM + features básicos | Baseline AUC ~0.54 |
| v1.x | Multi-model ensemble, SHAP selection, triple-barrier, ATR stops | Val +17.65% |
| v2.0 | **Modelo sector-relative + Walk-Forward CV** | |
| | - Target: `fwd_ret_20d_sector_rel` (probado mejor que binary) | WF-CV estable |
| | - LGB Regressor (probado más estable que LambdaRank) | 6/7 Sharpe>0 |
| v2.1 | **Optimización TOP_K=8, hold=20d** | Sharpe 0.75, DD -8.7% |
| | - Sweep TOP_K∈{2..10} × hold∈{15,20,30,44} | vs -0.39 anterior |
| | - Diversificación (8 pos) reduce drawdown de -23% a -8.7% | |
| | - 26 features (selección estable multi-seed) | IC=+0.063 |
| | - Eliminados mkt_* features (ruido puro) | AUC mkt=0.500 |
| | - Eliminado código muerto (~1,500 líneas) | Limpieza |
| | - Archivados scripts de diagnóstico | Organización |
| v2.2 | **Free Data Sources Layer** | |
| | - 5 conectores: Yahoo, SEC EDGAR, FINRA, FRED, Nasdaq Trader | `src/app/data/free_sources/` |
| | - Yahoo backfill: 521K rows, 318 tickers, 2019→2026 | +360K rows pre-Massive |
| | - Quality reconciliation: median diff 0.00%, mean 2.31% | Adjusted vs unadjusted |
| | - Value experiment: Hybrid IC=0.076 vs Massive IC=0.122 | Top-8 ret +40% |
| | - Reports: `free_data_sources_plan.md`, `quality.json` | Documentación |
| v2.3 | **Meta-Learner: Error-Aware Features** | |
| | - 5 meta features: error_ticker, error_sector, hit_rate, IC, error_vol | `src/app/features/meta_features.py` |
| | - Signal history via WF backtest: 1,672 señales, 1,511 con outcomes | `signal_history_backtest.parquet` |
| | - Validación: top-8 return +10.75pp (16.57% → 27.32%), RMSE -2.3% | IC baja -25% (esperado) |
| | - Integrado en producción: 31 features (26 base + 5 meta) | Auto-correctivo |
| **v3.0** | **V3 LambdaRank + Sector Enrichment** | |
| | - Objetivo LambdaRank (16 bins, truncation_level=8) reemplaza regressor | Sharpe 2.94 → 3.71 |
| | - Sector enrichment via yfinance (Unknown 54% → 43.6%) | Mejores sector features |
| | - HP tuning: leaves=31, depth=6, mc=30, lr=0.05, 600 trees | SelMed cruzó a +0.47% |
| | - 33 features (26 base + 2 EDGAR + 5 meta) | |
| | - Walk-forward: 16 folds, 1,600 trades, Sharpe 4.62, WR 54% | 16/16 folds positivos |
| v3.1 | **Adaptive Stop + Dual Paper Trading** | |
| | - Adaptive stop: tighten trail to 6% after day 5 if profitable | WR +10.3pp (52.75% → 63.06%) |
| | - Dual portfolio paper trading (Baseline + Adaptive) | `data/paper_trading/adaptive/` |
| | - `scai` CLI (run / web / monitor) | `src/app/cli/main.py` |
| | - Dashboard con tabs por estrategia | Dark theme, dual canvases |
| | - Intraday monitor via Polygon snapshots | `scripts/intraday_monitor.py` |
| | - Paper trading activo desde 2026-05-19 | €1,000 por estrategia |
| v3.2 | **Tradability gate (hotfix producción)** | |
| | - Root cause de pérdidas live: señales sin filtro compraban zombies sub-penny (SRNE @ $0.0006) | WR live 18-36% → diagnóstico |
| | - `tradable_mask`: close ≥ $1.50, ADV20 ≥ $500K, solo en selección | `src/app/features/tradability.py` |
| | - Harness con filtro + IC tradable + costes + caché de predicciones | `scripts/v3/_v3_harness.py` |
| **v4.0** | **Honest re-baseline + exit engineering (2026-06-11)** | |
| | - Re-baseline sin meta leak, con filtro y 15bps/lado: +12.7%/fold (~4.1%/mes), Sharpe 2.52, WR 51.3%, 14/16 folds | `v4_filt_baseline.json` |
| | - Exit sweep (19 políticas sobre caché): pt40 gana (Sharpe 2.79); adaptive6_pt40 para portfolio adaptive (WR 59.7%, maxDD −15.1%) | `scripts/v3/21_exit_sweep.py` |
| | - Feature batches E/B/A y variantes de modelo (bins8/32, blend) TODAS rechazadas — IC amplio no transfiere a top-8 | `v4_feat_*.json`, `v4_model_*.json` |
| | - Profit target +40% portado al engine live (ambas estrategias) | `src/app/paper_trading.py` |
| | - Portfolios reseteados (era V4); informe final con CIs | `reports/v4_final_report.md` |
