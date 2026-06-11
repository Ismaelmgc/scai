# SCAI — Claude Code Instructions

> Contexto base para cada sesión. **Documentación completa y detallada: `PROJECT.md`** (estructura, evolución de versiones, protocolo anti-leak, etc.). Este archivo solo recoge lo no obvio y operativo.

## Qué es

Plataforma de trading algorítmico ML para US small-cap ($50M–$2B).
Pipeline: universo → OHLCV → features → LGB LambdaRank (sector-relative, 16 bins) → señales BUY top-8 → dual paper trading.

## Plataforma (IMPORTANTE)

El repo se desarrolló en **macOS arm64**, pero esta copia corre en **Windows** (`.venv/`, Python 3.11).
- En **macOS**: ejecutar scripts ML con `DYLD_LIBRARY_PATH=.local/lib` (sin esto → segfault por libomp).
- En **Windows**: `DYLD_LIBRARY_PATH` es irrelevante (no hace nada). El CLI `scai` lo setea igual, es inofensivo.
- Siempre `PYTHONPATH=src` para los imports (`from app...`). El CLI `scai` ya lo gestiona.

## Modelo en producción (V3)

- LGB **LambdaRank**, target `fwd_ret_20d_sector_rel`, binned a 16 niveles, 600 trees.
- **28 features** = 26 base + 2 EDGAR (`dilution_pct`, `current_ratio`). **0 meta** (retiradas 2026-05-22 por data leak).
- TOP_K=8 equal-weight, holding 20d, rebalance 5d, trailing stop ATR clipped [10%, 16%], cooldown 5d tras salida.
- **Tradability gate (v3.2, 2026-06-11)**: señales solo sobre tickers con `close ≥ $1.50` y `ADV20 ≥ $500K` (`src/app/features/tradability.py`, validado por sweep en `scripts/v3/20_filter_sweep.py`). El filtro aplica SOLO a la selección — el entrenamiento mantiene deslistados (anti-survivorship). Baseline honesto post-filtro: `data/v3_benchmarks/v4_filt_baseline.json`.
- Dual paper trading: Baseline (trail [10-16%]) + Adaptive Stop (tighten a 6% tras día 5 si profitable). €1,000 c/u desde 2026-05-19.
- Artefacto: `data/models/smallcap_v3_lambdarank.pkl`.

## Comandos

```bash
scai run            # daily pipeline (descarga incremental + features + retrain c/7d + dual paper trading)
scai web            # dashboard FastAPI → http://localhost:8501
scai monitor        # check intradía trailing stops (Polygon snapshots)
PYTHONPATH=src pytest tests/unit -v --tb=short
```

`scai run` hace descarga **incremental**: requiere que `data/processed/ohlcv_smallcap.parquet` ya exista. El bootstrap inicial (descarga completa del histórico) no es parte del pipeline diario.

## Archivos clave

- `scripts/daily_pipeline.py` — ★ producción (V3 + dual paper trading). Reentrena cada 7 días **sobre todo el OHLCV almacenado** (incluye deslistados → evita survivorship bias).
- `scripts/run_smallcap_pipeline.py` — pipeline de análisis/backtest (~2000 líneas).
- `src/app/features/pipeline.py` — `build_feature_matrix()`.
- `src/app/data/store/parquet_store.py` — ParquetStore (read/write/upsert vía DuckDB).
- `src/app/data/massive/` — cliente Polygon.io. (Resto de la estructura → PROJECT.md.)

## Reglas de desarrollo

1. **Anti-leakage (OBLIGATORIO)**: features = info disponible a T 00:00. Usar `as_of()` / `lag_safe_merge()`. Todo entrenamiento de producción DEBE pasar `scripts/v3/18_verify_no_leak.py` (gate automático, ver PROJECT.md §10). No añadir nada a `V2_FEATURES`/`V2_EDGAR_FEATURES` sin validar antes con `scripts/v3/_v3_harness.py` + el verificador.
2. **n_jobs=1** en todos los modelos ML (conflicto libomp multi-thread).
3. No dejar código muerto (funciones sin caller, imports/módulos sin uso). No abstracciones prematuras. No YAML decorativos que nadie carga.
4. Scripts de investigación terminados → `scripts/archived/`.

## Datos y entorno

- Universo OHLCV ~1.000 tickers (activos + deslistados), ~830K filas, histórico ~2021→presente. Cifras exactas: consultar el parquet, no fiarse de números hardcodeados (envejecen).
- `.env`: `SCAI_POLYGON_API_KEY` (plan de pago, `MASSIVE_CALLS_PER_MINUTE=50`), `SCAI_SEED=42`, `SCAI_ENV`.
