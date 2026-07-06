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

## Modelo en producción (V4)

- LGB **LambdaRank**, target `fwd_ret_20d_sector_rel`, binned a 16 niveles, 600 trees. Modelo idéntico a V3 — V4 cambió la capa de ejecución, no el modelo (todas las variantes de features/modelo fueron rechazadas por el harness, ver `data/v3_benchmarks/v4_*.json`).
- **28 features** = 26 base + 2 EDGAR (`dilution_pct`, `current_ratio`). **0 meta** (retiradas 2026-05-22 por data leak).
- TOP_K=8 **conviction-weighted** (sizing ∝ max(z,0.1) del z-score cross-sectional del pred sobre el universo tradable, normalizado a 1 — el #1 del ranking pesa más que el #8; `conviction_sizes()` en `daily_pipeline.py`. Validado +1.6%/fold vs equal-weight, Sharpe 1.60→1.74, DD igual, bootstrap CI[+0.3%,+2.9%]>0; ver `scripts/v3/43_capital_deployment.py`. Reemplazó el equal-weight 1/8 el 2026-06-23). **Acciones fraccionadas a 4dp = precisión IBKR** (sin cash ocioso por redondeo, calza con el backtest, ejecutable vía orden por importe), holding 20d, rebalance 5d, trailing stop ATR clipped [10%, 16%], cooldown 5d tras salida, **profit target +40%** (v4.0).
- **Tradability gate (v3.2, 2026-06-11)**: señales solo sobre tickers con `close ≥ $1.50` y `ADV20 ≥ $500K` (`src/app/features/tradability.py`). El filtro aplica SOLO a la selección — el entrenamiento mantiene deslistados (anti-survivorship).
- Dual paper trading (reset 2026-06-11, €1,000 c/u): Baseline (trail + pt40) + Adaptive (además tighten a 6% tras día 5 si profitable; WR backtest 59.7%, maxDD −15.1%).
- **Métricas honestas TRAS PURGING (2026-07-03, leak de 20d corregido)**: +1.0-1.4%/mes a 15bps planos (CI del fold **cruza 0**), ~0%/mes spread-aware, WR 45-55%, Sharpe ~1.2. Las cifras antiguas (+4%/mes, Sharpe ~2.7) estaban infladas por look-ahead de 20d en el harness (sin purge) — NO citarlas. El edge está estadísticamente sin confirmar; el paper trading en vivo es el árbitro (y mide el coste real de fills).
- Artefacto: `data/models/smallcap_v3_lambdarank.pkl`.

## Segundo producto: LIQUIDCAP (S&P 500, 2026-07-06)

- **Book de paper trading en vivo** (Supabase estrategia `liquidcap`, €1.000): spec congelada `GI_fs15_fund_20d` — LGB LambdaRank 16 bins, **25 features** (15 precio/volumen top-gain pre-2018 + 10 ratios fundamentales SEC EDGAR point-in-time), top-8, hold 20d, salidas de producción. Backtest purgado (34 folds 2018-26, universo PIT con deslistados): **+1.69%/mes flat 5bps / +0.95%/mes spread-aware, Sharpe 2.43**. Resultados: `data/v3_benchmarks/liquidcap_screen_20260704.json`.
- Job diario (local, tras cierre US): `PYTHONPATH=src python scripts/liquidcap/daily_liquidcap.py`. Retrain semanal purgado-por-construcción + refresh EDGAR + re-descarga completa del panel (convención **dividend-adjusted** = yfinance auto_adjust; NO usar Polygon aquí — es split-only y corrompe features en ex-dividendos).
- Rechazados con gates (no repetir sin ángulo nuevo): LSTM (IC −0.002, réplica Fischer-Krauss), horizonte 10d/5d (coste de rotación), MLP/Ridge (no superan árboles), conviction sizing (artefacto del leak), K>8.
- Scripts activos en `scripts/liquidcap/`; research terminada en `scripts/archived/liquidcap_research/`.

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
- `scripts/morning_execute.py` — job intradía (`morning.yml`, tras la apertura): llena los BUY pendientes al `open` (Finnhub REST) y refresca `dashboard_view`. SOLO `execute_pending` (no exits/señales/día-idx) → estado idéntico al flujo de cierre; idempotente. Salidas se quedan a cierre (backtest 2026-06-15: intradía hunde Sharpe 2.79→1.15 por mechas de small-caps).
- `scripts/monitor_live_ic.py` — monitor de generalización en vivo (`monitor.yml`, semanal). Calcula la IC en vivo (Spearman score↔ret20d real, por fecha) + WR de los traded vs banda backtest; el retrain NO valida nada, esto es la única alarma out-of-sample. Veredicto DEGRADADO → exit 2 → run en rojo → email de GitHub. WARM-UP hasta que maduren ~20 sesiones. De paso backfillea `actual_ret_20d` en Supabase (que el pipeline nunca re-sube → arregla la columna del dashboard).
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
- `.env`: `SCAI_POLYGON_API_KEY` (plan de pago hasta **2026-06-19** → `MASSIVE_CALLS_PER_MINUTE=50`; tras esa fecha pasar a **free** → `=5`), `SCAI_SEED=42`, `SCAI_ENV`, `FINNHUB_TOKEN` (precios en vivo, free).

## Fuentes de datos (Polygon de pago → free)

- **Barras diarias** (entrenamiento/señales): Polygon (`massive`). El histórico 5 años ya está en el parquet de git (con deslistados → anti-survivorship). Al **cancelar el plan de pago**, poner `MASSIVE_CALLS_PER_MINUTE=5` (env del step en `daily.yml`): el job diario NO falla, solo tarda ~1 h (1 llamada/ticker, espaciadas 12s) y la barra end-of-day está disponible tras el cierre. Sin cambios de código (lo lee `daily_pipeline.update_ohlcv`).
- **Precio en vivo** (dashboard + monitor): **Finnhub free** (60 req/min, WebSocket real-time 50 símbolos). El dashboard hace streaming client-side vía `wss://ws.finnhub.io` (WebSocket no sufre CORS → token embebido en el HTML, como la anon key de Supabase). El monitor usa REST `/quote` server-side (`app.data.free_sources.finnhub`). Reemplaza los snapshots de Polygon (que en free no dan tiempo real). Requiere secret/`.env` `FINNHUB_TOKEN`.
