# SCAI — Claude Code Instructions

> Contexto base para cada sesión. **Documentación completa: `PROJECT.md`** (productos,
> modelos, metodología, registro de investigación). **Operación y runbook:
> `docs/OPERATIONS.md`.** Este archivo solo recoge lo no obvio y operativo.

## Qué es

Plataforma de investigación + *paper trading* algorítmico, automatizada en GitHub
Actions con estado en Supabase. Libros en vivo (€1.000 c/u):

- **liquidcap** — S&P 500, LGB LambdaRank (25 features: 15 precio/volumen + 10
  fundamentales EDGAR point-in-time), top-8, 20d, entrada al cierre (MOC-sim). Producto principal.
- **btc_trend** — BTC comprado si cierre > SMA100·1,03, fuera si < SMA100·0,97; fuera →
  ETF UCITS de futuros gestionados (`DBMF.PA`). Libro re-sembrado desde la señal real (2026-08-19).
- **bounce** (sombra) — compra de caídas en el S&P 500. **illiquid** (sombra) — prima de
  iliquidez small cap, puntuada con el modelo small cap.
- Retirados: **baseline** small cap (sin ventaja tras spreads) y **adaptive**.

## Plataforma (IMPORTANTE)

Desarrollado en macOS arm64; esta copia corre en **Windows** (`.venv/`, Python 3.11).
- macOS: scripts ML con `DYLD_LIBRARY_PATH=.local/lib` (sin esto → segfault por libomp).
  En Windows es irrelevante.
- Siempre `PYTHONPATH=src` (`from app...`). El `Makefile` y el CLI `scai` ya lo ponen.
- Consola Windows: `PYTHONIOENCODING=utf-8` (los scripts imprimen ✓ ▸ →).

## Métricas honestas (no citar las antiguas)

- **liquidcap**: backtest purgado 34 folds 2018-26 = +1,69%/mes coste plano 5 pb,
  +0,95%/mes con spread. "Sharpe 2,43" = media de Sharpes por fold de 3 meses; el
  Sharpe mensual de cartera es ≈0,7 (≈ SPY).
- **btc_trend**: 2015-26 ~69%/año pero **máx. caída ~−71%** (2018); 2019-26 ~59-63%/año,
  máx. caída −21% mensual / −31% diaria. Las tablas que empiezan en 2019 infravaloran el riesgo.
- **small cap**: +1-1,4%/mes con coste plano, ~0% con spread. Las cifras pre-julio
  (+4%/mes, Sharpe ~3-4,6) estaban infladas por fuga (meta-features + sin purga de 20d).
- El paper trading en vivo es el árbitro; aún hay muy poco historial.

## Comandos

```bash
make test lint          # lo que corre el CI (pytest tests/unit + ruff src/ tests/)
make btc-dry            # cualquier libro sin escribir: btc-dry / liquidcap-dry / bounce-dry / smallcap-dry
make web                # dashboard local → http://localhost:8501
make leak-check         # verificador anti-fuga (puerta MANUAL al cambiar features/spec)
```

`make smallcap-dry` / `scai run` necesitan `data/processed/ohlcv_smallcap.parquet` (en git).
El panel de liquidcap está en `.gitignore` y lo reconstruye su job.

## Archivos clave

- `scripts/liquidcap/daily_liquidcap.py` — ★ libro S&P 500 (reentreno semanal purgado
  por construcción + refresco EDGAR + panel **ajustado por dividendos** vía yfinance;
  NO usar Polygon aquí: solo ajusta splits y corrompe features en ex-dividendo).
- `scripts/btc_trend_daily.py` — libro BTC ↔ CTA (órdenes pendientes hasta la siguiente
  apertura de París; nunca rellena con precios no finitos).
- `scripts/daily_pipeline.py` — modelo small cap: datos Polygon, features, reentreno c/7d,
  señales (registradas en Supabase `signals` con clave histórica `"baseline"`). Ya no opera libro.
- `scripts/illiquid_shadow.py`, `scripts/bounce_daily.py` — libros sombra.
- `scripts/monitor_live_ic.py` — única alarma fuera de muestra del modelo small cap
  (IC en vivo por fecha; DEGRADADO → exit 2 → run en rojo → email).
- `src/app/paper_trading.py` — motor PaperTrader (`_finite_price`, `_reconcile_splits`).
- `src/app/web/dashboard_data.py` — vistas del dashboard (`build_view(ohlcv, pt_dir, strategy, bench=, bench2=)`).
- `scripts/v3/_v3_harness.py` — walk-forward purgado (fuente de las cifras de backtest).

## Reglas de desarrollo

1. **Anti-fuga (OBLIGATORIO)**: features = info disponible a las 00:00 de T; `as_of()` /
   `lag_safe_merge()`; datos externos por fecha de publicación. Cambiar features o spec
   exige pasar el harness purgado **y** `scripts/v3/18_verify_no_leak.py` (no se ejecuta
   solo en los reentrenos semanales: hay que lanzarlo).
2. **n_jobs=1** en todos los modelos ML.
3. Sin código muerto, sin abstracciones prematuras, sin configuración decorativa. Al
   borrar un módulo, borrar sus tests.
4. Investigación terminada → `scripts/archived/` (no se mantiene).
5. **Replicable en IBKR como minorista UE**: nada de ETFs domiciliados en EE. UU. como
   instrumento ejecutable (PRIIPs); usar acciones, cripto nativa de IBKR o UCITS.
6. Git: rama → PR → CI verde → merge. Los *hotfix* de producción de los jobs pueden ir
   directos a `main`. Antes de subir: `make lint test`.
7. Dependencias pesadas con techo de versión en `pyproject.toml` (el CI instala desde
   cero en cada ejecución; pyarrow 26 rompió todos los jobs el 2026-10-09).

## Fuentes de datos

- Small cap: Polygon (`massive`), plan **gratuito** → `MASSIVE_CALLS_PER_MINUTE=5` (lento, no falla).
- S&P 500: yfinance (precios) + SEC EDGAR (fundamentales); Tiingo solo para el bootstrap.
- BTC: mirror público de Binance (`data-api.binance.vision`; el API principal bloquea los
  runners de EE. UU.) con Coinbase de respaldo.
- Precios en vivo: Cloudflare Worker (Finnhub REST, horario EE. UU.) → `live_prices`;
  cripto 24/7 directamente desde Binance en el navegador.
