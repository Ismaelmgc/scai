# SCAI — documentación técnica

> Plataforma de investigación y *paper trading* algorítmico, automatizada de punta a
> punta. Este documento explica qué hay en producción, cómo funciona, cómo se evalúa
> y qué se ha descartado (y por qué). Operación y despliegue: `docs/OPERATIONS.md`.

---

## 1. Qué es

SCAI empezó (mayo 2026) como un ranking ML de small caps de EE. UU. Tras corregir una
fuga de información en la evaluación y medir costes reales, el small cap resultó no
tener ventaja tras costes, y el proyecto evolucionó hacia:

- **liquidcap** — ranking ML del S&P 500 (el producto principal).
- **btc_trend** — regla de tendencia en BTC con rotación a futuros gestionados.
- Dos libros "en sombra" para validar fuera de muestra efectos menores (*bounce*, *illiquid*).

Todo corre en la nube a diario con datos gratuitos, cada libro con €1.000 ficticios.

---

## 2. Libros en vivo (paper trading)

| Libro | Tipo | Desde | Universo | Idea |
|---|---|---|---|---|
| `liquidcap` | ML | 2026-07-06 | S&P 500 (miembros point-in-time) | LambdaRank sobre 25 features → 4 tandas de top-8 (máx. 2 por sector), 20 días |
| `btc_trend` | Reglas | 2026-08-19 | BTC + ETF de futuros gestionados | Comprado si cierre > SMA100·1,03; fuera si < SMA100·0,97; fuera → CTA |
| `bounce` | Reglas (sombra) | 2026-08-28 | S&P 500 | Compra caídas: RSI<35, A/D-norm(150)<20, ROC(20) ≤ −7% |
| `illiquid` | ML (sombra) | 2026-07-24 | Small caps | Prima de iliquidez: tercil de menor ADV, sin el 20% de mayor spread, ordenado por el modelo small-cap |

Retirados: `baseline` (ML small cap, 2026-06-11 → 2026-10, cerró en −16,7%) y
`adaptive` (variante con stop ajustado, retirada 2026-08-27).

Foto a 2026-10-26 (valor del libro desde €1.000): liquidcap 898,8 · btc_trend 1.154,6 ·
bounce 988,2 · illiquid 1.070,6 (bruto, sin impacto de mercado). Con 2-4 meses de
historia, ninguna cifra es todavía estadísticamente informativa.

---

## 3. Arquitectura

```
GitHub Actions (disparado por un Cloudflare Worker con cron fiable)
  ├─ btc.yml        00:10 UTC  vela diaria BTC → regla → libro BTC↔CTA
  ├─ liquidcap.yml  03:00 UTC  yfinance + SEC EDGAR → features → (reentreno semanal) → libro
  ├─ bounce.yml     04:00 UTC  yfinance → señales → libro sombra
  ├─ daily.yml      06:00 UTC  Polygon → features → (reentreno semanal) → señales → libro illiquid
  └─ monitor.yml    lunes      IC en vivo del modelo small-cap (alarma fuera de muestra)
        │
        ▼
Supabase (Postgres + Auth + RLS): portfolio_state, nav_history, signals, trades,
dashboard_view (vista ya renderizada por libro), live_prices
        │
        ▼
GitHub Pages: shell de login estático → Supabase Auth → lee dashboard_view.
Precios en vivo: Cloudflare Worker (Finnhub, horario EE. UU.) → live_prices;
cripto directamente desde la API pública de Binance en el navegador (24/7).
Telegram: resumen por ejecución y alerta en fallo.
```

Decisiones de diseño:

- **Supabase como fuente de verdad del estado** (no git): cada job lee el estado,
  procesa solo sesiones nuevas (idempotente por `last_update`) y lo reescribe.
- **La página no contiene datos**: solo login; las vistas se sirven tras autenticar (RLS).
- **Scheduler propio**: el cron de GitHub llega tarde o se pierde; un Worker de
  Cloudflare dispara `workflow_dispatch` a la hora.
- **Fallos ruidosos**: un precio no finito nunca entra en un libro (un `null` en
  `portfolio_state` bloquea el job); un error de datos pone el run en rojo + Telegram.

---

## 4. Estructura del repositorio

```
src/app/
├── config/              Settings (pydantic-settings, prefijo SCAI_)
├── data/
│   ├── massive/         Cliente Polygon: reference, aggregates, snapshots, corporate actions, jobs/
│   ├── free_sources/    yahoo.py (precios), sec_edgar.py (fundamentales point-in-time)
│   ├── store/           ParquetStore (DuckDB)
│   ├── smallcap_ohlcv.py     descarga incremental small cap (barras agrupadas)
│   ├── smallcap_universe.py  descubrimiento point-in-time del universo small cap
│   └── supabase_store.py     estado / NAV / señales / vistas / precios en vivo
├── features/            pipeline.py (build_feature_matrix + labels), price_action, momentum,
│                        volatility, liquidity, microstructure, market_regime, sector,
│                        cross_sectional, fundamentals, alpha_features, lags, tradability
├── utils/               logging, semillas, Telegram, point_in_time (as_of, lag_safe_merge)
├── engine/cohort_book.py  motor ÚNICO de liquidcap: backtest y libro en vivo usan el mismo step()
├── books/publish.py     publicación común de los 4 libros (estado, señales, NAV, vista, precios)
├── smallcap/            pipeline del modelo small cap: data, features, model, signals
├── paper_trading.py     motor PaperTrader (libro illiquid)
├── signal_tracker.py    histórico de señales + resultado a 20 días
└── web/                 dashboard_data (vistas), server (FastAPI local), templates/
scripts/
├── daily_pipeline.py    orquesta el modelo small cap (app.smallcap)
├── illiquid_shadow.py   libro sombra illiquid
├── liquidcap/           universo S&P 500 (+ gics_sectors.json), fundamentales, features, job
│                        diario, backtest_cohorts.py (backtest con el motor en vivo), bootstrap
├── btc_trend_daily.py   libro BTC ↔ CTA
├── bounce_daily.py      libro sombra bounce
├── monitor_live_ic.py   monitor de IC en vivo
├── refresh_universe.py  refresco mensual del universo small cap
├── render_static_dashboard.py  genera la web estática (Pages)
├── supabase/schema.sql  tablas + RLS
├── v3/                  _v3_harness.py (walk-forward purgado), 18_verify_no_leak.py, 34_mc_validate.py
└── archived/            investigación terminada (no se mantiene; ver su README)
cloudflare/              scheduler/ (cron) y live-prices/ (Finnhub → Supabase)
data/                    paneles OHLCV, modelos, hechos EDGAR, resultados de benchmark (v3_benchmarks/)
tests/unit/              tests que ejecuta el CI
```

---

## 5. Datos

| Fuente | Uso | Notas |
|---|---|---|
| Polygon.io ("massive") | barras diarias small cap, catálogo de tickers con deslistados | plan gratuito: 5 llamadas/min; ajuste por splits |
| yfinance | precios S&P 500 (liquidcap, bounce), QQQ, ETF CTA | ajustado por dividendos (`auto_adjust`) |
| Tiingo | bootstrap histórico S&P 500 con deslistados | solo arranque |
| SEC EDGAR | fundamentales XBRL; CIK de empresas deslistadas | point-in-time por **fecha de presentación** |
| Binance (mirror público) / Coinbase | velas diarias y horarias de BTC | el API principal de Binance bloquea IPs de EE. UU. (runners de GitHub) |
| Finnhub | precios en vivo (worker) | gratuito |

**Universos sin sesgo de supervivencia**: el small cap incluye tickers deslistados
del catálogo de Polygon (y excluye SPACs/fondos por SIC); liquidcap usa la
composición histórica del S&P 500 (`membership_sp500.parquet`) con los miembros que
salieron (`ohlcv_delisted.parquet`). Los deslistados se mantienen en el
**entrenamiento**; solo la **selección** se restringe a lo negociable.

**Puerta de negociabilidad** (`features/tradability.py`): solo se seleccionan tickers
con cierre ≥ $1,50 y volumen medio 20d ≥ $500K. (En vivo se llegaron a comprar
"zombies" a $0,0006 antes de existir.)

---

## 6. Modelos ML

### 6.1 Formulación común

- **Learning-to-rank**, no regresión: LightGBM `lambdarank`, una "query" por fecha,
  los activos de esa fecha son los documentos.
- **Target**: `fwd_ret_20d_sector_rel` = retorno a 20 días **menos la media de su
  sector ese día**, discretizado por fecha en 16 niveles de relevancia (qcut).
  Neutraliza el mercado: el modelo aprende selección, no *timing*. **Ojo en
  liquidcap**: sus features se construyen con todos los tickers en el sector
  "Unknown", así que allí el target es en realidad relativo a **todo el mercado** y
  `sector_ret_60d` es el retorno del mercado (igual para todos en una fecha). El A/B
  con sectores GICS reales (agosto, re-evaluado con el motor de tandas en octubre)
  dio lo mismo (Sharpe 0,89 vs 0,88) → spec congelada sin cambios.
- **Entrenamiento**: semanal, ventana expansiva sobre todo el histórico (deslistados
  incluidos). Purgado por construcción: las filas cuyo retorno a 20 días aún no se
  conoce tienen target NaN y se descartan.
- `n_jobs=1` siempre (conflicto de libomp en macOS con multi-hilo).

### 6.2 liquidcap (S&P 500) — spec `GI_fs15_fund_20d`

- **25 features**: 15 de precio/volumen (`ret_252d`, `sector_ret_60d`, `rev_21d`,
  `realized_vol_120d`, `amihud_60d`, `beta_60d`, `downside_vol_60d`, `sma_200`,
  `ema_26`, `max_dd_60d`, `vol_of_vol_ratio`, `adv_60d`, `intraday_avg_20d`,
  `price_roc_smooth_120d`, `obv`) + 10 ratios fundamentales point-in-time
  (`earnings_yield`, `cfo_yield`, `sales_yield`, `book_to_market`, `debt_to_equity`,
  `roa`, `op_margin`, `accruals`, `asset_growth`, `earnings_chg`).
- 300 árboles, 16 bins.
- **Construcción de cartera (motor `app/engine/cohort_book.py`)**: el capital se divide
  en 4 tandas; cada 5 sesiones una tanda vende lo que le quede y compra el top-8 del
  día (peso igual, **máx. 2 por sector GICS**) → ~12-13 valores distintos a la vez.
  Salidas: objetivo +40%, *trailing stop* ATR·5,3 acotado a [10%, 16%] desde el
  máximo de cierre, vencimiento a 20 sesiones; el dinero de una salida **espera en su
  tanda** hasta su próxima rotación (no se recompra el siguiente del ranking).
- **Un solo motor para backtest y vivo**: `scripts/liquidcap/backtest_cohorts.py`
  aplica a las predicciones fuera de muestra la misma función `step()` que ejecuta el
  job diario; un test de paridad garantiza que avanzar día a día con el estado
  guardado/recargado en JSON da exactamente el mismo resultado que el backtest.
  Hasta 2026-10 el libro en vivo usaba otra construcción (8 huecos que se
  rellenaban al día siguiente de cada salida): en la misma reproducción, Sharpe 0,65
  y caída máx. −48% frente a 0,89 y −27% → la caída de octubre (−18% en 12 sesiones
  durante la corrección de semiconductores) era normal para lo que se operaba, no
  para lo validado. Migración sin ventas: las 8 posiciones pasaron a "herencia",
  sus salidas alimentan una reserva y cada tanda se rellena al rotar (convergencia en
  un ciclo de 20 sesiones).
- **Recupera sesiones perdidas solo**: el job puntúa y procesa en orden todas las
  sesiones posteriores a `last_update`.
- **Entrada al cierre** de la misma sesión de la señal (simula una orden MOC);
  antes entraba a la apertura siguiente y perdía ~0,16 pp por operación.
- **Backtest purgado**, 34 folds 2018-2026, universo point-in-time con deslistados:
  **+1,69%/mes con coste plano de 5 pb; +0,95%/mes con coste según spread**; con el
  motor de tandas + límite de sector: ~22%/año, Sharpe 0,89, caída máx. −27%.
  Ojo: el "Sharpe 2,43" que aparece en los resultados es la **media de Sharpes por
  fold de 3 meses**; reconstruido mes a mes como cartera, el Sharpe es ≈0,7 (≈ SPY).
  La ventaja es exceso de rentabilidad, no menos riesgo.

### 6.3 Modelo small cap

- 28 features: 26 de precio/volumen/microestructura (`max_dd_60d`, `vol_of_vol_60d`,
  `ret_kurtosis_60d`, `avg_trade_size_20d`, `obv`, `amihud_60d`, `spread_proxy_20d`,
  `sma_200`, `sector_ret_60d`, `macd_hist`, `beta_60d`, …) + 2 de EDGAR
  (`dilution_pct`, `current_ratio`). 600 árboles.
- Ya no opera libro propio (sin ventaja tras spreads realistas); alimenta al libro
  sombra *illiquid* y el monitor de IC en vivo lo vigila fuera de muestra.

---

## 7. Estrategias por reglas

### 7.1 btc_trend (BTC ↔ futuros gestionados)

- Señal sobre el cierre diario (00:00 UTC): entra si cierre > SMA100·1,03, sale si
  cierre < SMA100·0,97 (la banda evita entrar y salir en falso: ~2,4 ciclos/año).
- Dentro: 100% BTC. Fuera: 100% en el iMGP DBi Managed Futures UCITS ETF (gemelo
  europeo del DBMF, comprable por un minorista de la UE).
- Ejecución realista: la salida vende BTC al cierre de la vela y compra el ETF en la
  siguiente apertura de París; la entrada vende el ETF en la apertura y compra BTC a
  esa hora. Costes: 12 pb BTC, 10 pb ETF por lado.
- **Backtest**: 2019-2026 (DBMF real) ~59-63%/año, Sharpe ~1,2, máx. caída −21%
  (mensual) / −31% (diaria). 2015-2026 con AQR Managed Futures como sustituto: ~69%/año
  pero **máx. caída −71%** (2018: −57%; la pérdida ocurre *dentro* de BTC tras el techo
  parabólico de 2017, la media de 100 días va muy por detrás). La ventaja viene de
  evitar desplomes (2018, 2022), no de ganar más en los años alcistas.

### 7.2 bounce (S&P 500, sombra)

RSI(14) < 35, A/D normalizado (150) < 20 y ROC(20) ≤ −7%; prioriza la mayor caída;
máx. 8 posiciones; stop −12% hasta que +12% arma un *trailing* = max(entrada·1,10,
pico·0,98); límite de 60 sesiones. Elegido en muestra → solo cuenta su resultado en vivo.

---

## 8. Ejecución y paper trading

- **PaperTrader** (`src/app/paper_trading.py`): llena órdenes, gestiona *trailing
  stops*, objetivo de beneficio, vencimiento por días, *splits* (reescala posiciones
  abiertas cuando el histórico se ajusta) y **nunca** acepta precios no finitos.
- Acciones fraccionadas a 4 decimales (como IBKR).
- **Replicabilidad real**: todo debe poder ejecutarse en IBKR como minorista de la UE.
  La normativa PRIIPs bloquea los ETFs domiciliados en EE. UU. (SPY, IBIT, DBMF…),
  así que se usan acciones individuales, cripto nativa de IBKR y ETFs UCITS. Con
  €1.000 los mínimos por orden de IBKR (≈$0,35) se comerían liquidcap; necesita ~€10k+.

---

## 9. Metodología de evaluación

### 9.1 Protocolo anti-fuga (obligatorio para cambiar la spec de un modelo)

En mayo de 2026 unas *meta-features* filtraban resultados futuros y en julio se
descubrió que el harness no purgaba el solape de 20 días entre entrenamiento y test:
juntas inflaban el backtest (+4%/mes, Sharpe ~3 → honesto ~+1%/mes). Desde entonces:

1. **Features = información disponible a las 00:00 de T.** Lo que dependa de `(T, T+H]` es un label.
2. Rolling temporal **por ticker**; cross-sectional **misma fecha**. Nunca mezclar.
3. Datos externos con `merge_asof(direction="backward")` sobre la **fecha de publicación**.
4. Prefijos/sufijos de label prohibidos en features (`fwd_*`, `future_*`, `actual_ret*`, `tb_label*`, `*_positive`).
5. **Walk-forward purgado**: se descartan del entrenamiento las filas cuyo horizonte
   de 20 días solapa con el periodo de test.
6. **Puerta anti-fuga automática** (`app/features/leak_checks.py`): **cada
   reentrenamiento de producción** (small cap y liquidcap) la ejecuta antes de guardar
   el modelo; si falla, el run se pone en rojo y se conserva el modelo anterior.
   Comprueba nombres de label, columnas degeneradas, |Pearson| < 0,10 y |IC medio por
   fecha| < 0,25. Se usa el IC **con signo**: una exposición como beta tiene mucha
   correlación diaria pero cambia de signo con el mercado; una fuga es constante.
   Calibrada con datos reales (features legítimas ≤ 0,12; una copia ruidosa del target
   0,74). A mano: `make leak-check` / `scripts/v3/18_verify_no_leak.py --book liquidcap`.

### 9.2 Estadística de dos niveles

- **Cribado** de muchas ideas: nº efectivo de pruebas independientes (correlación
  entre estrategias) + control de falsos descubrimientos Benjamini-Hochberg.
- **Despliegue**: bootstrap emparejado por fold, Sharpe deflactado, longitud mínima
  de historial (MinTRL), batir al índice de referencia con costes realistas.
- Costes: plano (5-15 pb) **y** según spread estimado (Corwin-Schultz / Abdi-Ranaldo).
- Validación fuera de muestra en el tiempo (diseñar en un periodo, juzgar en otro) y
  en otros activos (una regla real de BTC debe ayudar también en otras criptos).

---

## 10. Registro de investigación (qué funcionó y qué no)

**Adoptado**
- LambdaRank + target relativo al sector (frente a regresión MSE).
- Puerta de negociabilidad (precio/liquidez) — fue la causa raíz de los "zombies".
- Entrada al cierre (MOC) en liquidcap: +0,16 pp/operación.
- Fundamentales point-in-time de EDGAR en liquidcap.
- BTC: regla de tendencia SMA100 ±3% y rotación a futuros gestionados cuando está fuera.

**Rechazado (no repetir sin un ángulo nuevo)**
- Meta-features / tamaño por convicción / K>8: artefactos de la fuga.
- Small cap en general: bate al IWM solo con coste plano; con spreads reales, no.
  La prima de iliquidez es el único efecto que sobrevive y no es significativo (→ libro sombra).
- Indicadores técnicos clásicos (incl. réplica de RevHunter), money-flow, factores de
  calidad: suben el IC pero no el rendimiento operado ("IC marginal ≠ rendimiento").
- Resultados trimestrales (sizing y como feature desde 8-K de la SEC): IC ↑, Sharpe 2,42 → 1,80.
- Pares / arbitraje estadístico en el S&P 500: bruto ≈ 0. LSTM: IC ≈ 0.
- Atención/aprobación social (menciones en WallStreetBets, visitas a Wikipedia): tras
  los picos de menciones las acciones rinden **peor** (−3,2% a 20 días).
- Forex con tendencia (negativo 2011-2026); *timing* del oro (< comprar y mantener);
  carry de *funding* en cripto (≈3%/año ahora, < letras del Tesoro); *altcoins* con
  tendencia (diluyen); cortos en BTC (peor que la rotación a CTA); temporalidades
  intradía (los costes se comen la señal); barridos de medias móviles (ruido: el
  mejor en 2015-20 rinde peor después que la regla en vivo).
- Estrategias publicadas de "5%/mes" (ruptura de Larry Williams en ETH, máximos/mínimos
  de 10 días): funcionaron hasta 2021 y dejaron de funcionar fuera de muestra.
- Corto en nuevas cotizaciones de Binance: efecto real (−22% mediana a 30 días) pero
  las subidas explosivas lo limitan a ~1-1,8%/mes con caídas de −83%.

Los resultados por configuración del harness están en `data/v3_benchmarks/`.

---

## 11. Limitaciones conocidas

1. **Historial en vivo corto** (2-4 meses por libro): el paper trading es el árbitro y aún no ha hablado.
2. **Regímenes**: el histórico small cap empieza en 2021; BTC tiene ~3 ciclos completos.
3. **Selección de la regla de BTC** tras probar muchas variantes → esperar menos que el backtest.
4. **Sesiones perdidas**: liquidcap las recupera solo; *illiquid* (PaperTrader) no.
5. **Divisa**: los backtests están en USD; un inversor en EUR añade riesgo EUR/USD.
6. **Impuestos** (19-28% en España) no incluidos en ningún backtest.

---

## 12. Reglas de desarrollo

1. **Anti-fuga** (sección 9.1) en cualquier feature o cambio de spec.
2. **`n_jobs=1`** en todos los modelos ML.
3. **Sin código muerto**: cada función con al menos un caller; al borrar un módulo, borrar sus tests.
4. **Sin abstracciones ni configuración decorativa**: los parámetros de cada libro
   viven congelados en su script, junto a la regla.
5. Investigación terminada → `scripts/archived/`.
6. Flujo git: rama → PR a `main` → CI en verde → merge. Los *hotfix* de producción de
   los jobs pueden ir directos a `main`. El bot de datos hace push directo de datos.
7. Antes de subir: `make lint test`.
8. **Dependencias fijadas** en `requirements.lock` (universal Windows/Linux, generado
   con `uv`): el CI instala exactamente eso. Tras cambiar `pyproject.toml`: `make lock`.
9. **Una estrategia, una implementación**: si backtest y vivo necesitan la misma
   lógica, va en un único módulo de `src/app/` y un test de paridad lo protege.

---

## 13. Historial de versiones

| Fecha | Hito |
|---|---|
| 2026-05 | V1-V2: pipeline small cap, LightGBM, 26+ features, walk-forward |
| 2026-05-21 | V3: LambdaRank + target relativo al sector; doble paper trading |
| 2026-05-22 | Se detecta fuga en meta-features → retiradas; protocolo anti-fuga |
| 2026-06-11 | V4: puerta de negociabilidad, salidas (+40%), reset de libros |
| 2026-07-03 | Se descubre la falta de purga (20 días) → métricas honestas ~+1%/mes |
| 2026-07-06 | Nace **liquidcap** (S&P 500 + fundamentales EDGAR) |
| 2026-07/08 | Libros sombra *illiquid* y *bounce*; scheduler en Cloudflare; Supabase como verdad |
| 2026-08-27 | liquidcap pasa a entrar al cierre (MOC) |
| 2026-09/10 | Investigación multiactivo y cripto; nace **btc_trend** (con rotación a CTA); se retira el *baseline* |
| 2026-10-27 | Limpieza del repo: pipeline V2 archivado, código muerto y dependencias eliminados |
| 2026-10-28 | Motor único de tandas para liquidcap (backtest = vivo) + límite de sector; dependencias bloqueadas; puerta anti-fuga automática; publicación común; paquete `app.smallcap` |
