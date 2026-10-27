# SCAI — algorithmic trading research platform

An end-to-end, fully automated quant research and paper-trading system: data
ingestion → point-in-time features → learning-to-rank models and rule-based
strategies → purged walk-forward evaluation → daily paper books running in the
cloud, tracked on a private web dashboard.

The emphasis is on **honest evaluation**: survivorship-free universes, strict
point-in-time data, purged cross-validation, realistic costs, multiple-testing
corrections — and a willingness to publish negative results. Several early
"great" backtests in this repo were later shown to be leakage artefacts and
corrected; the history is documented in [PROJECT.md](PROJECT.md).

## Live paper books

| Book | What it does | Type | Since |
|---|---|---|---|
| **liquidcap** | S&P 500 cross-sectional ranking: LightGBM LambdaRank on 15 price/volume + 10 point-in-time SEC fundamentals; top-8, 20-day hold, ATR trailing stop, market-on-close entry | ML | 2026-07 |
| **btc_trend** | BTC long/flat on the daily close vs its 100-day SMA (±3% band); while out of BTC the book sits in a managed-futures UCITS ETF | Rule-based | 2026-08 |
| **bounce** | S&P 500 dip-reversion (RSI + accumulation/distribution + 20-day drop), trailing-lock exit | Rule-based, shadow | 2026-08 |
| **illiquid** | Small-cap illiquidity premium: low-ADV, spread-trimmed names ranked by the small-cap LambdaRank model | ML, shadow | 2026-07 |

Each book starts at €1,000, runs daily in GitHub Actions and keeps its state in
Supabase. Retired: the original small-cap "baseline" book (no edge after
realistic spreads) and an adaptive-stop variant.

**What the backtests honestly say** (details and caveats in PROJECT.md):

- *liquidcap* — 34 purged walk-forward folds 2018-2026: +1.7%/month at flat
  5 bps, +0.95%/month with spread-aware costs. The monthly portfolio Sharpe is
  ≈0.7 (similar to the index): the edge is excess return, not lower risk.
- *btc_trend* — 2015-2026 with a managed-futures proxy: ~69%/yr CAGR but a
  **~−71% max drawdown** (2018). Trend-following in crypto works on the long
  side only and mainly by avoiding crashes; it lags buy-and-hold in strong bull years.
- Nothing found reliably reaches the "5%/month" often claimed online: the
  published strategies that did (e.g. a Larry Williams breakout on ETH) died
  out of sample.

## Architecture

```
            ┌──────────────── GitHub Actions (cron via Cloudflare Worker) ───────────────┐
 Polygon ──►│ daily.yml     small-cap data → features → weekly retrain → signals          │
 yfinance ─►│ liquidcap.yml S&P 500 data → EDGAR fundamentals → retrain → book            │
 SEC EDGAR─►│ btc.yml       Binance/Coinbase daily candle → trend rule → BTC ↔ CTA book    │──► Supabase
 Binance ──►│ bounce.yml    S&P 500 dip-reversion shadow book                             │   (state, NAV,
            │ monitor.yml   weekly live-IC monitor (out-of-sample alarm)                  │    signals, views,
            └─────────────────────────────────────────────────────────────────────────────┘    live prices)
                                                                                                    │
 Cloudflare Worker (Finnhub quotes, market hours) ──► live_prices ─────────────────────────────────┤
 Browser (Binance public API, 24/7 for crypto) ─────────────────────────────────────────────────────┤
                                                                                                    ▼
                                          GitHub Pages: static login shell → Supabase Auth (RLS)
                                          → render-ready dashboard views (no trade data in the page)
```

## Repository layout

```
src/app/
  data/          Polygon ("massive") client, Supabase store, Parquet store (DuckDB),
                 free sources (Yahoo, SEC EDGAR), small-cap download + universe discovery
  features/      point-in-time feature pipeline: price/volume, volatility, liquidity,
                 microstructure, cross-sectional/sector, fundamentals, tradability gate
  paper_trading.py   PaperTrader engine (fills, trailing stops, profit target, splits,
                     NaN-price guard) shared by the ML books
  signal_tracker.py  signal history + matured-outcome backfill
  web/           dashboard view builder, FastAPI local server, HTML template
scripts/
  daily_pipeline.py, illiquid_shadow.py      small-cap model + illiquid shadow book
  liquidcap/                                 S&P 500 book: universe, fundamentals, features, daily job
  btc_trend_daily.py, bounce_daily.py        rule-based books
  monitor_live_ic.py                         out-of-sample live-IC monitor
  v3/                                        evaluation harness + anti-leak verifier
  archived/                                  finished research (not maintained)
cloudflare/      scheduler worker (reliable cron) + live-prices worker
data/            committed foundational data (OHLCV panels, models, benchmark results)
tests/unit/      unit tests run by CI
```

## Quick start

```bash
python -m pip install -e ".[dev]"      # Python 3.11
cp .env.example .env                  # fill in keys (Polygon, Supabase, …)
make test lint                        # what CI runs
make btc-dry                          # run a book without writing anything
make web                              # local dashboard → http://localhost:8501
```

Operations, schedules, secrets and the incident runbook: [docs/OPERATIONS.md](docs/OPERATIONS.md).
Full technical documentation and research log: [PROJECT.md](PROJECT.md).

## Tech stack

Python 3.11 · pandas · LightGBM (LambdaRank) · DuckDB/Parquet · pydantic-settings ·
structlog · FastAPI/Jinja2 · Supabase (Postgres + Auth + RLS) · GitHub Actions ·
GitHub Pages · Cloudflare Workers · Telegram alerts.

*Paper trading only. Not investment advice.*
