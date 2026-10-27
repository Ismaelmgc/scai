# Operations

How the system runs in production, how to set it up from scratch, and what to
do when something breaks. Everything runs on free tiers: GitHub Actions (public
repo), GitHub Pages, Supabase, Cloudflare Workers.

## Daily schedule

The Cloudflare **scheduler worker** (`cloudflare/scheduler`) dispatches the
workflows on time; some workflows also keep a best-effort GitHub `schedule:` as a
backup. Every job is idempotent, so a duplicate trigger is harmless.

| UTC | Workflow | Job |
|---|---|---|
| 00:10 daily | `btc.yml` | BTC ↔ managed-futures book (daily candle closes 00:00 UTC) |
| 03:00 weekdays (+ backup daily) | `liquidcap.yml` | S&P 500 book; weekly retrain |
| 04:00 daily | `bounce.yml` | S&P 500 dip-reversion shadow book |
| 06:00 daily | `daily.yml` | small-cap model + illiquid shadow book + Pages deploy + data commit |
| Mon 13:00 | `monitor.yml` | live-IC monitor (red run + GitHub email if the model degrades) |
| 1st of month 04:00 | `liquidcap-maintain.yml` | S&P 500 membership refresh (+ archive departed members) |
| 1st of month 06:00 | `universe.yml` | small-cap universe refresh |
| on push | `ci.yml` | `ruff check src/ tests/` + `pytest tests/unit` |
| on push to dashboard files / on demand | `pages.yml` | re-render and deploy the dashboard shell |

Every book sends a Telegram summary per run and an alert on failure.

## One-time setup

1. **Secrets** (repo → Settings → Secrets and variables → Actions):

   | Secret | Used by |
   |---|---|
   | `SCAI_POLYGON_API_KEY` | small-cap data (daily, universe) |
   | `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` | every book (state, NAV, signals, views) |
   | `SUPABASE_ANON_KEY` | Pages render (public, RLS-protected) |
   | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | run summaries / alerts (optional) |

   On the free Polygon tier set `MASSIVE_CALLS_PER_MINUTE=5` in the workflow env
   (already the case): the download takes longer but does not fail.
2. **Supabase**: create the tables and RLS policies in
   `scripts/supabase/schema.sql`. Views are readable only by authenticated users;
   the browser never receives raw trade tables.
3. **Pages**: repo → Settings → Pages → Source: *GitHub Actions*.
4. **Cloudflare workers** (`wrangler deploy` in each folder):
   - `cloudflare/scheduler` — needs a fine-grained PAT (`Actions: Read and write`)
     as the `GITHUB_TOKEN` secret.
   - `cloudflare/live-prices` — needs `SUPABASE_SERVICE_KEY` and `FINNHUB_TOKEN`;
     polls Finnhub quotes for open positions during US market hours. Crypto
     positions are priced 24/7 directly in the browser from Binance's public API.

## Local use

```bash
make test lint          # what CI runs
make btc-dry            # any book: btc-dry / liquidcap-dry / bounce-dry / smallcap-dry
make web                # local dashboard on http://localhost:8501
make leak-check         # anti-leakage gate for the small-cap retrain
```

`.env` (from `.env.example`) provides the same keys locally. The small-cap
pipeline needs the committed OHLCV panel in `data/processed/`; the liquidcap
panel is git-ignored and rebuilt by its job (bootstrap with
`scripts/liquidcap/download_tiingo.py` + `download_yf_fallback.py`).

## Runbook

**All parquet-reading jobs fail at once, no code change** (e.g. `Unable to find a
usable engine; tried using: 'pyarrow'`). CI installs fresh dependencies every
run, so a new upstream release can break it overnight. Compare the first failure
time with PyPI upload times. 2026-10-09: pyarrow 26 required NumPy 2 while we pin
`numpy<2` → fixed by `pyarrow<26`. Heavy dependencies have upper bounds in
`pyproject.toml` for this reason.

**A book crashes on arithmetic with `None`** (`unsupported operand ... NoneType`).
A NaN price reached the book and was stored as `null` in `portfolio_state`; every
later run then crashes. `PaperTrader` now holds positions/keeps orders pending on
non-finite prices (`_finite_price`), and the BTC book refuses to mark itself
without prices. To repair: inspect `read_state(<book>)`, rebuild the bad fields
from the ledger (initial capital + Σ closed P&L − Σ open cost), back up first.

**A job was down for several sessions.** `bounce`, `btc_trend` and the small-cap
pipeline catch up missed days automatically. `liquidcap` and `illiquid` process
only the latest session: exits that should have happened on skipped days must
be replayed manually with the same `PaperTrader` config and the real closes of
those days, then the NAV backfilled.

**Liquidcap run "canceled" during the weekly retrain.** Out of memory: the full
feature build peaks above the runner's RAM. The workflow enlarges swap to 16 GB
before running; keep that step.

**Small-cap download 403 / partial session.** Free Polygon publishes a session's
grouped bar hours after the close; the job runs at 06:00 UTC for that reason.
A past-session error fails the run (red) instead of committing stale data, and
the Telegram summary reports per-session coverage.

**A split shows as a huge loss.** Prices are split-adjusted retroactively while a
stored entry price is not. `PaperTrader._reconcile_splits` rescales open
positions; check it ran before trusting a sudden −50%/−70% trade.
