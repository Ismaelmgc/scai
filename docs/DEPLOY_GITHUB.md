# Deploy SCAI on GitHub (no PC required)

This runs the production daily pipeline on GitHub's servers and publishes the
dashboard as a static site on GitHub Pages. Your PC never needs to be on.

- **Pipeline** → GitHub Actions, weekday cron (`.github/workflows/daily.yml`).
- **Dashboard** → static snapshot rendered each run, served from GitHub Pages
  (`scripts/render_static_dashboard.py`). Public URL, mobile-friendly.
- **State** → each run commits the updated `data/` (portfolios, OHLCV, model)
  back to the repo so the next run continues from it.

## What's automated vs. what you do once

The code is ready. You do the following one-time setup on GitHub.

### 1. Create a public repo and push

```powershell
cd C:\Users\34630\Downloads\SCAI
git add -A
git commit -m "Initial commit: SCAI + GitHub Actions deploy"
gh repo create scai --public --source=. --remote=origin --push
# or, without gh:  git remote add origin https://github.com/<you>/scai.git ; git push -u origin main
```

The 2.2 GB `features_smallcap.parquet` is git-ignored (rebuilt each run); the
~20 MB OHLCV bootstrap, the model, and the portfolios are committed.

### 2. Add your Polygon API key as a secret

Repo → **Settings → Secrets and variables → Actions → New repository secret**

- Name: `SCAI_POLYGON_API_KEY`
- Value: the key from your local `.env`

(`.env` itself is git-ignored and never pushed.)

### 3. Turn on GitHub Pages from Actions

Repo → **Settings → Pages → Build and deployment → Source: GitHub Actions**.

### 4. Do a test run

Repo → **Actions → "SCAI daily" → Run workflow**. This triggers it now instead
of waiting for the cron. Watch the logs. On success, the Pages URL appears on the
deploy step and under Settings → Pages (e.g. `https://<you>.github.io/scai/`).
Open it on your phone.

## Schedule

`.github/workflows/daily.yml` runs weekdays at **21:05 UTC**, which is at/after the
US close (16:00 ET) all year: exactly +5 min in winter, ~+65 min in summer (UTC
cron can't track DST). It never fires before close, so the EOD bar is final.
GitHub cron is best-effort and may be delayed a few minutes; the pipeline is
idempotent, so that's harmless.

Want exactly 16:05 ET year-round? Add a second cron `5 20 * * 1-5` plus an
early-exit guard for "before 16:00 ET" (otherwise the summer trigger fires at
15:05 ET, before close). Single-cron is simpler and was chosen as the default.

## Things to watch on the first run

- **Memory.** Building the 2.2 GB feature matrix over ~830K rows on a free
  runner (2 vCPU / 7 GB) is the most likely failure point. If the job OOMs,
  options: a larger (paid) runner, or reduce the feature-build footprint.
- **Download time.** Incremental Polygon pull for ~1000 tickers at 50 calls/min
  is ~15-20 min per run — within the job limit, just not instant.
- **Cost.** Public repo = unlimited Actions minutes, free Pages. Nothing to pay.

## Privacy note

The Pages site is **public** (free Pages requires a public repo). It shows paper
-trading results only — no API keys (those stay in the secret), no real money, no
personal data. If you later want it private, switch to a free PaaS host
(Render/Railway) running the live FastAPI app with a password instead of Pages.

## Local use still works

`scai run`, `scai web` (http://localhost:8501), and `scai monitor` are unchanged.
To preview the static snapshot locally:

```powershell
$env:PYTHONPATH="src"; .venv\Scripts\python.exe scripts\render_static_dashboard.py
# open site\index.html
```
