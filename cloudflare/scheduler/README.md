# scai-scheduler (Cloudflare Worker)

Reliable cron trigger for the paper-trading jobs. GitHub's scheduled cron is
best-effort (runs late / gets dropped under load); this Worker fires on
Cloudflare's cron and dispatches the matching GitHub Actions workflow via
`workflow_dispatch`, so the jobs run on time.

## What it triggers

| Cloudflare cron (UTC) | Workflow        | When |
| --------------------- | --------------- | ---- |
| `10 0 * * *`          | `btc.yml`       | 00:10 UTC every day (BTC daily candle closes 00:00 UTC) |
| `0 3 * * 1-5`         | `liquidcap.yml` | 03:00 UTC weekdays |
| `0 4 * * *`           | `bounce.yml`    | 04:00 UTC every day |
| `0 6 * * *`           | `daily.yml`     | 06:00 UTC every day |

All jobs are idempotent (Supabase state guards), so a duplicate trigger — e.g.
a workflow's own backup `schedule:` firing as well — is harmless. `daily.yml`
runs every day on purpose: a weekend run self-heals a failed Friday.

## Config & secret

- `wrangler.toml [vars]`: `GITHUB_REPO` (`Ismaelmgc/scai`), `GITHUB_REF` (`main`).
- Secret: `GITHUB_TOKEN` — a **fine-grained PAT** scoped to the `scai` repo with
  Repository permission **Actions: Read and write** (nothing else). This is the
  minimum needed to call `workflow_dispatch`.

## Deploy

```bash
cd cloudflare/scheduler
npm i -g wrangler          # if not installed
wrangler login             # once, to the SCAI Cloudflare account
wrangler secret put GITHUB_TOKEN   # paste the fine-grained PAT
wrangler deploy
```

## Test without waiting for cron

The deploy prints the Worker URL (`https://scai-scheduler.<subdomain>.workers.dev`).

```bash
curl "https://scai-scheduler.<subdomain>.workers.dev/?job=btc"        # dispatch the BTC book now
curl "https://scai-scheduler.<subdomain>.workers.dev/?job=liquidcap"  # dispatch liquidcap now
```

A successful dispatch returns `{"job":...,"status":204}` and a run appears under
GitHub → Actions for that workflow. Watch a scheduled fire live with
`wrangler tail`.
