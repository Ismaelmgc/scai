# scai-scheduler (Cloudflare Worker)

Reliable cron trigger for the paper-trading jobs. GitHub's scheduled cron is
best-effort (runs late / gets dropped under load); this Worker fires on
Cloudflare's cron and dispatches the matching GitHub Actions workflow via
`workflow_dispatch`, so the jobs run on time.

## What it triggers (weekdays)

| Cloudflare cron (UTC) | Workflow        | When (ET)        |
| --------------------- | --------------- | ---------------- |
| `0 3 * * 1-5`         | `liquidcap.yml` | 03:00 UTC (fixed) |
| `5 11 * * 1-5`        | `daily.yml`     | 11:05 UTC (fixed) |
| `32 13,14 * * 1-5`    | `morning.yml`   | 09:32 ET (open+2min) |

The morning cron fires at both 13:32 and 14:32 UTC; the Worker only dispatches
the one that is 09:32 in New York, so it stays at open+2min across EDT and EST.

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
curl "https://scai-scheduler.<subdomain>.workers.dev/?job=daily"      # dispatch small-cap daily now
curl "https://scai-scheduler.<subdomain>.workers.dev/?job=liquidcap"  # dispatch liquidcap daily now
curl "https://scai-scheduler.<subdomain>.workers.dev/?job=morning"    # dispatch morning fill now
```

A successful dispatch returns `{"job":...,"status":204}` and a run appears under
GitHub → Actions for that workflow. Watch a scheduled fire live with
`wrangler tail`.

## Rollout note

While both this Worker and the workflows' own `schedule:` crons are active, a job
can be triggered twice (harmless — the jobs are idempotent via Supabase state).
Once the Worker is verified, remove the `schedule:` block from `daily.yml`,
`liquidcap.yml`, and `morning.yml` (keep `workflow_dispatch:`) so the Worker is
the sole scheduled trigger and there is no duplicate work.
