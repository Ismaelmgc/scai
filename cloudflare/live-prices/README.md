# SCAI live-prices worker (Cloudflare)

A Cloudflare Worker that polls Finnhub REST for the open-position tickers every
minute during US market hours and upserts them into the Supabase `live_prices`
table (one row per ticker, constant size). The logged-in dashboard reads that
table to show live prices for small-caps (Finnhub's free WebSocket doesn't
stream them; its REST does, but only server-side — no CORS).

## One-time deploy

Prereqs: a free [Cloudflare account](https://dash.cloudflare.com/sign-up) and
Node.js installed.

```bash
cd cloudflare/live-prices

# 1. Log in to Cloudflare (opens a browser)
npx wrangler login

# 2. Set the Supabase project URL in wrangler.toml ([vars] SUPABASE_URL),
#    replacing https://YOUR-PROJECT.supabase.co

# 3. Add the two secrets (encrypted, never in git):
npx wrangler secret put SUPABASE_SERVICE_KEY   # paste the Supabase service_role key
npx wrangler secret put FINNHUB_TOKEN          # paste the Finnhub token

# 4. Deploy
npx wrangler deploy
```

## Verify

- Manual run (during market hours): open the worker URL printed by `deploy`.
  Off-hours add `?force=1` → it should return `{"updated": N, "tickers": [...]}`.
- Check the table in Supabase: `select * from live_prices;` → ~N rows, one per
  open position, `updated_at` recent.
- The cron (`* 13-21 * * 1-5`) then refreshes every minute during market hours;
  the worker exits early outside 09:30–16:00 ET.

## Cost

Free tier: every-minute cron ≈ 1.4k invocations/day (limit 100k/day). Supabase
storage is constant (upsert by ticker). Finnhub: ≤ (open positions) calls/min,
well under the 60/min free limit.
