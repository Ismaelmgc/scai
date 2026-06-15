/**
 * SCAI live-prices worker.
 *
 * Finnhub's FREE WebSocket only streams highly-liquid large-caps, not the
 * small-caps in the portfolio — but its REST /quote returns them fine (and has
 * no CORS, so it can't be called from the browser). This Cloudflare Worker runs
 * on a 1-minute cron during US market hours, reads the open-position tickers
 * from Supabase, fetches their quotes from Finnhub REST, and UPSERTs them into
 * the `live_prices` table (one row per ticker → constant size). The logged-in
 * dashboard reads that table to fill the live price/percent columns.
 *
 * Secrets (wrangler secret put): SUPABASE_SERVICE_KEY, FINNHUB_TOKEN.
 * Vars (wrangler.toml [vars]): SUPABASE_URL.
 */
const FINNHUB_QUOTE = "https://finnhub.io/api/v1/quote";

function marketOpenET(now = new Date()) {
  const et = new Date(now.toLocaleString("en-US", { timeZone: "America/New_York" }));
  const day = et.getDay();
  if (day === 0 || day === 6) return false; // weekend
  const mins = et.getHours() * 60 + et.getMinutes();
  return mins >= 9 * 60 + 30 && mins <= 16 * 60; // 09:30–16:00 ET
}

async function sbGet(env, path) {
  const r = await fetch(`${env.SUPABASE_URL}/rest/v1/${path}`, {
    headers: {
      apikey: env.SUPABASE_SERVICE_KEY,
      Authorization: `Bearer ${env.SUPABASE_SERVICE_KEY}`,
    },
  });
  if (!r.ok) throw new Error(`supabase GET ${path}: ${r.status}`);
  return r.json();
}

async function openTickers(env) {
  const rows = await sbGet(env, "portfolio_state?select=state");
  const set = new Set();
  for (const row of rows) {
    for (const p of (row.state && row.state.positions) || []) {
      if (p.ticker) set.add(p.ticker);
    }
  }
  return [...set];
}

async function update(env) {
  const tickers = await openTickers(env);
  if (!tickers.length) return { updated: 0, reason: "no open positions" };

  const now = new Date().toISOString();
  const rows = [];
  for (const t of tickers) {
    try {
      const q = await fetch(
        `${FINNHUB_QUOTE}?symbol=${encodeURIComponent(t)}&token=${env.FINNHUB_TOKEN}`,
      ).then((r) => r.json());
      if (q && q.c) {
        rows.push({ ticker: t, price: q.c, change_percent: q.dp ?? 0, updated_at: now });
      }
    } catch (e) {
      // skip this ticker; best-effort
    }
  }
  if (!rows.length) return { updated: 0, reason: "no quotes" };

  const r = await fetch(`${env.SUPABASE_URL}/rest/v1/live_prices?on_conflict=ticker`, {
    method: "POST",
    headers: {
      apikey: env.SUPABASE_SERVICE_KEY,
      Authorization: `Bearer ${env.SUPABASE_SERVICE_KEY}`,
      "Content-Type": "application/json",
      Prefer: "resolution=merge-duplicates",
    },
    body: JSON.stringify(rows),
  });
  if (!r.ok) throw new Error(`supabase upsert: ${r.status} ${await r.text()}`);
  return { updated: rows.length, tickers: rows.map((x) => x.ticker) };
}

export default {
  async scheduled(event, env, ctx) {
    if (!marketOpenET()) return; // cron over-schedules in UTC; gate precisely by ET
    ctx.waitUntil(update(env));
  },

  // Manual trigger for testing: GET the worker URL (add ?force=1 to run off-hours).
  async fetch(request, env) {
    const url = new URL(request.url);
    if (!marketOpenET() && url.searchParams.get("force") !== "1") {
      return Response.json({ skipped: "market closed (add ?force=1 to override)" });
    }
    try {
      return Response.json(await update(env));
    } catch (e) {
      return Response.json({ error: String(e) }, { status: 500 });
    }
  },
};
