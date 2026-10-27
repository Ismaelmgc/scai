/**
 * SCAI scheduler worker.
 *
 * GitHub's scheduled cron is best-effort: it routinely runs hours late or is
 * dropped entirely under load, which delays or skips the paper-trading jobs.
 * This Worker fires on Cloudflare's (reliable) cron and dispatches the matching
 * GitHub Actions workflow via the REST API (workflow_dispatch), so each job runs
 * on time regardless of GitHub's own scheduler. The workflows keep their
 * workflow_dispatch trigger (manual "Run workflow" still works).
 *
 * Jobs:
 *   00:10 UTC every day -> btc.yml        (BTC <-> managed futures; daily candle closes 00:00 UTC)
 *   03:00 UTC weekdays  -> liquidcap.yml  (S&P 500 book; yfinance EOD ready)
 *   04:00 UTC every day -> bounce.yml     (S&P 500 dip-reversion shadow book)
 *   06:00 UTC every day -> daily.yml      (small-cap model + illiquid shadow book;
 *                                          daily so a failed Fri self-heals Sat/Sun)
 *
 * Config: GITHUB_REPO, GITHUB_REF in wrangler.toml [vars].
 * Secret (wrangler secret put): GITHUB_TOKEN — fine-grained PAT on this repo
 * with Repository permission "Actions: Read and write".
 */

const WORKFLOWS = {
  liquidcap: "liquidcap.yml",
  daily: "daily.yml",
  bounce: "bounce.yml",
  btc: "btc.yml",
};

// Map a matched cron expression (exactly as registered in wrangler.toml) to its job.
const CRON_JOBS = {
  "0 6 * * *": "daily",
  "0 3 * * 1-5": "liquidcap",
  "0 4 * * *": "bounce",
  "10 0 * * *": "btc",
};

async function dispatch(env, job) {
  const file = WORKFLOWS[job];
  if (!file) throw new Error(`unknown job: ${job}`);
  const r = await fetch(
    `https://api.github.com/repos/${env.GITHUB_REPO}/actions/workflows/${file}/dispatches`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${env.GITHUB_TOKEN}`,
        Accept: "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "scai-scheduler",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ ref: env.GITHUB_REF || "main" }),
    },
  );
  // GitHub returns 204 No Content on a successful dispatch.
  if (r.status !== 204) throw new Error(`dispatch ${file}: ${r.status} ${await r.text()}`);
  return { job, file, status: r.status };
}

export default {
  async scheduled(event, env, ctx) {
    const job = CRON_JOBS[event.cron];
    if (!job) return;
    ctx.waitUntil(dispatch(env, job));
  },

  // Manual trigger for testing: GET the worker URL with ?job=daily|liquidcap|bounce|btc
  // to dispatch immediately. No arg -> usage JSON.
  async fetch(request, env) {
    const url = new URL(request.url);
    const job = url.searchParams.get("job");
    if (!job) {
      return Response.json({
        worker: "scai-scheduler",
        usage: "GET ?job=daily|liquidcap|bounce|btc to dispatch now",
        jobs: Object.keys(WORKFLOWS),
      });
    }
    try {
      return Response.json(await dispatch(env, job));
    } catch (e) {
      return Response.json({ error: String(e) }, { status: 500 });
    }
  },
};
