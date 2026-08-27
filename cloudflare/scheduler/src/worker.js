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
 * Jobs (weekdays):
 *   03:00 UTC          -> liquidcap.yml  (S&P 500 daily; yfinance EOD ready)
 *   11:05 UTC          -> daily.yml      (small-cap daily; after free Polygon T+1)
 *   09:32 ET (open+2m) -> morning.yml    (fill pending BUYs at the open)
 *
 * The morning cron is registered at BOTH 13:32 and 14:32 UTC (one expression);
 * the handler gates on New York local time == 09:32 so exactly one of the two
 * fires per season (EDT vs EST) — the same DST-robust trick the native workflow
 * uses, just 13 min earlier (open+2min instead of 09:45).
 *
 * Config: GITHUB_REPO, GITHUB_REF in wrangler.toml [vars].
 * Secret (wrangler secret put): GITHUB_TOKEN — fine-grained PAT on this repo
 * with Repository permission "Actions: Read and write".
 */

const WORKFLOWS = {
  liquidcap: "liquidcap.yml",
  daily: "daily.yml",
  morning: "morning.yml",
};

// Map a matched cron expression (exactly as registered in wrangler.toml) to its
// job. The morning entry is additionally ET-gated in scheduled().
const CRON_JOBS = {
  "0 3 * * 1-5": "liquidcap",
  "5 11 * * 1-5": "daily",
  "32 13,14 * * 1-5": "morning",
};

function nyHourMinute(now = new Date()) {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
  }).formatToParts(now);
  const get = (t) => Number(parts.find((x) => x.type === t).value);
  return { hour: get("hour"), minute: get("minute") };
}

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
    if (job === "morning") {
      // The 13,14 cron fires twice (EDT and EST candidates); only the one that
      // lands on 09:32 New York time is the real open+2min for the season.
      const { hour, minute } = nyHourMinute();
      if (!(hour === 9 && minute === 32)) return;
    }
    ctx.waitUntil(dispatch(env, job));
  },

  // Manual trigger for testing: GET the worker URL with ?job=daily|liquidcap|morning
  // to dispatch immediately (bypasses the ET gate). No arg -> usage JSON.
  async fetch(request, env) {
    const url = new URL(request.url);
    const job = url.searchParams.get("job");
    if (!job) {
      return Response.json({
        worker: "scai-scheduler",
        usage: "GET ?job=daily|liquidcap|morning to dispatch now",
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
