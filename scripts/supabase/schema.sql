-- SCAI — paper-trading state schema for Supabase (run once in the SQL Editor).
-- Reads are restricted to LOGGED-IN users (Supabase Auth); the pipeline writes
-- with the service_role key, which bypasses RLS, so no write policies are needed.
-- The public dashboard is a shell: it logs in, then reads `dashboard_view`.

create table if not exists portfolio_state (
  strategy   text primary key,           -- 'baseline' | 'adaptive'
  state      jsonb not null,             -- full PortfolioState (asdict)
  updated_at timestamptz not null default now()
);

create table if not exists trades (
  id          bigserial primary key,
  strategy    text not null,
  ticker      text not null,
  entry_date  date,
  exit_date   date,
  entry_price numeric,
  exit_price  numeric,
  shares      numeric,
  pnl_pct     numeric,
  pnl_usd     numeric,
  exit_reason text,
  days_held   int,
  created_at  timestamptz not null default now()
);
-- Dedup: a closed trade is unique per strategy/ticker/entry/exit → idempotent appends.
create unique index if not exists trades_unique_idx
  on trades(strategy, ticker, entry_date, exit_date);
create index if not exists trades_strategy_exit_idx on trades(strategy, exit_date);

create table if not exists signals (
  id             bigserial primary key,
  strategy       text not null,
  signal_date    date not null,
  ticker         text not null,
  score          numeric,
  recommendation text,
  was_traded     boolean,
  skip_reason    text,
  actual_ret_20d numeric,
  created_at     timestamptz not null default now()
);
create unique index if not exists signals_unique_idx
  on signals(strategy, signal_date, ticker);

create table if not exists nav_history (
  strategy        text not null,
  date            date not null,
  portfolio_value numeric not null,
  primary key (strategy, date)
);

-- Render-ready dashboard view (one row per strategy): the exact dict the front
-- needs to paint KPIs/chart/tables. The pipeline computes + upserts it (it has
-- OHLCV + the service key); the logged-in client reads it. Keeps raw tables out
-- of the browser and avoids client-side computation.
create table if not exists dashboard_view (
  strategy   text primary key,           -- 'baseline' | 'adaptive'
  view       jsonb not null,             -- {paper, signals, data_info, ...}
  updated_at timestamptz not null default now()
);

-- Row Level Security: enable on all tables.
alter table portfolio_state enable row level security;
alter table trades          enable row level security;
alter table signals         enable row level security;
alter table nav_history     enable row level security;
alter table dashboard_view  enable row level security;

-- Reads require a logged-in user (Supabase Auth). The anon key alone (embedded
-- in the public HTML) can no longer read anything. To rotate from the old public
-- policies, drop them first:
--   drop policy if exists "public read portfolio_state" on portfolio_state;
--   drop policy if exists "public read trades"          on trades;
--   drop policy if exists "public read signals"         on signals;
--   drop policy if exists "public read nav_history"     on nav_history;
create policy "auth read portfolio_state" on portfolio_state for select to authenticated using (true);
create policy "auth read trades"          on trades          for select to authenticated using (true);
create policy "auth read signals"         on signals         for select to authenticated using (true);
create policy "auth read nav_history"     on nav_history     for select to authenticated using (true);
create policy "auth read dashboard_view"  on dashboard_view  for select to authenticated using (true);
