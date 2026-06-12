#!/usr/bin/env python3
"""SCAI – Full Small-Cap Pipeline using Massive API.

Discovers US small-cap universe dynamically from the API (Russell 2000-style),
downloads OHLCV + corporate actions, trains models, generates predictions,
and runs a backtest.

This is the SINGLE pipeline script for the project. Update it in place
rather than creating new scripts.

Usage:
    python scripts/run_smallcap_pipeline.py
    python scripts/run_smallcap_pipeline.py --train-start 2022-01-01
    python scripts/run_smallcap_pipeline.py --max-tickers 50
    python scripts/run_smallcap_pipeline.py --skip-download  # reuse cached data
"""

from __future__ import annotations

import argparse
import faulthandler
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

# Enable faulthandler to print traceback on segfault
faulthandler.enable()

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from app.config import get_settings
from app.utils import setup_logging, set_global_seed, get_logger

log = get_logger(__name__)

# ── Russell 2000-style small-cap seed ───────────────────────
# Broadly diversified across sectors matching the Russell 2000 profile.
# The API will verify each has market cap $50M–$2B before inclusion.
RUSSELL_SEED = [
    # ── Technology / Software ───────────────────────────────
    "BBAI",   # BigBear.ai — AI/analytics
    "GENI",   # Genius Sports — sports data
    "RCAT",   # Red Cat Holdings — drones
    "COUR",   # Coursera — edtech
    "PAYO",   # Payoneer — global payments
    "BFLY",   # Butterfly Network — portable ultrasound
    "ENVX",   # Enovix — batteries
    "FLNC",   # Fluence Energy — energy storage
    "SHLS",   # Shoals Technologies — solar connectors
    "ARRY",   # Array Technologies — solar trackers
    "OUST",   # Ouster — lidar
    "NUVB",   # Nuvation Bio — biotech
    "RDW",    # Redwire — space manufacturing
    "TALK",   # Talkspace — telehealth
    "HYLN",   # Hyliion — EV powertrains
    "NNOX",   # Nano-X Imaging — medical imaging
    "SKIN",   # Beauty Health — aesthetics
    "STEM",   # Stem Inc — energy AI
    "CHPT",   # ChargePoint — EV charging
    "DCGO",   # DocGo — mobile health
    "DNA",    # Ginkgo Bioworks — synthetic bio
    "DNUT",   # Krispy Kreme — food/consumer
    "BIGC",   # BigCommerce — ecommerce platform
    "BRZE",   # Braze — customer engagement
    "ALKT",   # Alkami Technology — digital banking
    "SRAD",   # Sportradar — sports data
    "TOST",   # Toast — restaurant tech
    "MAPS",   # WM Technology — cannabis tech
    "BTDR",   # Bitdeer Technologies — crypto mining
    "VERX",   # Vertex Inc — tax tech

    # ── Financials ──────────────────────────────────────────
    "CUBI",   # Customers Bancorp
    "FBK",    # FB Financial
    "BUSE",   # First Busey
    "HOPE",   # Hope Bancorp
    "BANR",   # Banner Financial
    "HTLF",   # Heartland Financial
    "IBOC",   # International Bancshares
    "PPBI",   # Pacific Premier
    "SBCF",   # Seacoast Banking
    "TBBK",   # The Bancorp
    "WSBC",   # WesBanco
    "NBTB",   # NBT Bancorp
    "RBCAA",  # Republic Bancorp
    "FFBC",   # First Financial Bankshares
    "BHLB",   # Berkshire Hills Bancorp
    "CATY",   # Cathay General Bancorp
    "WAFD",   # WaFd Inc (Washington Federal)
    "PNFP",   # Pinnacle Financial Partners
    "GBCI",   # Glacier Bancorp
    "TCBI",   # Texas Capital Bankshares
    "SFBS",   # ServisFirst Bancshares
    "HOMB",   # Home BancFunds
    "ABCB",   # Ameris Bancorp
    "FIBK",   # First Interstate BancSystem
    "SFNC",   # Simmons Financial Group

    # ── Healthcare / Biotech ────────────────────────────────
    "IRTC",   # iRhythm Technologies
    "INMD",   # InMode
    "GMED",   # Globus Medical
    "ITCI",   # Intra-Cellular Therapies
    "HALO",   # Halozyme Therapeutics
    "PRCT",   # PROCEPT BioRobotics
    "TNDM",   # Tandem Diabetes Care
    "ACVA",   # ACV Auctions
    "RXST",   # RxSight
    "SERA",   # Sera Prognostics
    "RVMD",   # Revolution Medicines
    "TVTX",   # Travere Therapeutics
    "BCYC",   # Bicycle Therapeutics
    "NRIX",   # Nurix Therapeutics
    "KROS",   # Keros Therapeutics
    "DYN",    # Dyne Therapeutics
    "DAWN",   # Day One Biopharmaceuticals
    "PCVX",   # Vaxcyte — vaccines
    "KRYS",   # Krystal Biotech
    "IONS",   # Ionis Pharmaceuticals
    "HLAH",   # Hamilton Lane — alt investments
    "MDXH",   # MDxHealth — diagnostics
    "ETNB",   # 89bio — liver diseases
    "IMVT",   # Immunovant — autoimmune
    "ACLX",   # Arcellx — cell therapy
    "FOLD",   # Amicus Therapeutics
    "RARE",   # Ultragenyx Pharmaceutical
    "BEAM",   # Beam Therapeutics — gene editing
    "VCEL",   # Vericel — regenerative medicine

    # ── Industrials / Defense ───────────────────────────────
    "KTOS",   # Kratos Defense
    "VRRM",   # Verra Mobility
    "EAF",    # GrafTech International
    "ATKR",   # Atkore
    "MYRG",   # MYR Group — electrical construction
    "PRIM",   # Primoris Services
    "STRL",   # Sterling Infrastructure
    "ROAD",   # Construction Partners
    "WLDN",   # Willdan Group
    "DLX",    # Deluxe Corporation
    "APOG",   # Apogee Enterprises
    "MTRN",   # Materion
    "ASTE",   # Astec Industries
    "GMS",    # GMS Inc — building products
    "NVEE",   # NV5 Global — infrastructure
    "ASGN",   # ASGN Inc — professional services
    "LNTH",   # Lantheus Holdings — medical imaging
    "TDW",    # Tidewater — offshore energy
    "RUSHA",  # Rush Enterprises — trucks
    "EPAC",   # Enerpac Tool Group

    # ── Energy ──────────────────────────────────────────────
    "MTDR",   # Matador Resources
    "TALO",   # Talos Energy
    "CPE",    # Callon Petroleum
    "GPOR",   # Gulfport Energy
    "RRC",    # Range Resources
    "CEIX",   # CONSOL Energy
    "NEXT",   # NextDecade
    "VET",    # Vermilion Energy
    "PTEN",   # Patterson-UTI Energy
    "PUMP",   # ProPetro Holding
    "REI",    # Ring Energy
    "SM",     # SM Energy
    "CIVI",   # Civitas Resources
    "CHRD",   # Chord Energy
    "VTLE",   # Vital Energy
    "MGY",    # Magnolia Oil & Gas

    # ── Consumer Discretionary ──────────────────────────────
    "BOOT",   # Boot Barn
    "SHCO",   # Soho House
    "PRPL",   # Purple Innovation
    "CTOS",   # Custom Truck One Source
    "XPOF",   # Xponential Fitness
    "PLYA",   # Playa Hotels
    "ARKO",   # ARKO Corp
    "DORM",   # Dorman Products
    "FOXF",   # Fox Factory
    "LCII",   # LCI Industries
    "SEM",    # Select Medical
    "JBSS",   # John B. Sanfilippo
    "CLAR",   # Clarus Corp — outdoor
    "ONEW",   # OneWater Marine
    "LESL",   # Leslie's — pool supplies
    "MODG",   # Topgolf Callaway
    "SMPL",   # Simply Good Foods
    "WRBY",   # Warby Parker — eyewear
    "TASK",   # TaskUs — digital outsourcing
    "FIGS",   # FIGS — healthcare apparel

    # ── REITs / Real Estate ─────────────────────────────────
    "NXRT",   # NexPoint Residential
    "GTY",    # Getty Realty
    "AKR",    # Acadia Realty
    "IIPR",   # Innovative Industrial
    "ELME",   # Elme Communities
    "GOOD",   # Gladstone Commercial
    "UMH",    # UMH Properties
    "APLE",   # Apple Hospitality REIT
    "BRT",    # BRT Realty
    "GMRE",   # Global Medical REIT
    "IIPR",   # Innovative Industrial
    "SACH",   # Sachem Capital — mortgage REIT
    "STAG",   # STAG Industrial
    "JBGS",   # JBG SMITH Properties

    # ── Materials / Mining ──────────────────────────────────
    "CENX",   # Century Aluminum
    "HAYN",   # Haynes International
    "SXC",    # SunCoke Energy
    "ITE",    # ITeos Therapeutics
    "AMRX",   # Amneal Pharmaceuticals
    "MP",     # MP Materials — rare earths
    "RYAM",   # Rayonier Advanced Materials
    "IOSP",   # Innospec
    "HCC",    # Warrior Met Coal
    "ARCH",   # Arch Resources — coal

    # ── Utilities / Infrastructure ──────────────────────────
    "ARIS",   # Aris Water Solutions
    "NWN",    # Northwest Natural
    "UTL",    # UNITIL Corp
    "MSEX",   # Middlesex Water
    "CWEN",   # Clearway Energy
    "NOVA",   # Sunnova Energy — solar
    "RUN",    # Sunrun — residential solar
    "SEDG",   # SolarEdge Technologies
]

# De-duplicate
RUSSELL_SEED = list(dict.fromkeys(RUSSELL_SEED))


def discover_universe(
    ref,
    cfg,
    max_tickers: int = 120,
    store=None,
    existing_universe: pd.DataFrame | None = None,
    train_start: str = "2020-01-01",
) -> list[dict]:
    """Build small-cap universe via point-in-time dynamic discovery.

    Phase 1: Use Polygon list_tickers API to get ALL US common stocks
             (active AND inactive/delisted) to avoid survivorship bias.

    Phase 2: Verify market cap point-in-time via get_ticker_details(date_=).
             - Excludes SPACs, blank checks, closed-end funds, ETFs
             - Reuse already-verified tickers from existing_universe
             - Only check new candidates to fill up to max_tickers

    Phase 3 (post-OHLCV): Filter by liquidity, price, min trading days.
             This happens OUTSIDE this function, in filter_universe_quality().
    """
    OTC_EXCHANGES = {"OTCBB", "GREY", "PINK", "OTCQB", "OTCQX"}

    # SIC codes to exclude (non-operating companies)
    EXCLUDED_SICS = {
        "6726",  # Investment offices (closed-end funds)
        "6722",  # Management investment companies
        "6770",  # Blank checks (SPACs)
        "6795",  # Trusts (except educational, religious, charitable)
    }

    # Name patterns that indicate non-operating / non-tradeable entities
    EXCLUDED_NAME_PATTERNS = [
        "Acquisition Corp",
        "SPAC",
        "Blank Check",
        " Fund ",
        "Fund,",
        "Opportunities Fund",
        "Income Fund",
        "Total Return Fund",
        "Convertible Fund",
        "Municipal Fund",
        "Capital Corp Class A Ordinary",
        "Capital Corp. Class A Ordinary",
    ]

    # ── Phase 1: Get full ticker catalog ──
    # Include BOTH active and inactive tickers to avoid survivorship bias
    catalog_cache_domain = "ticker_catalog_v2"  # v2 = includes inactive
    catalog = None
    if store and store.exists(catalog_cache_domain):
        cached = store.read(catalog_cache_domain)
        if "fetched_at" in cached.columns:
            age = (pd.Timestamp(date.today()) - pd.Timestamp(cached["fetched_at"].iloc[0])).days
            if age < 30:
                catalog = cached
                n_active = (catalog["active"] == True).sum() if "active" in catalog.columns else "?"
                n_inactive = (catalog["active"] == False).sum() if "active" in catalog.columns else "?"
                print(f"  📋 Ticker catalog v2 cached ({age}d old) — "
                      f"{len(catalog)} tickers ({n_active} active, {n_inactive} inactive)")

    if catalog is None:
        print("  📋 Fetching full US ticker catalog (active + inactive)...")
        # Fetch active tickers
        active_tickers = ref.list_tickers(
            market="stocks", locale="us", ticker_type="CS", active=True, limit=1000,
        )
        active_adr = ref.list_tickers(
            market="stocks", locale="us", ticker_type="ADRC", active=True, limit=1000,
        )
        # Fetch INACTIVE/delisted tickers (survivorship bias fix)
        inactive_tickers = ref.list_tickers(
            market="stocks", locale="us", ticker_type="CS", active=False, limit=1000,
        )
        inactive_adr = ref.list_tickers(
            market="stocks", locale="us", ticker_type="ADRC", active=False, limit=1000,
        )

        all_tickers = active_tickers + active_adr + inactive_tickers + inactive_adr
        print(f"    Raw: {len(active_tickers)} active CS + {len(active_adr)} active ADR + "
              f"{len(inactive_tickers)} inactive CS + {len(inactive_adr)} inactive ADR")

        rows = []
        for t in all_tickers:
            exch = t.primary_exchange or ""
            if cfg.exclude_otc and exch in OTC_EXCHANGES:
                continue
            rows.append({
                "ticker": t.ticker,
                "name": t.name or "",
                "exchange": exch,
                "type": t.type,
                "active": t.active,
                "delisted_utc": t.delisted_utc.isoformat() if t.delisted_utc else None,
            })
        catalog = pd.DataFrame(rows)
        catalog["fetched_at"] = date.today().isoformat()

        # Pre-filter: remove obvious non-equities by name
        before = len(catalog)
        name_pattern = "|".join(EXCLUDED_NAME_PATTERNS)
        catalog = catalog[~catalog["name"].str.contains(name_pattern, case=False, na=False)]
        print(f"    Excluded {before - len(catalog)} by name patterns (SPACs/funds)")

        if store:
            store.write(catalog_cache_domain, catalog)
        n_active = (catalog["active"] == True).sum()
        n_inactive = (catalog["active"] == False).sum()
        print(f"  ✓ Catalog v2: {len(catalog)} tickers "
              f"({n_active} active, {n_inactive} inactive/delisted)")

    # Filter catalog: only tickers that were active during training period
    # A delisted ticker is valid if it was delisted AFTER train_start
    train_start_ts = pd.Timestamp(train_start, tz="UTC")
    if "delisted_utc" in catalog.columns:
        # Keep: active tickers OR tickers delisted after train_start
        mask_active = catalog["active"] == True
        mask_delisted_in_period = (
            catalog["delisted_utc"].notna() &
            (pd.to_datetime(catalog["delisted_utc"]) >= train_start_ts)
        )
        eligible = catalog[mask_active | mask_delisted_in_period].copy()
        n_excluded_old = len(catalog) - len(eligible)
        if n_excluded_old > 0:
            print(f"  📅 Excluded {n_excluded_old} tickers delisted before {train_start}")
    else:
        eligible = catalog.copy()

    # ── Phase 2: Verify market caps (point-in-time) ──
    verified = []
    already_known: set[str] = set()
    if existing_universe is not None and not existing_universe.empty:
        for _, row in existing_universe.iterrows():
            # Re-validate existing tickers against new exclusion rules
            name = str(row.get("name", ""))
            sic = str(row.get("sic_code", ""))
            if sic in EXCLUDED_SICS:
                continue
            if any(p.lower() in name.lower() for p in EXCLUDED_NAME_PATTERNS):
                continue
            verified.append({
                "ticker": row["ticker"],
                "name": name,
                "market_cap": row.get("market_cap", 0),
                "sic_code": sic,
                "exchange": row.get("exchange", ""),
                "type": row.get("type", "CS"),
                "active": row.get("active", True),
            })
            already_known.add(row["ticker"])
        print(f"  ♻️  Reusing {len(verified)} previously verified tickers "
              f"(after re-filtering)")

    if len(verified) >= max_tickers:
        print(f"  ✅ Already have {len(verified)} ≥ target {max_tickers}")
        return verified[:max_tickers]

    # Load rejected tickers cache
    rejected_cache_domain = "ticker_rejected_v2"
    rejected_set: set[str] = set()
    if store and store.exists(rejected_cache_domain):
        rej_df = store.read(rejected_cache_domain)
        if "fetched_at" in rej_df.columns:
            rej_age = (pd.Timestamp(date.today()) - pd.Timestamp(rej_df["fetched_at"].iloc[0])).days
            if rej_age < 30:
                rejected_set = set(rej_df["ticker"].tolist())
                print(f"  🚫 Skipping {len(rejected_set)} previously rejected ({rej_age}d old)")

    # Build candidate list: SEED first, then catalog
    seed_set = set(RUSSELL_SEED)
    seed_candidates = [t for t in RUSSELL_SEED if t not in already_known and t not in rejected_set]
    catalog_candidates = [t for t in eligible["ticker"].tolist()
                          if t not in seed_set and t not in already_known and t not in rejected_set]
    import random
    rng = random.Random(42)
    rng.shuffle(catalog_candidates)
    candidates = seed_candidates + catalog_candidates

    need = max_tickers - len(verified)
    print(f"  🔍 Need {need} more tickers | Candidates: {len(seed_candidates)} seed + "
          f"{len(catalog_candidates)} discovered = {len(candidates)}")
    print(f"  ⏳ Rate: 5/min → ~{need * 3 // 5} min estimated")

    # Point-in-time date for market cap verification
    # Use the START of the prediction period so we select stocks
    # that were small-caps at the time we'd be trading
    pit_date = None  # Will use current if None; set for true PIT

    rejected = []
    checked = 0

    for t in candidates:
        if len(verified) >= max_tickers:
            print(f"\n  ✅ Reached target {max_tickers} tickers after checking {checked} new.")
            break

        detail = ref.get_ticker_details(t, date_=pit_date)
        checked += 1
        if detail is None:
            rejected.append(f"{t}(not found)")
            continue

        mcap = detail.market_cap
        if mcap is None:
            rejected.append(f"{t}(no mcap)")
            continue

        mcap_m = mcap / 1e6
        if mcap < cfg.min_market_cap:
            rejected.append(f"{t}(${mcap_m:.0f}M<min)")
            continue
        if mcap > cfg.max_market_cap:
            rejected.append(f"{t}(${mcap_m:.0f}M>max)")
            continue

        if detail.type and detail.type not in ("CS", "ADRC"):
            rejected.append(f"{t}(type={detail.type})")
            continue

        exchange = detail.primary_exchange or ""
        if cfg.exclude_otc and exchange in OTC_EXCHANGES:
            rejected.append(f"{t}(OTC)")
            continue

        # P0: Exclude SPACs, blank checks, closed-end funds by SIC
        sic = detail.sic_code or "N/A"
        if sic in EXCLUDED_SICS:
            rejected.append(f"{t}(SIC={sic} excluded)")
            continue

        # P0: Exclude by name pattern
        name = detail.name or ""
        if any(p.lower() in name.lower() for p in EXCLUDED_NAME_PATTERNS):
            rejected.append(f"{t}(name=SPAC/fund)")
            continue

        source = "seed" if t in seed_set else "discovered"
        is_active = detail.active if hasattr(detail, "active") else True
        verified.append({
            "ticker": t,
            "name": name,
            "market_cap": mcap,
            "sic_code": sic,
            "exchange": exchange,
            "type": detail.type,
            "active": is_active,
        })

        if len(verified) - len(already_known) <= 20 or (len(verified) - len(already_known)) % 25 == 0:
            status = "" if is_active else " [DELISTED]"
            print(f"    ✓ {t:6s} — ${mcap_m:>8,.0f}M — "
                  f"{name[:30]:30s} [{exchange}]{status} ({source})", flush=True)

        if checked % 50 == 0:
            print(f"      ... {checked} checked, {len(verified)} verified, "
                  f"{len(rejected)} rejected", flush=True)

    n_new = len(verified) - len(already_known)
    n_seed = sum(1 for v in verified if v["ticker"] in seed_set)
    n_disc = len(verified) - n_seed
    n_delisted = sum(1 for v in verified if not v.get("active", True))
    print(f"\n  ✓ Universe: {len(verified)} tickers "
          f"({n_seed} seed + {n_disc} discovered, {n_delisted} delisted) "
          f"[{n_new} newly verified]")
    if rejected:
        short = rejected[:15]
        print(f"  ✗ Rejected ({len(rejected)}): {', '.join(short)}")
        if len(rejected) > 15:
            print(f"    ... and {len(rejected) - 15} more")

    # Save rejected tickers cache
    if store and rejected:
        rej_tickers = [r.split("(")[0] for r in rejected]
        all_rejected = rejected_set | set(rej_tickers)
        rej_save = pd.DataFrame({"ticker": sorted(all_rejected)})
        rej_save["fetched_at"] = date.today().isoformat()
        store.write(rejected_cache_domain, rej_save)

    return verified


def filter_universe_quality(ohlcv: pd.DataFrame, universe: list[dict], cfg) -> tuple[pd.DataFrame, list[dict]]:
    """Post-OHLCV quality filter: remove tickers that fail liquidity/price/history checks.

    Applied AFTER downloading OHLCV so we can use actual price & volume data.
    This is critical for avoiding penny-stock noise and illiquid names.

    Returns filtered (ohlcv, universe).
    """
    MIN_TRADING_DAYS = 250   # ~1 year of history minimum
    MIN_PRICE = 1.50         # exclude penny stocks
    MIN_ADV20_USD = 300_000  # minimum 20-day avg dollar volume

    print("  🧹 Post-OHLCV quality filter:")
    initial_tickers = ohlcv["ticker"].nunique()

    # 1. Min trading days
    days_per_ticker = ohlcv.groupby("ticker").size()
    short_tickers = set(days_per_ticker[days_per_ticker < MIN_TRADING_DAYS].index)

    # 2. Min price (median close must be > MIN_PRICE)
    median_price = ohlcv.groupby("ticker")["close"].median()
    penny_tickers = set(median_price[median_price < MIN_PRICE].index)

    # 3. Min ADV20 (average daily dollar volume over last 60 days)
    ohlcv_sorted = ohlcv.sort_values(["ticker", "date"])
    # Use last 60 days for ADV calculation
    recent_cutoff = ohlcv_sorted["date"].max() - pd.Timedelta(days=90)
    recent = ohlcv_sorted[ohlcv_sorted["date"] >= recent_cutoff]
    dollar_vol = recent.groupby("ticker").apply(
        lambda g: (g["close"] * g["volume"]).mean(), include_groups=False
    )
    illiquid_tickers = set(dollar_vol[dollar_vol < MIN_ADV20_USD].index)

    # Combine exclusions
    excluded = short_tickers | penny_tickers | illiquid_tickers
    kept_tickers = set(ohlcv["ticker"].unique()) - excluded

    print(f"    Initial:          {initial_tickers} tickers")
    print(f"    < {MIN_TRADING_DAYS} trading days: {len(short_tickers)} removed")
    print(f"    Price < ${MIN_PRICE}:     {len(penny_tickers)} removed")
    print(f"    ADV20 < ${MIN_ADV20_USD/1e3:.0f}K:    {len(illiquid_tickers)} removed")
    print(f"    Overlap:          {len(short_tickers & penny_tickers & illiquid_tickers)} in multiple")
    print(f"    ✓ Final:          {len(kept_tickers)} tickers")

    # Show what was removed
    if excluded:
        examples = sorted(excluded)[:10]
        reasons = []
        for t in examples:
            r = []
            if t in short_tickers:
                r.append(f"{days_per_ticker.get(t, 0)}d")
            if t in penny_tickers:
                r.append(f"${median_price.get(t, 0):.2f}")
            if t in illiquid_tickers:
                r.append(f"ADV${dollar_vol.get(t, 0)/1e3:.0f}K")
            reasons.append(f"{t}({','.join(r)})")
        print(f"    Removed: {', '.join(reasons)}")
        if len(excluded) > 10:
            print(f"    ... and {len(excluded) - 10} more")

    # Filter OHLCV
    ohlcv_filtered = ohlcv[ohlcv["ticker"].isin(kept_tickers)].copy()

    # Filter universe list
    universe_filtered = [u for u in universe if u["ticker"] in kept_tickers]

    return ohlcv_filtered, universe_filtered


def download_ohlcv(aggs, tickers, train_start, predict_to, existing_ohlcv=None):
    """Download OHLCV bars incrementally — only fetches missing days.

    If existing_ohlcv is provided, determines the last available date per
    ticker and only downloads bars after that date.  New tickers get a
    full history download.
    """
    import time as _time

    all_dfs = []
    failed = []
    skipped = 0
    downloaded = 0
    target_end = pd.Timestamp(predict_to)
    _dl_start = _time.monotonic()

    # Build per-ticker last-date index from existing data
    last_dates: dict[str, pd.Timestamp] = {}
    if existing_ohlcv is not None and not existing_ohlcv.empty:
        existing_ohlcv["date"] = pd.to_datetime(existing_ohlcv["date"])
        last_dates = existing_ohlcv.groupby("ticker")["date"].max().to_dict()

    for i, ticker in enumerate(tickers):
        last = last_dates.get(ticker)

        # Already up-to-date? Skip only if we already have the target day's bar.
        # A 1-day tolerance here meant a run holding yesterday's bar would never
        # fetch today's — the daily cron silently captured no new data for the
        # most recent trading day (bug fixed 2026-06-12). predict_to already
        # encodes weekend/holiday handling, so an exact compare is safe.
        if last is not None and last >= target_end:
            skipped += 1
            continue

        # Determine from_date: day after last known, or full history
        if last is not None:
            from_date = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            label = "incremental"
        else:
            from_date = train_start
            label = "full"

        bars = aggs.get_custom_bars(
            ticker, from_date=from_date, to_date=predict_to, adjusted=True,
        )

        if label == "full" and (not bars or len(bars) < 30):
            failed.append(f"{ticker}({len(bars) if bars else 0})")
            continue

        if bars:
            rows = [{
                "date": pd.Timestamp(b.trading_date),
                "ticker": b.ticker,
                "open": b.open, "high": b.high, "low": b.low,
                "close": b.close, "volume": b.volume,
                "vwap": b.vwap, "transactions": b.transactions,
            } for b in bars]
            all_dfs.append(pd.DataFrame(rows))
            downloaded += 1
            print(f"    ✓ {ticker:6s} — {len(bars):>4d} new bars  "
                  f"({bars[0].trading_date} → {bars[-1].trading_date})  [{label}]")
        else:
            downloaded += 1
            print(f"    · {ticker:6s} — 0 new bars (already current)")

        if downloaded > 0 and downloaded % 10 == 0:
            elapsed = _time.monotonic() - _dl_start
            total_to_download = len(tickers) - skipped
            pct = downloaded / total_to_download if total_to_download else 1
            eta_secs = (elapsed / pct - elapsed) if pct > 0 else 0
            eta_min = eta_secs / 60
            print(f"      ── {downloaded}/{total_to_download} descargados "
                  f"({elapsed/60:.1f} min) · ETA ~{eta_min:.0f} min ──")

    if skipped:
        print(f"    ⏩ {skipped} tickers already up-to-date (skipped)")
    if failed:
        print(f"\n  ⚠ Failed/insufficient: {', '.join(failed)}")

    new_data = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    # Merge new data with existing
    if existing_ohlcv is not None and not existing_ohlcv.empty and not new_data.empty:
        combined = pd.concat([existing_ohlcv, new_data], ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"])
        combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
        return combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    elif existing_ohlcv is not None and not existing_ohlcv.empty:
        return existing_ohlcv
    else:
        return new_data


def download_corporate_actions(ca, tickers, existing_splits=None, existing_divs=None):
    """Download splits and dividends incrementally.

    Checks existing data and only fetches records after the last known date.
    """
    from datetime import date as date_type

    # Determine cutoff: only fetch after the last known record
    split_cutoff = None
    if existing_splits is not None and not existing_splits.empty:
        last_split = pd.to_datetime(existing_splits["date"]).max()
        split_cutoff = last_split.date() if not pd.isna(last_split) else None

    div_cutoff = None
    if existing_divs is not None and not existing_divs.empty:
        last_div = pd.to_datetime(existing_divs["ex_date"]).max()
        div_cutoff = last_div.date() if not pd.isna(last_div) else None

    all_splits, all_divs = [], []
    for ticker in tickers:
        kwargs_s = {"ticker": ticker}
        if split_cutoff:
            kwargs_s["execution_date_gte"] = split_cutoff
        for s in ca.get_splits(**kwargs_s):
            all_splits.append({"ticker": s.ticker, "date": s.execution_date,
                               "split_from": s.split_from, "split_to": s.split_to})

        kwargs_d = {"ticker": ticker}
        if div_cutoff:
            kwargs_d["ex_dividend_date_gte"] = div_cutoff
        for d in ca.get_dividends(**kwargs_d):
            all_divs.append({"ticker": d.ticker, "ex_date": d.ex_dividend_date,
                             "amount": d.cash_amount})

    return all_splits, all_divs


def download_fundamentals(fin_api, tickers, existing_fund=None):
    """Download financial statements incrementally.

    Only fetches filings after the last known filing_date.
    """
    from datetime import date as date_type

    filing_cutoff = None
    if existing_fund is not None and not existing_fund.empty:
        last_filed = pd.to_datetime(existing_fund["filed"]).max()
        if not pd.isna(last_filed):
            filing_cutoff = last_filed.date()

    all_records = []
    for ticker in tickers:
        kwargs = {"ticker": ticker, "limit": 20, "timeframe": "quarterly"}
        if filing_cutoff:
            kwargs["filing_date_gte"] = filing_cutoff
        records = fin_api.get_financials(**kwargs)
        for rec in records:
            if not rec.financials:
                continue
            for section_name, section_data in rec.financials.items():
                if not isinstance(section_data, dict):
                    continue
                for concept, concept_data in section_data.items():
                    if isinstance(concept_data, dict) and "value" in concept_data:
                        all_records.append({
                            "ticker": rec.ticker,
                            "filed": str(rec.filing_date) if rec.filing_date else None,
                            "period_end": str(rec.period_of_report_date) if rec.period_of_report_date else None,
                            "concept": concept,
                            "value": concept_data["value"],
                            "form": rec.timeframe or "",
                        })
    return pd.DataFrame(all_records) if all_records else pd.DataFrame()


def train_models(features, predict_from, cfg):
    """Train multi-model ensemble (LightGBM + XGBoost + CatBoost).

    Uses: SHAP feature selection, purged CV, triple-barrier labels,
    multi-model stacking, probability calibration.
    """
    from app.models.tabular import TabularModel
    from app.models.multi_model import MultiModelEnsemble
    from app.models.feature_selection import select_features

    bt_start = pd.Timestamp(predict_from)
    train_data = features[features["date"] < bt_start].dropna(
        subset=["fwd_ret_5d_positive", "fwd_ret_5d"]
    )
    predict_data = features[features["date"] >= bt_start].copy()

    print(f"  Training rows:   {len(train_data):,}")
    print(f"  Prediction rows: {len(predict_data):,}")

    if len(train_data) < 200:
        print("  ⚠ Insufficient training data. Try earlier --train-start")
        sys.exit(1)

    # --- Feature selection (SHAP-based) ---
    from app.models.tabular import _auto_feature_cols
    raw_features = _auto_feature_cols(train_data)
    # Use risk-adjusted target for feature selection if available (better for our best model)
    shap_target = "fwd_ret_5d_risk_adj_positive" if "fwd_ret_5d_risk_adj_positive" in train_data.columns else "fwd_ret_5d_positive"
    selected_features = select_features(
        train_data, raw_features, shap_target,
        task="classification", top_k=60, method="shap",
    )
    print(f"  ✓ Feature selection (SHAP): {len(raw_features)} → {len(selected_features)} features")

    # --- Multi-model classification (LGB + XGB + CatBoost) ---
    print("  Training multi-model classifier (LightGBM + XGBoost + CatBoost)...")
    cls_model = MultiModelEnsemble(horizon=5, task="classification")
    cls_m = cls_model.train(train_data, feature_cols=selected_features)
    print(f"  ✓ Multi-model cls – AUC: {cls_m.get('val_auc', 0):.4f} "
          f"(models: {cls_m.get('models', '?')}, {cls_m.get('n_features', 0)} features)")

    # Also try triple-barrier target
    tb_label = "tb_label_5d_positive"
    if tb_label in train_data.columns:
        tb_train = train_data.dropna(subset=[tb_label]).copy()
        if len(tb_train) > 200:
            tb_cls = MultiModelEnsemble(horizon=5, task="classification")
            tb_train_renamed = tb_train.copy()
            tb_train_renamed = tb_train_renamed.drop(columns=["fwd_ret_5d_positive"], errors="ignore")
            tb_train_renamed = tb_train_renamed.rename(columns={tb_label: "fwd_ret_5d_positive"})
            tb_m = tb_cls.train(tb_train_renamed, feature_cols=selected_features)
            tb_auc = tb_m.get("val_auc", 0)
            print(f"  ✓ Triple-barrier cls – AUC: {tb_auc:.4f}")
            if tb_auc > cls_m.get("val_auc", 0) + 0.002:
                print(f"    → Using triple-barrier model (AUC {tb_auc:.4f} > {cls_m.get('val_auc', 0):.4f})")
                cls_model = tb_cls
                cls_m = tb_m

    # Also try risk-adjusted target
    ra_label = "fwd_ret_5d_risk_adj_positive"
    if ra_label in train_data.columns:
        ra_train = train_data.dropna(subset=[ra_label]).copy()
        if len(ra_train) > 200:
            ra_cls = MultiModelEnsemble(horizon=5, task="classification")
            ra_train = ra_train.drop(columns=["fwd_ret_5d_positive"], errors="ignore")
            ra_train = ra_train.rename(columns={ra_label: "fwd_ret_5d_positive"})
            ra_m = ra_cls.train(ra_train, feature_cols=selected_features)
            ra_auc = ra_m.get("val_auc", 0)
            print(f"  ✓ Risk-adjusted cls – AUC: {ra_auc:.4f}")
            if ra_auc > cls_m.get("val_auc", 0) + 0.002:
                print(f"    → Using risk-adjusted model (AUC {ra_auc:.4f} > {cls_m.get('val_auc', 0):.4f})")
                cls_model = ra_cls
                cls_m = ra_m

    # Also try cross-sectional relative target (outperform daily median)
    xsec_label = "fwd_ret_5d_xsec_positive"
    if xsec_label in train_data.columns:
        xsec_train = train_data.dropna(subset=[xsec_label]).copy()
        if len(xsec_train) > 200:
            xsec_cls = MultiModelEnsemble(horizon=5, task="classification")
            xsec_train_r = xsec_train.drop(columns=["fwd_ret_5d_positive"], errors="ignore")
            xsec_train_r = xsec_train_r.rename(columns={xsec_label: "fwd_ret_5d_positive"})
            xsec_m = xsec_cls.train(xsec_train_r, feature_cols=selected_features)
            xsec_auc = xsec_m.get("val_auc", 0)
            print(f"  ✓ Cross-sectional cls – AUC: {xsec_auc:.4f}")
            if xsec_auc > cls_m.get("val_auc", 0) + 0.001:
                print(f"    → Using cross-sectional model (AUC {xsec_auc:.4f} > {cls_m.get('val_auc', 0):.4f})")
                cls_model = xsec_cls
                cls_m = xsec_m

    # Try downside-aware "survived" target (P10 discriminator)
    # Note: survived model has higher AUC but selects conservatively (lower returns)
    # Only use if AUC is dramatically better AND for blending into quantile models
    surv_label = "fwd_survived_5d"
    if surv_label in train_data.columns and train_data[surv_label].notna().sum() > 200:
        surv_train = train_data.dropna(subset=[surv_label]).copy()
        surv_cls = MultiModelEnsemble(horizon=5, task="classification")
        surv_train_r = surv_train.drop(columns=["fwd_ret_5d_positive"], errors="ignore")
        surv_train_r = surv_train_r.rename(columns={surv_label: "fwd_ret_5d_positive"})
        surv_m = surv_cls.train(surv_train_r, feature_cols=selected_features)
        surv_auc = surv_m.get("val_auc", 0)
        print(f"  ✓ Downside-aware cls (survived) – AUC: {surv_auc:.4f}")
        # Don't switch to survived model — it's too conservative for return maximization
        # Instead, keep it for potential blending or P10 filtering enhancement

    # --- Multi-model regression ---
    print("  Training multi-model regressor...")
    reg_model = MultiModelEnsemble(horizon=5, task="regression")
    reg_m = reg_model.train(train_data, feature_cols=selected_features)
    print(f"  ✓ Regression – RMSE: {reg_m.get('val_rmse', 0):.4f} "
          f"(models: {reg_m.get('models', '?')})")

    # Quantile models (LightGBM only for speed)
    q_models = {}
    for alpha, label in [(0.1, "p10"), (0.5, "p50"), (0.9, "p90")]:
        qm = TabularModel(
            horizon=5, task="quantile", quantile_alpha=alpha,
            use_feature_selection=False,
            use_hyperparam_tuning=False,
            use_stacking=False,
            use_calibration=False,
        )
        qm.feature_cols = selected_features
        qm.train(train_data, feature_cols=selected_features)
        q_models[label] = qm
    print(f"  ✓ Quantile models (p10, p50, p90)")

    # --- LambdaRank model (cross-sectional stock ranking) ---
    # Addresses clustered probabilities problem: ranking objective provides
    # better differentiation than classification when P(ret>0) are similar
    import lightgbm as lgb
    rank_model = None
    try:
        rank_train = train_data.dropna(subset=["fwd_ret_5d"]).copy()
        rank_train["date_group"] = rank_train["date"].astype(str)
        # Create ranking target: discretized percentile rank (0-9) within each date
        rank_train["rank_target"] = rank_train.groupby("date_group")["fwd_ret_5d"].transform(
            lambda x: pd.qcut(x, q=10, labels=False, duplicates="drop")
        ).fillna(0).astype(int)
        # Query sizes (number of stocks per date)
        group_sizes = rank_train.groupby("date_group").size().values
        X_rank = rank_train[selected_features].values
        y_rank = rank_train["rank_target"].values

        rank_ds = lgb.Dataset(X_rank, label=y_rank, group=group_sizes,
                              feature_name=selected_features)
        rank_params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [4],
            "num_leaves": 31,
            "learning_rate": 0.05,
            "min_data_in_leaf": 20,
            "verbose": -1,
            "seed": 42,
        }
        rank_lgb = lgb.train(rank_params, rank_ds, num_boost_round=200)
        rank_model = (rank_lgb, selected_features)
        print(f"  ✓ LambdaRank model (cross-sectional ranking)")
    except Exception as e:
        print(f"  ⚠ LambdaRank failed: {e}")

    model_dir = cfg.data_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    cls_model.save(model_dir / "smallcap_cls_5d.pkl")
    reg_model.save(model_dir / "smallcap_reg_5d.pkl")
    for label, qm in q_models.items():
        qm.save(model_dir / f"smallcap_q{label}_5d.pkl")

    return cls_model, cls_m, reg_model, reg_m, q_models, predict_data, rank_model


def _build_dividend_features(divs_df: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Build dividend-related features from raw dividend data.

    Features: has_dividend, dividend_yield_annualized, days_since_ex_date.
    """
    if divs_df.empty:
        return pd.DataFrame()

    divs = divs_df.copy()
    divs["ex_date"] = pd.to_datetime(divs["ex_date"])
    divs["amount"] = pd.to_numeric(divs["amount"], errors="coerce")
    divs = divs.dropna(subset=["ex_date", "amount"])

    # Annual dividend per ticker (sum of last year's dividends)
    annual_div = divs.groupby("ticker")["amount"].sum().reset_index()
    annual_div.columns = ["ticker", "annual_div"]

    # Get all (ticker, date) combos from OHLCV
    ohlcv_slim = ohlcv[["ticker", "date", "close"]].copy()
    ohlcv_slim["date"] = pd.to_datetime(ohlcv_slim["date"])
    result = ohlcv_slim.merge(annual_div, on="ticker", how="left")
    result["has_dividend"] = (result["annual_div"].fillna(0) > 0).astype(int)
    result["dividend_yield"] = result["annual_div"].fillna(0) / result["close"].clip(lower=0.01)
    result["dividend_yield"] = result["dividend_yield"].clip(upper=0.30)  # cap at 30%

    # Days since last ex-dividend date
    divs_sorted = divs.sort_values(["ticker", "ex_date"])
    for ticker in divs_sorted["ticker"].unique():
        t_divs = divs_sorted[divs_sorted["ticker"] == ticker]
        mask = result["ticker"] == ticker
        t_dates = result.loc[mask, "date"]
        days_since = pd.Series(np.nan, index=t_dates.index)
        for _, div_row in t_divs.iterrows():
            ex = div_row["ex_date"]
            after_mask = t_dates >= ex
            candidate = (t_dates[after_mask] - ex).dt.days
            # Only overwrite if closer
            for idx, val in candidate.items():
                if pd.isna(days_since.loc[idx]) or val < days_since.loc[idx]:
                    days_since.loc[idx] = val
        result.loc[mask, "days_since_ex_div"] = days_since

    result["days_since_ex_div"] = result["days_since_ex_div"].fillna(999)
    return result[["ticker", "date", "has_dividend", "dividend_yield", "days_since_ex_div"]]


def _build_split_features(splits_df: pd.DataFrame) -> pd.DataFrame:
    """Build split-related features.

    Features: had_reverse_split (binary, forward-filled per ticker after event).
    """
    if splits_df.empty:
        return pd.DataFrame()

    splits = splits_df.copy()
    splits["date"] = pd.to_datetime(splits["date"])
    splits["is_reverse"] = (splits["split_from"] > splits["split_to"]).astype(int)

    reverse = splits[splits["is_reverse"] == 1][["ticker", "date"]].copy()
    reverse["had_reverse_split"] = 1
    return reverse[["ticker", "date", "had_reverse_split"]]


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged momentum and mean-reversion features.

    These capture autoregressive patterns the model can exploit.
    """
    df = df.sort_values(["ticker", "date"]).copy()
    grouped = df.groupby("ticker")

    # Lagged returns — model can learn momentum/reversion from recent history
    for col in ["ret_1d", "ret_5d", "ret_20d"]:
        if col in df.columns:
            for lag in [1, 5, 10]:
                df[f"{col}_lag{lag}"] = grouped[col].shift(lag)

    # Lagged volume ratio
    if "volume_ratio" in df.columns:
        for lag in [1, 5]:
            df[f"volume_ratio_lag{lag}"] = grouped["volume_ratio"].shift(lag)

    # Lagged RSI
    if "rsi_14" in df.columns:
        for lag in [1, 5]:
            df[f"rsi_14_lag{lag}"] = grouped["rsi_14"].shift(lag)

    # Momentum acceleration (2nd derivative)
    if "ret_5d" in df.columns:
        df["momentum_accel_5d"] = grouped["ret_5d"].diff()
    if "ret_20d" in df.columns:
        df["momentum_accel_20d"] = grouped["ret_20d"].diff()

    # Return dispersion across recent windows (vol of vol proxy)
    if all(c in df.columns for c in ["ret_1d", "ret_5d", "ret_20d"]):
        df["return_dispersion"] = df[["ret_1d", "ret_5d", "ret_20d"]].std(axis=1)

    # Mean reversion signal: distance from rolling mean
    if "close" in df.columns:
        for w in [10, 30]:
            sma = grouped["close"].transform(lambda x: x.rolling(w, min_periods=w//2).mean())
            df[f"mean_rev_{w}d"] = (df["close"] - sma) / sma

    # Volume trend
    if "volume" in df.columns:
        vol_sma_5 = grouped["volume"].transform(lambda x: x.rolling(5, min_periods=3).mean())
        vol_sma_20 = grouped["volume"].transform(lambda x: x.rolling(20, min_periods=10).mean())
        df["volume_trend"] = vol_sma_5 / vol_sma_20.clip(lower=1)

    return df


def generate_signals(cls_model, reg_model, q_models, predict_data, top_n,
                     features_df=None, ohlcv=None, rank_model=None):
    """Generate signals using probability ranking + ATR-adaptive stops.

    Improvements from quant research analysis:
    - ATR-ADAPTIVE STOPS: trailing stops scaled to each stock's volatility
    - Optional LambdaRank blending for better cross-sectional differentiation
    """
    BUY_PER_DATE = min(top_n, 4)
    REBALANCE_FREQ = 20
    MAX_PER_TICKER = 15
    P10_RANK_WEIGHT = 0.4
    PROB_RANK_WEIGHT = 0.6

    pred_dates = sorted(predict_data["date"].unique())
    rebal_dates = pred_dates[::REBALANCE_FREQ]

    all_signals = []
    for dt in rebal_dates:
        day_df = predict_data[predict_data["date"] == dt].copy()
        if day_df.empty:
            continue

        cls_preds = cls_model.predict_df(day_df)
        reg_preds = reg_model.predict_df(day_df)
        q_preds = {label: qm.predict_df(day_df, col_name=f"pred_q{label}_5d")
                   for label, qm in q_models.items()}

        merged = day_df.copy()
        merged = merged.merge(cls_preds, on=["ticker", "date"], how="left")
        merged = merged.merge(reg_preds, on=["ticker", "date"], how="left")
        for label, qdf in q_preds.items():
            merged = merged.merge(qdf, on=["ticker", "date"], how="left")

        merged["calibrated_prob"] = merged["pred_classification_5d"].fillna(0.5)
        merged["expected_return"] = merged["pred_regression_5d"].fillna(0.0)

        scored_rows = []
        for _, row in merged.iterrows():
            prob = float(row["calibrated_prob"])
            exp_ret = float(row.get("pred_regression_5d", 0.0))
            vol = float(row.get("realized_vol_20d", 0.3))
            atr_pct = float(row.get("atr_pct_20d", vol / np.sqrt(252) * 2))

            p10 = float(row.get("pred_qp10_5d", exp_ret - 2 * vol / 16 * 2.24))
            p50 = float(row.get("pred_qp50_5d", exp_ret))
            p90 = float(row.get("pred_qp90_5d", exp_ret + 2 * vol / 16 * 2.24))

            scored_rows.append({
                "row": row, "prob": prob, "exp_ret": exp_ret,
                "vol": vol, "atr_pct": atr_pct,
                "p10": p10, "p50": p50, "p90": p90,
            })

        n = len(scored_rows)
        if n == 0:
            continue

        # RANK by PROBABILITY only — proven to identify winners (TALK #1 → +47%)
        # LambdaRank kept for analysis but NOT used for ranking (hurts return)
        # No P10 floor filter: volatile stocks (RCAT +42.9%) are alpha sources
        scored_rows.sort(key=lambda x: x["prob"], reverse=True)
        for i, sr in enumerate(scored_rows):
            sr["combined_rank"] = i

        buys = scored_rows[:BUY_PER_DATE]

        # FIXED SIZING: proven optimal in baseline (50/20/17/13)
        FIXED_WEIGHTS = [0.50, 0.20, 0.17, 0.13]

        # Median ATR for adaptive stops
        median_atr = np.median([sr["atr_pct"] for sr in buys]) if buys else 0.03

        records = []
        buy_count = 0

        for sr in scored_rows:
            row = sr["row"]
            prob, exp_ret, vol = sr["prob"], sr["exp_ret"], sr["vol"]
            atr_pct = sr["atr_pct"]
            p10, p50, p90 = sr["p10"], sr["p50"], sr["p90"]
            ticker = str(row.get("ticker", ""))
            dt_str = str(row.get("date", ""))

            if buy_count < BUY_PER_DATE:
                recommendation = "BUY"
                position_size = FIXED_WEIGHTS[buy_count] if buy_count < len(FIXED_WEIGHTS) else 0.10
                buy_count += 1
            else:
                recommendation = "HOLD"
                position_size = 0.0

            take_profit = 1.0  # disabled

            # ATR-ADAPTIVE trailing stop
            if median_atr > 0 and recommendation == "BUY":
                atr_ratio = atr_pct / median_atr
                adaptive_trail = np.clip(0.16 * atr_ratio, 0.10, 0.16)
            else:
                adaptive_trail = 0.16

            records.append({
                "ticker": ticker,
                "date": dt_str,
                "recommendation": recommendation,
                "ensemble_score": prob,
                "calibrated_prob": prob,
                "expected_return": exp_ret,
                "expected_loss": abs(p10) * (1 - prob),
                "ev_score": prob * max(p90, 0),
                "p10": p10, "p50": p50, "p90": p90,
                "reward_risk": max(p90, 0) / max(abs(p10), 0.01),
                "liquidity_score": float(row.get("liquidity_score_raw", 0.5)),
                "position_size_pct": position_size,
                "stop_loss_pct": adaptive_trail,
                "take_profit_pct": take_profit,
                "trailing_stop_pct": adaptive_trail,
                "rejection_reasons": "",
            })

        if records:
            all_signals.append(pd.DataFrame(records))

    signals_df = pd.concat(all_signals, ignore_index=True) if all_signals else pd.DataFrame()

    # Limit per-ticker BUY signals
    if not signals_df.empty:
        buy_mask = signals_df["recommendation"] == "BUY"
        buy_signals = signals_df[buy_mask].copy()
        if not buy_signals.empty:
            buy_signals["_rank"] = buy_signals.groupby("ticker").cumcount()
            over_limit = buy_signals["_rank"] >= MAX_PER_TICKER
            idx_to_demote = buy_signals.index[over_limit]
            signals_df.loc[idx_to_demote, "recommendation"] = "HOLD"
            demoted = over_limit.sum()
            if demoted > 0:
                print(f"  ✓ Diversity: demoted {demoted} duplicate-ticker BUY → HOLD")

    n_buys = len(signals_df[signals_df["recommendation"] == "BUY"]) if not signals_df.empty else 0
    n_shorts = len(signals_df[signals_df["recommendation"] == "SHORT"]) if not signals_df.empty else 0
    print(f"  Validation signals: {len(signals_df)}")
    print(f"    BUY: {n_buys} | SHORT: {n_shorts} | Rebalance dates: {len(rebal_dates)}")
    if not signals_df.empty and n_buys > 0:
        buy_df = signals_df[signals_df["recommendation"] == "BUY"]
        print(f"    Avg BUY position size: {buy_df['position_size_pct'].mean():.1%}")
        print(f"    Top picks: {buy_df.groupby('ticker').size().sort_values(ascending=False).head(8).to_dict()}")

    return signals_df


def walk_forward_cv(features, ohlcv, cfg, n_folds=5, purge_days=10, min_train_rows=500):
    """Combinatorial Purged Walk-Forward Cross-Validation.

    Replaces the compromised single holdout with a rigorous multi-fold
    temporal validation. Each fold:
      1. Train on expanding window [start, split)
      2. Purge gap of `purge_days` to prevent label leakage
      3. Test on [split + purge, split + step)
      4. Generate signals & backtest on the test period

    All OOS trades are aggregated for combined statistical analysis.
    This provides more trades and covers more market regimes than a single holdout.
    """
    from app.models.multi_model import MultiModelEnsemble
    from app.models.tabular import TabularModel, _auto_feature_cols
    from app.models.feature_selection import select_features
    from app.backtest import BacktestConfig, Backtester

    features = features.copy()
    features["date"] = pd.to_datetime(features["date"])
    features = features.sort_values("date")

    all_dates = sorted(features["date"].unique())
    n_dates = len(all_dates)

    # Minimum training: 60% of data, rest split into n_folds test windows
    min_train_pct = 0.50
    min_train_idx = int(n_dates * min_train_pct)
    remaining = n_dates - min_train_idx
    step = max(remaining // n_folds, 20)  # at least 20 trading days per fold

    print(f"\n  ── WALK-FORWARD CROSS-VALIDATION ({n_folds} folds) ──")
    print(f"  Total dates: {n_dates} ({all_dates[0].date()} → {all_dates[-1].date()})")
    print(f"  Min train: {min_train_idx} dates | Step: {step} dates | Purge: {purge_days} days")

    fold_results = []
    all_oos_trades = []
    all_oos_daily_returns = []
    all_oos_portfolio_values = []

    for fold in range(n_folds):
        split_idx = min_train_idx + fold * step
        test_start_idx = min(split_idx + purge_days, n_dates - 1)
        test_end_idx = min(split_idx + step + purge_days, n_dates)

        if test_start_idx >= n_dates or test_end_idx - test_start_idx < 10:
            break

        train_end_date = all_dates[split_idx]
        test_start_date = all_dates[test_start_idx]
        test_end_date = all_dates[min(test_end_idx - 1, n_dates - 1)]

        train_data = features[features["date"] < train_end_date].dropna(
            subset=["fwd_ret_5d_positive", "fwd_ret_5d"]
        )
        test_data = features[
            (features["date"] >= test_start_date) & (features["date"] <= test_end_date)
        ].copy()

        if len(train_data) < min_train_rows or len(test_data) < 10:
            print(f"  Fold {fold+1}: skipped (train={len(train_data)}, test={len(test_data)})")
            continue

        print(f"\n  ── Fold {fold+1}/{n_folds} ──")
        print(f"    Train: {train_data['date'].min().date()} → {train_end_date.date()} ({len(train_data):,} rows)")
        print(f"    Purge: {purge_days} days")
        print(f"    Test:  {test_start_date.date()} → {test_end_date.date()} ({len(test_data):,} rows)")

        # Feature selection on this fold's training data
        raw_features = _auto_feature_cols(train_data)
        shap_target = "fwd_ret_5d_risk_adj_positive" if "fwd_ret_5d_risk_adj_positive" in train_data.columns else "fwd_ret_5d_positive"
        try:
            selected = select_features(
                train_data, raw_features, shap_target,
                task="classification", top_k=60, method="shap",
            )
        except Exception:
            selected = raw_features[:60]

        # Train models on this fold
        cls_model = MultiModelEnsemble(horizon=5, task="classification")
        ra_label = "fwd_ret_5d_risk_adj_positive"
        if ra_label in train_data.columns:
            ra_train = train_data.dropna(subset=[ra_label]).copy()
            ra_train = ra_train.drop(columns=["fwd_ret_5d_positive"], errors="ignore")
            ra_train = ra_train.rename(columns={ra_label: "fwd_ret_5d_positive"})
            cls_m = cls_model.train(ra_train, feature_cols=selected)
        else:
            cls_m = cls_model.train(train_data, feature_cols=selected)
        auc = cls_m.get("val_auc", 0)

        reg_model = MultiModelEnsemble(horizon=5, task="regression")
        reg_model.train(train_data, feature_cols=selected)

        q_models = {}
        for alpha, label in [(0.1, "p10"), (0.5, "p50"), (0.9, "p90")]:
            qm = TabularModel(
                horizon=5, task="quantile", quantile_alpha=alpha,
                use_feature_selection=False, use_hyperparam_tuning=False,
                use_stacking=False, use_calibration=False,
            )
            qm.feature_cols = selected
            qm.train(train_data, feature_cols=selected)
            q_models[label] = qm

        # Generate signals on test data
        signals = generate_signals(
            cls_model, reg_model, q_models, test_data, 15,
            features_df=features, ohlcv=ohlcv,
        )

        if signals.empty:
            print(f"    ⚠ No signals generated for fold {fold+1}")
            continue

        n_buys = len(signals[signals["recommendation"] == "BUY"])
        print(f"    Signals: {len(signals)} (BUY: {n_buys})")

        # Backtest this fold
        bt_config = BacktestConfig(
            start_date=str(test_start_date.date()),
            end_date=str(test_end_date.date()),
            initial_capital=1000,
            max_positions=4,
            rebalance_frequency="weekly",
            commission_bps=cfg.commission_bps,
            slippage_bps=cfg.slippage_bps,
            holding_period_trading_days=44,
            use_stop_loss=False,
            use_take_profit=False,
            trailing_stop_pct=0.16,
        )
        bt = Backtester(bt_config)
        result = bt.run(signals, ohlcv)
        m = result.metrics

        fold_results.append({
            "fold": fold + 1,
            "train_end": str(train_end_date.date()),
            "test_start": str(test_start_date.date()),
            "test_end": str(test_end_date.date()),
            "train_rows": len(train_data),
            "test_rows": len(test_data),
            "auc": auc,
            "total_return": m.get("total_return", 0),
            "sharpe": m.get("sharpe_ratio", 0),
            "max_dd": m.get("max_drawdown", 0),
            "n_trades": m.get("n_trades", 0),
        })

        # Collect OOS data for aggregation
        if not result.trades.empty:
            fold_trades = result.trades.copy()
            fold_trades["fold"] = fold + 1
            all_oos_trades.append(fold_trades)
        if not result.daily_returns.empty:
            all_oos_daily_returns.append(result.daily_returns)

        ret_str = f"{m.get('total_return', 0):+.2%}"
        sr_str = f"{m.get('sharpe_ratio', 0):.2f}"
        dd_str = f"{m.get('max_drawdown', 0):.2%}"
        print(f"    Result: Return={ret_str}  Sharpe={sr_str}  MaxDD={dd_str}  Trades={m.get('n_trades', 0)}")

    if not fold_results:
        print("\n  ⚠ No valid folds — insufficient data for walk-forward CV")
        return

    # ── Aggregate results ──
    print(f"\n  {'═' * 60}")
    print(f"  WALK-FORWARD CV SUMMARY ({len(fold_results)} folds)")
    print(f"  {'═' * 60}")

    # Per-fold table
    print(f"\n  {'Fold':<6} {'Test Period':<25} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>7} {'AUC':>6}")
    print(f"  {'─'*6} {'─'*25} {'─'*10} {'─'*8} {'─'*8} {'─'*7} {'─'*6}")
    for fr in fold_results:
        print(f"  {fr['fold']:<6} {fr['test_start']} → {fr['test_end'][:5]}  "
              f"{fr['total_return']:>+9.2%} {fr['sharpe']:>8.2f} {fr['max_dd']:>7.2%} "
              f"{fr['n_trades']:>7} {fr['auc']:>5.3f}")

    # Aggregated statistics
    returns = [fr["total_return"] for fr in fold_results]
    sharpes = [fr["sharpe"] for fr in fold_results]
    trades = [fr["n_trades"] for fr in fold_results]
    total_trades = sum(trades)

    avg_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if len(returns) > 1 else 0
    med_ret = np.median(returns)
    avg_sr = np.mean(sharpes)
    pct_positive = sum(1 for r in returns if r > 0) / len(returns) * 100

    print(f"\n  ── Aggregate ──")
    print(f"  Avg Return:       {avg_ret:+.2%} ± {std_ret:.2%}")
    print(f"  Median Return:    {med_ret:+.2%}")
    print(f"  Avg Sharpe:       {avg_sr:.2f}")
    print(f"  % Positive Folds: {pct_positive:.0f}%")
    print(f"  Total OOS trades: {total_trades}")

    # Combined OOS trade analysis
    if all_oos_trades:
        combined_trades = pd.concat(all_oos_trades, ignore_index=True)
        buys = combined_trades[combined_trades["action"] == "BUY"]
        sells = combined_trades[combined_trades["action"] == "SELL"]

        if not buys.empty and not sells.empty:
            trade_returns = []
            for _, buy in buys.iterrows():
                sell = sells[
                    (sells["ticker"] == buy["ticker"]) & (sells["date"] > buy["date"])
                ].head(1)
                if not sell.empty:
                    s = sell.iloc[0]
                    pnl = (s["price"] - buy["price"]) / buy["price"]
                    trade_returns.append(pnl)

            if trade_returns:
                tr_arr = np.array(trade_returns)
                n = len(tr_arr)
                mean_r = np.mean(tr_arr)
                std_r = np.std(tr_arr, ddof=1) if n > 1 else 0
                se = std_r / np.sqrt(n) if n > 0 else 0
                win_rate = np.mean(tr_arr > 0) * 100

                # t-test: is the mean return significantly > 0?
                from scipy.stats import t as t_dist
                t_stat = mean_r / se if se > 0 else 0
                p_value = 1 - t_dist.cdf(t_stat, df=max(n - 1, 1))

                print(f"\n  ── Combined OOS Trade Analysis ──")
                print(f"  Total round-trips:   {n}")
                print(f"  Mean return/trade:   {mean_r:+.2%} ± {std_r:.2%}")
                print(f"  Win rate:            {win_rate:.0f}%")
                print(f"  t-statistic:         {t_stat:.2f}")
                print(f"  p-value (H₀: μ≤0):  {p_value:.4f}")
                print(f"  Significant (p<.05): {'✅ YES' if p_value < 0.05 else '❌ NO'}")
                print(f"  95% CI:              [{mean_r - 1.96*se:+.2%}, {mean_r + 1.96*se:+.2%}]")

    # Combined daily returns for aggregate Sharpe
    if all_oos_daily_returns:
        combined_daily = pd.concat(all_oos_daily_returns)
        combined_daily = combined_daily.sort_index()
        # Remove duplicate dates (overlapping folds), take mean
        combined_daily = combined_daily.groupby(combined_daily.index).mean()
        if len(combined_daily) > 5:
            agg_sr = combined_daily.mean() / max(combined_daily.std(), 1e-10) * np.sqrt(252)
            print(f"\n  Aggregate OOS Sharpe: {agg_sr:.2f}")

    print()


def _compute_robustness_metrics(result, n_trials=24):
    """Compute Deflated Sharpe Ratio and Monte Carlo confidence intervals.

    Deflated Sharpe Ratio (López de Prado, 2014):
    Adjusts for multiple hypothesis testing — what is the probability that
    the observed Sharpe is merely the best out of n_trials random strategies?

    Monte Carlo: Bootstrap daily returns to estimate return confidence intervals.
    """
    from scipy.stats import norm

    pv = pd.DataFrame(result.portfolio_values)
    if len(pv) < 5:
        return

    daily_rets = pv["value"].pct_change().dropna().values
    T = len(daily_rets)
    sr = np.mean(daily_rets) / max(np.std(daily_rets, ddof=1), 1e-10) * np.sqrt(252)
    skew = float(pd.Series(daily_rets).skew())
    kurt = float(pd.Series(daily_rets).kurtosis())

    # Standard error of Sharpe (Lo, 2002)
    se_sr = np.sqrt((1 + 0.5 * sr**2 - skew * sr + (kurt / 4) * sr**2) / T)

    # Expected maximum Sharpe under null (Euler-Mascheroni approximation)
    # E[max(SR)] ≈ sqrt(2 * ln(n_trials)) - (γ + ln(π/2)) / (2 * sqrt(2 * ln(n_trials)))
    gamma_em = 0.5772
    if n_trials > 1:
        v = np.sqrt(2 * np.log(n_trials))
        sr0 = v - (gamma_em + np.log(np.pi / 2)) / (2 * v)
    else:
        sr0 = 0.0

    # Deflated Sharpe Ratio: P(SR > SR₀)
    dsr = norm.cdf((sr - sr0) / max(se_sr, 1e-10))

    # Monte Carlo bootstrap (1000 trials)
    rng = np.random.default_rng(42)
    mc_returns = []
    for _ in range(1000):
        boot_rets = rng.choice(daily_rets, size=T, replace=True)
        mc_returns.append(float(np.prod(1 + boot_rets) - 1))
    mc_returns = np.array(mc_returns)
    mc_p5 = np.percentile(mc_returns, 5) * 100
    mc_p25 = np.percentile(mc_returns, 25) * 100
    mc_median = np.median(mc_returns) * 100
    mc_p75 = np.percentile(mc_returns, 75) * 100
    mc_p95 = np.percentile(mc_returns, 95) * 100
    prob_positive = np.mean(mc_returns > 0) * 100

    print(f"\n  ── ROBUSTNESS METRICS ──")
    print(f"  Deflated Sharpe Ratio:  {dsr:.4f}  (p-value that SR is real, not luck)")
    print(f"    Observed SR:          {sr:.2f}")
    print(f"    Expected SR₀ (null):  {sr0:.2f}  (best of {n_trials} random trials)")
    print(f"    SR standard error:    {se_sr:.2f}")
    print(f"    Daily skew:           {skew:.2f},  kurtosis: {kurt:.2f}")
    print(f"  Monte Carlo (1000 bootstraps):")
    print(f"    P5/P25/Median/P75/P95: {mc_p5:+.1f}% / {mc_p25:+.1f}% / {mc_median:+.1f}% / {mc_p75:+.1f}% / {mc_p95:+.1f}%")
    print(f"    P(positive return):    {prob_positive:.1f}%")
    print()


def display_results(verified_tickers, cls_model, cls_m, reg_m, signals_df,
                    buys_enriched, result, cfg, args, n_tickers, output_path):
    """Display rich tables with results."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        console = Console()
    except ImportError:
        print("Install 'rich' for formatted output: pip install rich")
        return

    # Universe summary
    uni_table = Table(title="[bold]Verified Small-Cap Universe[/bold]", show_lines=False)
    uni_table.add_column("Ticker", style="cyan", width=8)
    uni_table.add_column("Name", width=35)
    uni_table.add_column("Market Cap", justify="right", style="green", width=12)
    uni_table.add_column("SIC", width=6)
    uni_table.add_column("Exchange", width=8)
    for v in sorted(verified_tickers, key=lambda x: x["market_cap"], reverse=True):
        uni_table.add_row(
            v["ticker"], (v["name"] or "")[:35],
            f"${v['market_cap']/1e6:,.0f}M", str(v["sic_code"]),
            v["exchange"][:8] if v["exchange"] else "",
        )
    console.print(uni_table)
    console.print()

    # Feature importance
    fi = cls_model.feature_importance()
    fi_table = Table(title="Top 10 Features", show_lines=False)
    fi_table.add_column("Feature", style="cyan")
    fi_table.add_column("Importance %", justify="right", style="green")
    for _, row in fi.head(10).iterrows():
        fi_table.add_row(str(row["feature"]), f"{row['importance_pct']:.2%}")
    console.print(fi_table)
    console.print()

    # BUY recommendations
    buys = signals_df[signals_df["recommendation"] == "BUY"]
    unique_buys = buys_enriched.sort_values("date").drop_duplicates("ticker", keep="first")
    if not unique_buys.empty:
        buy_table = Table(title="[bold green]★ Small-Cap BUY Recommendations 2026 ★[/bold green]", show_lines=True)
        buy_table.add_column("Ticker", style="bold cyan", width=8)
        buy_table.add_column("Signal Date", width=12)
        buy_table.add_column("Price $", justify="right", width=10)
        buy_table.add_column("Mkt Cap", justify="right", width=12)
        buy_table.add_column("P(up 5d)", justify="right", style="green", width=10)
        buy_table.add_column("E[ret]", justify="right", width=10)
        buy_table.add_column("Stop $", justify="right", style="red", width=10)
        buy_table.add_column("Target $", justify="right", width=10)
        buy_table.add_column("Size %", justify="right", style="yellow", width=8)

        for _, r in unique_buys.iterrows():
            dt_str = pd.Timestamp(r["date"]).strftime("%Y-%m-%d")
            price = r.get("close_price", 0)
            if pd.isna(price) or price <= 0:
                continue
            stop_price = price * (1 - abs(r["stop_loss_pct"]))
            target_price = price * (1 + r["p50"])
            mcap_info = next((v for v in verified_tickers if v["ticker"] == r["ticker"]), None)
            mcap_str = f"${mcap_info['market_cap']/1e6:,.0f}M" if mcap_info else "N/A"

            buy_table.add_row(
                str(r["ticker"]), dt_str, f"${price:.2f}", mcap_str,
                f"{r['calibrated_prob']:.1%}", f"{r['expected_return']:+.2%}",
                f"${stop_price:.2f}", f"${target_price:.2f}",
                f"{r['position_size_pct']:.2%}",
            )
        console.print(buy_table)
    console.print()

    # Trade history
    if not result.trades.empty:
        trades = result.trades.copy()
        trades_buy = trades[trades["action"] == "BUY"].sort_values("date")
        trades_sell = trades[trades["action"] == "SELL"].sort_values("date")

        trade_pairs = []
        for _, buy_row in trades_buy.iterrows():
            sell = trades_sell[
                (trades_sell["ticker"] == buy_row["ticker"]) &
                (trades_sell["date"] > buy_row["date"])
            ].head(1)
            if not sell.empty:
                s = sell.iloc[0]
                pnl = (s["price"] - buy_row["price"]) / buy_row["price"]
                trade_pairs.append({
                    "ticker": buy_row["ticker"],
                    "buy_date": buy_row["date"].strftime("%Y-%m-%d"),
                    "buy_price": buy_row["price"],
                    "sell_date": s["date"].strftime("%Y-%m-%d"),
                    "sell_price": s["price"],
                    "return": pnl,
                })

        if trade_pairs:
            tt = Table(title="[bold]Executed Trades 2026[/bold]", show_lines=True)
            tt.add_column("Ticker", style="cyan", width=8)
            tt.add_column("Buy", width=12)
            tt.add_column("Buy $", justify="right", width=10)
            tt.add_column("Sell", width=12)
            tt.add_column("Sell $", justify="right", width=10)
            tt.add_column("P/L", justify="right", width=10)

            for t in trade_pairs[:30]:
                st = "green" if t["return"] >= 0 else "red"
                tt.add_row(
                    t["ticker"], t["buy_date"], f"${t['buy_price']:.2f}",
                    t["sell_date"], f"${t['sell_price']:.2f}",
                    f"[{st}]{t['return']:+.2%}[/{st}]",
                )

            won = sum(1 for t in trade_pairs if t["return"] > 0)
            total = len(trade_pairs)
            console.print(tt)
            if total > 0:
                console.print(f"\n  Win rate: {won}/{total} ({won/total:.0%})")
            console.print()

    # Open positions
    if result.open_positions:
        op_table = Table(title="[bold yellow]📊 Open Positions (not yet closed)[/bold yellow]", show_lines=True)
        op_table.add_column("Ticker", style="bold cyan", width=8)
        op_table.add_column("Side", width=6)
        op_table.add_column("Entry", width=12)
        op_table.add_column("Entry $", justify="right", width=10)
        op_table.add_column("Current $", justify="right", width=10)
        op_table.add_column("P/L", justify="right", width=10)
        op_table.add_column("Shares", justify="right", width=8)
        op_table.add_column("Days", justify="right", width=6)
        op_table.add_column("Remaining", justify="right", width=10)
        op_table.add_column("Trail Stop $", justify="right", style="red", width=12)

        for op in result.open_positions:
            st = "green" if op["pnl_pct"] >= 0 else "red"
            op_table.add_row(
                op["ticker"], op["side"], op["entry_date"],
                f"${op['entry_price']:.2f}", f"${op['current_price']:.2f}",
                f"[{st}]{op['pnl_pct']:+.1f}%[/{st}]",
                str(op["shares"]),
                str(op["days_held"]),
                f"{op['days_remaining']}d",
                f"${op['trail_trigger_price']:.2f} ({op['trailing_stop_pct']:.0f}%)",
            )
        console.print(op_table)
        console.print()

    # Summary
    m = result.metrics
    buys_total = len(signals_df[signals_df["recommendation"] == "BUY"])
    summary = f"""
[bold]SCAI Small-Cap Pipeline – 2026[/bold]

  Data source:       Massive API (real market data)
  Market cap filter:  ${cfg.min_market_cap/1e6:.0f}M – ${cfg.max_market_cap/1e6:.0f}M
  Training range:    {args.train_start} → {args.predict_from}
  Validation range:  {args.predict_from} → {args.holdout_from}
  Holdout range:     {args.holdout_from} → {args.predict_to}  {'(EVALUATED)' if args.eval_holdout else '(RESERVED)'}
  Universe:          {n_tickers} verified small-cap stocks

  [bold]Enhancements:[/bold]
    Feature selection:   SHAP-based top-60 + correlation pruning
    Multi-model stack:   LightGBM + XGBoost + CatBoost → meta-learner
    Indicators:          Bollinger, Stochastic, ADX, Ichimoku, CCI, Williams
    Triple barrier:      TP/SL/timeout labels (López de Prado) [vectorized]
    Regime awareness:    Vol-regime threshold adjustment
    Portfolio mgmt:      Sector limits (30%), total exposure (90%)
    Dynamic thresholds:  Walk-forward Sharpe optimization
    Calibration:         Isotonic (classification)
    Microstructure:      VWAP dev, volume profile, Corwin-Schultz
    Sector features:     SIC→sector, sector-relative, rotation
    Risk-adj targets:    Return/vol, sector-relative labels
    ATR-adaptive stops:  Per-position trailing stops scaled to volatility
    LambdaRank:          Cross-sectional ranking model (informational)
    Robustness:          Deflated Sharpe ratio + Monte Carlo bootstrap

  Model AUC:         {cls_m.get('val_auc', 0):.4f}  (features: {cls_m.get('n_features', '?')})
  Model RMSE:        {reg_m.get('val_rmse', 0):.4f}

  Total signals:     {len(signals_df)}
  BUY signals:       {buys_total}
  Trades executed:   {m.get('n_trades', 0)}
  Total Return:      {m.get('total_return', 0):+.2%}
  Sharpe Ratio:      {m.get('sharpe_ratio', 0):.2f}
  Max Drawdown:      {m.get('max_drawdown', 0):.2%}

  Reports:           {output_path}
"""
    console.print(Panel(summary, title="Results", border_style="blue"))


def main() -> None:
    parser = argparse.ArgumentParser(description="SCAI Small-Cap Pipeline (Massive API)")
    parser.add_argument("--train-start", default="2020-01-01")
    parser.add_argument("--predict-from", default="2026-01-02")
    parser.add_argument("--holdout-from", default="2026-04-01",
                        help="Start of holdout period (never looked at until --eval-holdout)")
    parser.add_argument("--predict-to", default="2026-05-06")
    parser.add_argument("--eval-holdout", action="store_true",
                        help="Evaluate on the holdout period (use only for final evaluation)")
    parser.add_argument("--wf-cv", action="store_true",
                        help="Run walk-forward cross-validation (replaces compromised holdout)")
    parser.add_argument("--wf-folds", type=int, default=5,
                        help="Number of walk-forward CV folds (default: 5)")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--max-tickers", type=int, default=200,
                        help="Max tickers to include in universe (dynamic discovery from Polygon)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Reuse cached OHLCV data (skip steps 1-3)")
    parser.add_argument("--skip-features", action="store_true",
                        help="Reuse cached features parquet (skip step 4, avoids memory fragmentation)")
    parser.add_argument("--output", default="data/reports/smallcap_2026.html")
    args = parser.parse_args()

    cfg = get_settings()
    setup_logging("INFO")
    set_global_seed(cfg.seed)

    from app.data.store.parquet_store import ParquetStore
    store = ParquetStore()

    # Determine backtest end based on holdout
    if args.eval_holdout:
        bt_end = args.predict_to
        holdout_msg = f"  Holdout:         {args.holdout_from} → {args.predict_to} (INCLUDED)"
    else:
        bt_end = args.holdout_from
        holdout_msg = f"  Holdout:         {args.holdout_from} → {args.predict_to} (RESERVED — use --eval-holdout)"

    print()
    print("=" * 70)
    print("  SCAI – Small-Cap Pipeline (Massive API)")
    print(f"  Market cap filter: ${cfg.min_market_cap/1e6:.0f}M – ${cfg.max_market_cap/1e6:.0f}M")
    print(f"  Seed universe:   {len(RUSSELL_SEED)} Russell 2000-style candidates")
    print(f"  Max tickers:     {args.max_tickers}")
    print(f"  Training:        {args.train_start} → {args.predict_from}")
    print(f"  Validation:      {args.predict_from} → {args.holdout_from}")
    print(holdout_msg)
    print("=" * 70)
    print()

    if args.skip_download:
        print("  --skip-download: reusing cached data\n")
        ohlcv = store.read("ohlcv_smallcap")
        uni_df = store.read("smallcap_universe")
        verified_tickers = uni_df.to_dict("records")
        n_tickers = ohlcv["ticker"].nunique()
        print(f"  Cached: {len(ohlcv):,} rows, {n_tickers} tickers\n")
    else:
        # ═══════════════════════════════════════════════════
        # STEP 1: Discover universe
        # ═══════════════════════════════════════════════════
        # Re-use cached universe if < 7 days old (market caps don't change daily)
        cached_uni = store.read("smallcap_universe") if store.exists("smallcap_universe") else None
        universe_fresh = False
        if cached_uni is not None and "as_of_date" in cached_uni.columns:
            as_of = pd.Timestamp(cached_uni["as_of_date"].iloc[0])
            age_days = (pd.Timestamp(date.today()) - as_of).days
            # Reuse if fresh AND has enough tickers AND was built with v2 filters
            has_active_col = "active" in cached_uni.columns
            if age_days < 7 and len(cached_uni) >= args.max_tickers * 0.9 and has_active_col:
                universe_fresh = True

        from app.data.massive import MassiveClient, ReferenceAPI, AggregatesAPI
        client = MassiveClient(calls_per_minute=5)

        if universe_fresh:
            print(f"STEP 1/6 ▸ Universe cached ({age_days}d old, <7d) — reusing {len(cached_uni)} tickers")
            verified_tickers = cached_uni.to_dict("records")
        else:
            print(f"STEP 1/6 ▸ Building small-cap universe (dynamic discovery)...")
            print(f"  Seed tickers: {len(RUSSELL_SEED)} | Target: {args.max_tickers}")
            print(f"  ⏳ Verifying market caps via API (5 calls/min)...\n")

            ref = ReferenceAPI(client)
            verified_tickers = discover_universe(
                ref, cfg, max_tickers=args.max_tickers, store=store,
                existing_universe=cached_uni, train_start=args.train_start,
            )

            if len(verified_tickers) < 5:
                print("  ⚠ Too few verified small caps.")
                client.close()
                sys.exit(1)

            uni_df = pd.DataFrame(verified_tickers)
            uni_df["as_of_date"] = date.today().isoformat()
            store.write("smallcap_universe", uni_df)

        tickers_for_download = [v["ticker"] for v in verified_tickers]
        print()

        # ═══════════════════════════════════════════════════
        # STEP 2: Download OHLCV (incremental)
        # ═══════════════════════════════════════════════════
        n = len(tickers_for_download)
        existing_ohlcv = store.read("ohlcv_smallcap") if store.exists("ohlcv_smallcap") else None
        if existing_ohlcv is not None and not existing_ohlcv.empty:
            existing_ohlcv["date"] = pd.to_datetime(existing_ohlcv["date"])
            last_date = existing_ohlcv["date"].max().date()
            n_existing = existing_ohlcv["ticker"].nunique()
            print(f"STEP 2/6 ▸ Updating OHLCV for {n} tickers (incremental from {last_date})...")
            print(f"  📦 Existing: {len(existing_ohlcv):,} rows, {n_existing} tickers, up to {last_date}")
        else:
            print(f"STEP 2/6 ▸ Downloading OHLCV for {n} tickers (full history)...")
            print(f"  ⏳ Estimated: ~{n * 13 // 60} min")
        print()

        aggs = AggregatesAPI(client)
        ohlcv = download_ohlcv(aggs, tickers_for_download, args.train_start,
                               args.predict_to, existing_ohlcv=existing_ohlcv)

        if ohlcv.empty:
            print("  ✗ No data downloaded!")
            client.close()
            sys.exit(1)

        n_tickers = ohlcv["ticker"].nunique()
        print(f"\n  ✓ Total: {len(ohlcv):,} rows, {n_tickers} tickers")
        print(f"  ✓ Range: {ohlcv['date'].min().date()} → {ohlcv['date'].max().date()}\n")

        # ═══════════════════════════════════════════════════
        # STEP 2b: Post-OHLCV quality filter (P0+P1)
        # ═══════════════════════════════════════════════════
        ohlcv, verified_tickers = filter_universe_quality(ohlcv, verified_tickers, cfg)
        tickers_for_download = [v["ticker"] for v in verified_tickers]
        n_tickers = ohlcv["ticker"].nunique()
        print(f"  ✓ After quality filter: {len(ohlcv):,} rows, {n_tickers} tickers\n")

        store.write("ohlcv_smallcap", ohlcv)
        # Update universe with filtered list
        uni_df = pd.DataFrame(verified_tickers)
        uni_df["as_of_date"] = date.today().isoformat()
        store.write("smallcap_universe", uni_df)

        # ═══════════════════════════════════════════════════
        # STEP 3: Corporate actions + fundamentals + SPY
        # ═══════════════════════════════════════════════════
        # Splits/dividends/fundamentals change rarely (quarterly at most).
        # Only re-fetch if data is more than 7 days old.
        from app.data.massive import CorporateActionsAPI, FinancialsAPI
        import os

        def _parquet_age_days(domain: str) -> int | None:
            """Return age in days of a parquet file, or None if missing."""
            path = store._path(domain)
            if not path.exists():
                return None
            mtime = pd.Timestamp(os.path.getmtime(path), unit="s")
            return (pd.Timestamp.now() - mtime).days

        CORP_MAX_AGE = 7  # days before re-fetching corporate data

        # ── Splits & Dividends ──
        splits_age = _parquet_age_days("smallcap_splits")
        divs_age = _parquet_age_days("smallcap_dividends")
        existing_splits = store.read("smallcap_splits") if store.exists("smallcap_splits") else None
        existing_divs = store.read("smallcap_dividends") if store.exists("smallcap_dividends") else None

        need_corp = (splits_age is None or splits_age >= CORP_MAX_AGE
                     or divs_age is None or divs_age >= CORP_MAX_AGE)

        if need_corp:
            print("STEP 3/6 ▸ Downloading corporate actions (splits/dividends)...")
            ca = CorporateActionsAPI(client)
            all_splits, all_divs = download_corporate_actions(
                ca, tickers_for_download,
                existing_splits=existing_splits, existing_divs=existing_divs,
            )
            print(f"  ✓ New splits: {len(all_splits)} | New dividends: {len(all_divs)}")

            if all_splits:
                splits_df = pd.DataFrame(all_splits)
                if existing_splits is not None and not existing_splits.empty:
                    splits_df = pd.concat([existing_splits, splits_df], ignore_index=True)
                    splits_df = splits_df.drop_duplicates(subset=["ticker", "date"], keep="last")
                store.write("smallcap_splits", splits_df)
                for _, s in splits_df.iterrows():
                    if s["split_from"] > s["split_to"]:
                        print(f"    ⚠ REVERSE SPLIT: {s['ticker']} {s['split_from']}:{s['split_to']} on {s['date']}")
            if all_divs:
                divs_df = pd.DataFrame(all_divs)
                if existing_divs is not None and not existing_divs.empty:
                    divs_df = pd.concat([existing_divs, divs_df], ignore_index=True)
                    divs_df = divs_df.drop_duplicates(subset=["ticker", "ex_date"], keep="last")
                store.write("smallcap_dividends", divs_df)
        else:
            n_s = len(existing_splits) if existing_splits is not None else 0
            n_d = len(existing_divs) if existing_divs is not None else 0
            print(f"STEP 3/6 ▸ Corporate actions cached ({splits_age}d old) — "
                  f"{n_s} splits, {n_d} dividends ✓")

        # ── Fundamentals ──
        fund_age = _parquet_age_days("smallcap_fundamentals")
        existing_fund = store.read("smallcap_fundamentals") if store.exists("smallcap_fundamentals") else None

        if fund_age is None or fund_age >= CORP_MAX_AGE:
            print("  Downloading fundamentals...")
            fin_api = FinancialsAPI(client)
            fund_df = download_fundamentals(fin_api, tickers_for_download, existing_fund=existing_fund)
            if not fund_df.empty:
                if existing_fund is not None and not existing_fund.empty:
                    fund_df = pd.concat([existing_fund, fund_df], ignore_index=True)
                    fund_df = fund_df.drop_duplicates(
                        subset=["ticker", "filed", "concept"], keep="last"
                    )
                store.write("smallcap_fundamentals", fund_df)
                print(f"  ✓ Fundamentals: {len(fund_df)} records")
            elif existing_fund is not None:
                print(f"  ✓ Fundamentals: {len(existing_fund)} records (no new)")
        else:
            n_f = len(existing_fund) if existing_fund is not None else 0
            print(f"  Fundamentals cached ({fund_age}d old) — {n_f} records ✓")

        # ── SPY (incremental — daily) ──
        print("  Downloading SPY (incremental)...")
        existing_spy = store.read("smallcap_spy") if store.exists("smallcap_spy") else None
        spy_from = args.train_start
        need_spy_download = True
        if existing_spy is not None and not existing_spy.empty:
            existing_spy["date"] = pd.to_datetime(existing_spy["date"])
            spy_last = existing_spy["date"].max()
            if spy_last >= pd.Timestamp(args.predict_to) - pd.Timedelta(days=3):
                print(f"  ✓ SPY: already current ({spy_last.date()})")
                need_spy_download = False
            else:
                spy_from = (spy_last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        if need_spy_download:
            spy_bars = aggs.get_custom_bars("SPY", from_date=spy_from,
                                            to_date=args.predict_to, adjusted=True)
            if spy_bars and len(spy_bars) > 0:
                spy_rows = [{"date": pd.Timestamp(b.trading_date), "ticker": "SPY",
                             "open": b.open, "high": b.high, "low": b.low,
                             "close": b.close, "volume": b.volume} for b in spy_bars]
                spy_new = pd.DataFrame(spy_rows)
                if existing_spy is not None and not existing_spy.empty:
                    spy_df = pd.concat([existing_spy, spy_new], ignore_index=True)
                    spy_df = spy_df.drop_duplicates(subset=["date"], keep="last")
                else:
                    spy_df = spy_new
                store.write("smallcap_spy", spy_df)
                print(f"  ✓ SPY: {len(spy_bars)} new bars → {len(spy_df)} total")
            else:
                print("  ⚠ SPY download failed")

        client.close()
        print()

    # ── News sentiment (independent of skip-download) ──
    # Download recent news for sentiment features (cached 1 day)
    news_df = None
    existing_news = store.read("smallcap_news") if store.exists("smallcap_news") else None

    def _news_age_days() -> int | None:
        import os
        path = store._path("smallcap_news")
        if not path.exists():
            return None
        mtime = pd.Timestamp(os.path.getmtime(path), unit="s")
        return (pd.Timestamp.now() - mtime).days

    skip_dl = getattr(args, 'skip_download', False)

    if existing_news is not None and not existing_news.empty:
        news_df = existing_news
        n_age = _news_age_days()
        if skip_dl or (n_age is not None and n_age < 1):
            print(f"  News cached ({n_age}d old) — {len(news_df)} rows ✓")
        else:
            print("  Refreshing news for sentiment features...")
            try:
                from app.data.massive import MassiveClient
                from app.data.massive.news import NewsAPI
                from app.features.sentiment import download_news_bulk
                nc = MassiveClient(calls_per_minute=5)
                na = NewsAPI(nc)
                news_df = download_news_bulk(na, args.train_start, args.predict_to,
                                             chunk_days=30, max_pages_per_chunk=10)
                nc.close()
                if not news_df.empty:
                    store.write("smallcap_news", news_df)
                    print(f"  ✓ News: {len(news_df):,} rows")
            except Exception as e:
                print(f"  ⚠ News download failed: {e}")
                news_df = existing_news
    elif not skip_dl:
        print("  Downloading news for sentiment features...")
        try:
            from app.data.massive import MassiveClient
            from app.data.massive.news import NewsAPI
            from app.features.sentiment import download_news_bulk
            nc = MassiveClient(calls_per_minute=5)
            na = NewsAPI(nc)
            news_df = download_news_bulk(na, args.train_start, args.predict_to,
                                         chunk_days=30, max_pages_per_chunk=10)
            nc.close()
            if not news_df.empty:
                store.write("smallcap_news", news_df)
                print(f"  ✓ News: {len(news_df):,} rows")
        except Exception as e:
            print(f"  ⚠ News download failed: {e}")

    print()
    if getattr(args, 'skip_features', False):
        print("STEP 4/6 ▸ Loading cached features (--skip-features)...")
        features = store.read("features_smallcap")
        if features is None:
            print("  ⚠ No cached features found. Remove --skip-features to rebuild.")
            sys.exit(1)
        print(f"  ✓ Loaded {len(features):,} rows × {len(features.columns)} features from cache\n")
    else:
        print("STEP 4/6 ▸ Building feature matrix (+ market regime, sector, microstructure)...")
        from app.features.pipeline import build_feature_matrix

        # Load fundamentals if available
        fundamentals = None
        try:
            fund_raw = store.read("smallcap_fundamentals")
            if fund_raw is not None and not fund_raw.empty:
                from app.features.fundamentals import _pivot_fundamentals, compute_fundamental_features
                fund_pivoted = _pivot_fundamentals(fund_raw)
                if not fund_pivoted.empty:
                    fundamentals = compute_fundamental_features(fund_pivoted)
                    print(f"  ✓ Loaded fundamentals: {len(fundamentals)} rows, "
                          f"{len([c for c in fundamentals.columns if c not in ('ticker', 'date')])} features")
        except Exception as e:
            print(f"  ⚠ Fundamentals not available: {e}")

        # Load SPY market data for regime features (beta, vol_regime, trend)
        market_df = None
        try:
            spy_data = store.read("smallcap_spy")
            if spy_data is not None and not spy_data.empty:
                market_df = spy_data
                print(f"  ✓ Loaded SPY market data: {len(market_df)} bars")
        except Exception as e:
            print(f"  ⚠ SPY data not available: {e}")

        # Load dividends for feature engineering
        div_features = None
        try:
            divs_raw = store.read("smallcap_dividends")
            if divs_raw is not None and not divs_raw.empty:
                div_features = _build_dividend_features(divs_raw, ohlcv)
                print(f"  ✓ Built dividend features: {len(div_features)} rows")
        except Exception as e:
            print(f"  ⚠ Dividend features not available: {e}")

        # Load splits for feature engineering
        split_features = None
        try:
            splits_raw = store.read("smallcap_splits")
            if splits_raw is not None and not splits_raw.empty:
                split_features = _build_split_features(splits_raw)
                print(f"  ✓ Built split features: {len(split_features)} rows")
        except Exception as e:
            print(f"  ⚠ Split features not available: {e}")

        features = build_feature_matrix(
            ohlcv, fundamentals=fundamentals, market_df=market_df,
            universe=verified_tickers, horizons=[1, 5, 10, 20],
        )

        # Merge dividend and split features
        if div_features is not None and not div_features.empty:
            features = features.merge(div_features, on=["ticker", "date"], how="left")
            for c in div_features.columns:
                if c not in ("ticker", "date"):
                    features[c] = features[c].fillna(0)

        if split_features is not None and not split_features.empty:
            features["date"] = pd.to_datetime(features["date"])
            split_features["date"] = pd.to_datetime(split_features["date"])
            features = features.merge(split_features, on=["ticker", "date"], how="left")
            for c in split_features.columns:
                if c not in ("ticker", "date"):
                    features[c] = features[c].fillna(0)

        # Add lag / autoregressive features
        features = _add_lag_features(features)

        # Add sentiment features from news data
        if news_df is not None and not news_df.empty:
            try:
                from app.features.sentiment import build_sentiment_features, build_market_sentiment_features
                print("  Building sentiment features from news...")

                sent_features = build_sentiment_features(news_df, ohlcv, lookback_days=7)
                if not sent_features.empty:
                    features["date"] = pd.to_datetime(features["date"])
                    sent_features["date"] = pd.to_datetime(sent_features["date"])
                    features = features.merge(sent_features, on=["ticker", "date"], how="left")
                    for c in sent_features.columns:
                        if c not in ("ticker", "date"):
                            features[c] = features[c].fillna(0)
                    n_with = (features["has_news_7d"] > 0).sum()
                    print(f"  ✓ Ticker sentiment: {len(sent_features)} rows, "
                          f"{n_with} with news ({n_with*100//max(len(features),1)}%)")

                # NOTE: Market-wide sentiment disabled — dominates model and
                # displaces per-ticker price-action features (tested 2026-05-11).
                # mkt_sent = build_market_sentiment_features(news_df, ohlcv, lookback_days=7)
                # if not mkt_sent.empty:
                #     features = features.merge(mkt_sent, on="date", how="left")
                #     for c in mkt_sent.columns:
                #         if c != "date":
                #             features[c] = features[c].fillna(0)
                #     print(f"  ✓ Market sentiment: {len(mkt_sent)} dates")
            except Exception as e:
                print(f"  ⚠ Sentiment features failed: {e}")

        store.write("features_smallcap", features)
        print(f"  ✓ {len(features):,} rows × {len(features.columns)} features\n")

        # Defragment the DataFrame to avoid segfault in C libraries (LGB/XGB)
        features = features.copy()

    # ═══════════════════════════════════════════════════════
    # STEP 5: Train models
    # ═══════════════════════════════════════════════════════
    print("STEP 5/6 ▸ Training models (LGB+XGB+CB, SHAP selection, triple-barrier, regime)...")
    cls_model, cls_m, reg_model, reg_m, q_models, predict_data, rank_model = (
        train_models(features, args.predict_from, cfg)
    )
    print()

    # ═══════════════════════════════════════════════════════
    # STEP 6: Generate signals & backtest
    # ═══════════════════════════════════════════════════════
    print("STEP 6/7 ▸ Generating predictions & validation backtest...")

    # Filter predict_data to validation period only
    val_predict = predict_data[
        pd.to_datetime(predict_data["date"]) < pd.Timestamp(args.holdout_from)
    ].copy()

    signals_df = generate_signals(cls_model, reg_model, q_models, val_predict, args.top,
                                   features_df=features, ohlcv=ohlcv, rank_model=rank_model)

    if signals_df.empty:
        print("  ⚠ No signals generated.")
        sys.exit(1)

    buys = signals_df[signals_df["recommendation"] == "BUY"]
    holds = signals_df[signals_df["recommendation"] == "HOLD"]
    no_trades = signals_df[signals_df["recommendation"] == "NO_TRADE"]
    print(f"  Validation signals: {len(signals_df)}")
    print(f"    BUY: {len(buys)} | HOLD: {len(holds)} | NO_TRADE: {len(no_trades)}")

    # Enrich BUYs with prices
    buys_enriched = buys.copy()
    buys_enriched["date"] = pd.to_datetime(buys_enriched["date"])
    ohlcv_prices = ohlcv[["date", "ticker", "close"]].copy()
    ohlcv_prices["date"] = pd.to_datetime(ohlcv_prices["date"])
    buys_enriched = buys_enriched.merge(
        ohlcv_prices.rename(columns={"close": "close_price"}),
        on=["ticker", "date"], how="left",
    )
    store.write("signals_smallcap_2026", signals_df)
    print()

    # Validation backtest
    from app.backtest import BacktestConfig, Backtester
    from app.reporting import generate_text_report, generate_html_report

    config = BacktestConfig(
        start_date=args.predict_from,
        end_date=args.holdout_from,
        max_positions=4,
        rebalance_frequency="weekly",
        commission_bps=cfg.commission_bps,
        slippage_bps=cfg.slippage_bps,
        holding_period_trading_days=44,
        use_stop_loss=False,
        use_take_profit=False,
        trailing_stop_pct=0.16,
    )
    bt = Backtester(config)
    result = bt.run(signals_df, ohlcv)

    print("  ── VALIDATION PERIOD ──")
    report = generate_text_report(result)
    print(report)

    # ── Deflated Sharpe Ratio & Monte Carlo robustness ──
    _compute_robustness_metrics(result, n_trials=24)

    output_path = Path(args.output)
    generate_html_report(result, output_path)
    print(f"  HTML report: {output_path}\n")

    # Display rich tables (validation only)
    display_results(verified_tickers, cls_model, cls_m, reg_m, signals_df,
                    buys_enriched, result, cfg, args, n_tickers, output_path)

    # ═══════════════════════════════════════════════════════
    # STEP 6b: Walk-Forward Cross-Validation (rigorous OOS evaluation)
    # ═══════════════════════════════════════════════════════
    if args.wf_cv:
        print("\n" + "=" * 70)
        print("  STEP 6b ▸ WALK-FORWARD CROSS-VALIDATION (purged, multi-fold)")
        print("  ⚠ This replaces the compromised holdout as primary validation")
        print("=" * 70)
        walk_forward_cv(features, ohlcv, cfg, n_folds=args.wf_folds)

    # ═══════════════════════════════════════════════════════
    # STEP 7: Holdout evaluation (only when explicitly requested)
    # ═══════════════════════════════════════════════════════
    if args.eval_holdout:
        print("\n" + "=" * 70)
        print("  STEP 7/7 ▸ HOLDOUT EVALUATION (final, unseen period)")
        print("=" * 70)

        holdout_predict = predict_data[
            pd.to_datetime(predict_data["date"]) >= pd.Timestamp(args.holdout_from)
        ].copy()

        holdout_signals = generate_signals(
            cls_model, reg_model, q_models, holdout_predict, args.top,
            features_df=features, ohlcv=ohlcv,
        )

        if holdout_signals.empty:
            print("  ⚠ No holdout signals generated.")
        else:
            h_buys = holdout_signals[holdout_signals["recommendation"] == "BUY"]
            print(f"  Holdout signals: {len(holdout_signals)} (BUY: {len(h_buys)})")

            # Save holdout signals
            store.write("signals_smallcap_2026_holdout", holdout_signals)
            print(f"  Saved holdout signals to store")

            # Mirror validation config for fair comparison
            holdout_config = BacktestConfig(
                start_date=args.holdout_from,
                end_date=args.predict_to,
                initial_capital=config.initial_capital,
                max_positions=config.max_positions,
                rebalance_frequency=config.rebalance_frequency,
                commission_bps=config.commission_bps,
                slippage_bps=config.slippage_bps,
                holding_period_trading_days=config.holding_period_trading_days,
                use_stop_loss=config.use_stop_loss,
                use_take_profit=config.use_take_profit,
                trailing_stop_pct=config.trailing_stop_pct,
            )
            holdout_bt = Backtester(holdout_config)
            holdout_result = holdout_bt.run(holdout_signals, ohlcv)

            print("  ── HOLDOUT PERIOD ──")
            holdout_report = generate_text_report(holdout_result)
            print(holdout_report)

            holdout_path = Path(str(args.output).replace(".html", "_holdout.html"))
            generate_html_report(holdout_result, holdout_path)
            print(f"  Holdout HTML report: {holdout_path}")

            hm = holdout_result.metrics
            vm = result.metrics
            print(f"\n  ── COMPARISON: Validation vs Holdout ──")
            print(f"    {'Metric':<20} {'Validation':>12} {'Holdout':>12}")
            print(f"    {'─'*20} {'─'*12} {'─'*12}")
            print(f"    {'Total Return':<20} {vm.get('total_return',0):>+11.2%} {hm.get('total_return',0):>+11.2%}")
            print(f"    {'Sharpe Ratio':<20} {vm.get('sharpe_ratio',0):>12.2f} {hm.get('sharpe_ratio',0):>12.2f}")
            print(f"    {'Max Drawdown':<20} {vm.get('max_drawdown',0):>11.2%} {hm.get('max_drawdown',0):>11.2%}")
            print(f"    {'# Trades':<20} {vm.get('n_trades',0):>12} {hm.get('n_trades',0):>12}")
    else:
        print(f"\n  ℹ Holdout period ({args.holdout_from} → {args.predict_to}) reserved.")
        print(f"    Use --eval-holdout for final evaluation (only once!).\n")


if __name__ == "__main__":
    main()
