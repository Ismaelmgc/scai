"""Point-in-time small-cap universe discovery (Polygon reference data).

Extracted verbatim from the retired V2 research pipeline; used by
scripts/refresh_universe.py (monthly universe refresh).
"""
from __future__ import annotations

from datetime import date

import pandas as pd

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
                n_inactive = ((catalog["active"] == False).sum()
                              if "active" in catalog.columns else "?")
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

        n_new = len(verified) - len(already_known)
        if n_new <= 20 or n_new % 25 == 0:
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
