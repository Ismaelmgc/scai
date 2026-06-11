"""SEC EDGAR connector — fundamentals and corporate event features.

Free, no API key needed. Uses XBRL Company Facts API.
All data is point-in-time via filing_date/accepted_date.

Key features for small-cap analysis:
- Dilution events (shelf registration, ATM offerings)
- Cash runway estimation
- Insider trading (Form 4)
- Going concern warnings
- Debt/warrant structure
"""
from __future__ import annotations

import time
from datetime import date

import httpx
import pandas as pd

import os

from app.utils import get_logger

log = get_logger(__name__)

EDGAR_BASE = "https://data.sec.gov"
_ua = os.environ.get("SEC_EDGAR_USER_AGENT", "SCAI-Research research@scai-project.com")
HEADERS = {"User-Agent": _ua, "Accept-Encoding": "gzip, deflate"}

# Key XBRL concepts for small-cap analysis
FINANCIAL_CONCEPTS = {
    # Cash & liquidity
    "CashAndCashEquivalentsAtCarryingValue": "cash",
    "CashCashEquivalentsAndShortTermInvestments": "cash_and_investments",
    # Revenue & profitability
    "Revenues": "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue_contracts",
    "NetIncomeLoss": "net_income",
    "OperatingIncomeLoss": "operating_income",
    # Balance sheet
    "Assets": "total_assets",
    "AssetsCurrent": "current_assets",
    "Liabilities": "total_liabilities",
    "LiabilitiesCurrent": "current_liabilities",
    "StockholdersEquity": "equity",
    # Debt
    "LongTermDebt": "long_term_debt",
    "LongTermDebtNoncurrent": "lt_debt_noncurrent",
    "DebtCurrent": "current_debt",
    "ConvertibleNotesPayable": "convertible_debt",
    # Shares
    "CommonStockSharesOutstanding": "shares_outstanding",
    "WeightedAverageNumberOfShareOutstandingBasicAndDiluted": "shares_diluted",
    "CommonStockSharesAuthorized": "shares_authorized",
    # Cash flow
    "NetCashProvidedByUsedInOperatingActivities": "cfo",
    "NetCashProvidedByUsedInFinancingActivities": "cff",
    "PaymentsToAcquirePropertyPlantAndEquipment": "capex",
}


def get_cik_map(tickers: list[str]) -> dict[str, str]:
    """Map ticker symbols to SEC CIK numbers."""
    r = httpx.get("https://www.sec.gov/files/company_tickers.json",
                   headers=HEADERS, timeout=15)
    r.raise_for_status()
    registry = r.json()

    ticker_set = set(t.upper() for t in tickers)
    cik_map = {}
    for _, v in registry.items():
        t = v.get("ticker", "").upper()
        if t in ticker_set:
            cik_map[t] = str(v["cik_str"])

    log.info("edgar_cik_mapped", requested=len(tickers), found=len(cik_map))
    return cik_map


def download_company_facts(
    tickers: list[str],
    max_tickers: int | None = None,
    delay: float = 0.15,
) -> pd.DataFrame:
    """Download key financial facts from SEC EDGAR for a list of tickers.

    Returns DataFrame with point-in-time financial data.
    Columns: ticker, concept, value, filed, end_date, form, source.
    """
    cik_map = get_cik_map(tickers)
    if max_tickers:
        items = list(cik_map.items())[:max_tickers]
        cik_map = dict(items)

    all_rows: list[dict] = []
    errors: list[str] = []

    for i, (ticker, cik) in enumerate(cik_map.items()):
        try:
            cik_padded = cik.zfill(10)
            url = f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik_padded}.json"
            r = httpx.get(url, headers=HEADERS, timeout=15)

            if r.status_code == 404:
                errors.append(ticker)
                continue
            r.raise_for_status()

            facts = r.json()
            us_gaap = facts.get("facts", {}).get("us-gaap", {})

            for concept, short_name in FINANCIAL_CONCEPTS.items():
                if concept not in us_gaap:
                    continue
                units = us_gaap[concept].get("units", {})
                # Get USD values (or shares for share-related concepts)
                for unit_type in ["USD", "shares"]:
                    if unit_type not in units:
                        continue
                    for entry in units[unit_type]:
                        all_rows.append({
                            "ticker": ticker,
                            "cik": cik,
                            "concept": short_name,
                            "value": entry.get("val"),
                            "filed": entry.get("filed"),
                            "end_date": entry.get("end"),
                            "start_date": entry.get("start"),
                            "form": entry.get("form"),
                            "unit": unit_type,
                            "source": "sec_edgar",
                        })

            if (i + 1) % 20 == 0:
                log.info("edgar_progress", downloaded=i + 1, total=len(cik_map))

        except Exception as e:
            log.warning("edgar_ticker_error", ticker=ticker, error=str(e))
            errors.append(ticker)

        time.sleep(delay)

    if errors:
        log.warning("edgar_errors", count=len(errors), tickers=errors[:10])

    if not all_rows:
        return pd.DataFrame()

    result = pd.DataFrame(all_rows)
    result["filed"] = pd.to_datetime(result["filed"])
    result["end_date"] = pd.to_datetime(result["end_date"])
    log.info("edgar_download_complete",
             tickers=len(cik_map) - len(errors),
             rows=len(result))
    return result


def compute_edgar_features(facts_df: pd.DataFrame) -> pd.DataFrame:
    """Compute small-cap specific features from EDGAR financial data.

    All features are point-in-time (based on filing date, not period end).
    Returns: ticker, filing_date, feature columns.
    """
    if facts_df.empty:
        return pd.DataFrame()

    df = facts_df.copy()
    # Only use 10-K and 10-Q filings
    df = df[df["form"].isin(["10-K", "10-Q"])].copy()
    # Use filing date as the point-in-time anchor
    df = df.sort_values(["ticker", "concept", "filed"])
    df = df.drop_duplicates(subset=["ticker", "concept", "filed"], keep="last")

    # Pivot: one row per (ticker, filing_date) with concept columns
    pivot = df.pivot_table(
        index=["ticker", "filed"],
        columns="concept",
        values="value",
        aggfunc="last",
    ).reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={"filed": "filing_date"})
    pivot = pivot.sort_values(["ticker", "filing_date"])

    # Compute derived features
    if "cash" in pivot.columns and "cfo" in pivot.columns:
        # Cash runway: months of cash at current burn rate
        quarterly_burn = pivot["cfo"].clip(upper=0).abs()
        monthly_burn = quarterly_burn / 3
        pivot["cash_runway_months"] = pivot["cash"].div(monthly_burn.replace(0, float("nan")))
        pivot["cash_runway_months"] = pivot["cash_runway_months"].clip(upper=120)

    if "shares_outstanding" in pivot.columns:
        # Dilution: % change in shares outstanding vs prior filing
        pivot["dilution_pct"] = pivot.groupby("ticker")["shares_outstanding"].pct_change()

    if "convertible_debt" in pivot.columns and "total_assets" in pivot.columns:
        pivot["convertible_debt_ratio"] = pivot["convertible_debt"] / pivot["total_assets"].replace(0, float("nan"))

    if "current_assets" in pivot.columns and "current_liabilities" in pivot.columns:
        pivot["current_ratio"] = pivot["current_assets"] / pivot["current_liabilities"].replace(0, float("nan"))

    if "revenue" in pivot.columns:
        pivot["revenue_growth"] = pivot.groupby("ticker")["revenue"].pct_change()

    pivot["source"] = "sec_edgar"
    return pivot
