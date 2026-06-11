"""Point-in-time fundamental features from SEC EDGAR XBRL data.

CRITICAL: We use the ``filed`` date as the availability date, **not** the
period end.  This is the core mechanism to prevent look-ahead bias in
fundamental data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.utils import get_logger

log = get_logger(__name__)

# XBRL concepts we extract (us-gaap taxonomy)
_CONCEPT_MAP: dict[str, str] = {
    "Revenues": "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "SalesRevenueNet": "revenue",
    "GrossProfit": "gross_profit",
    "OperatingIncomeLoss": "operating_income",
    "NetIncomeLoss": "net_income",
    "EarningsPerShareBasic": "eps_basic",
    "EarningsPerShareDiluted": "eps_diluted",
    "Assets": "total_assets",
    "Liabilities": "total_liabilities",
    "StockholdersEquity": "equity",
    "AssetsCurrent": "current_assets",
    "LiabilitiesCurrent": "current_liabilities",
    "CashAndCashEquivalentsAtCarryingValue": "cash",
    "LongTermDebt": "long_term_debt",
    "CommonStockSharesOutstanding": "shares_outstanding",
    "OperatingCashFlow": "operating_cf",  # alias
    "NetCashProvidedByUsedInOperatingActivities": "operating_cf",
}


def _pivot_fundamentals(raw_fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Pivot raw XBRL facts into one-row-per-(ticker, filed) with concept columns.

    Uses ``filed`` date (point-in-time), keeps only annual (10-K) and
    quarterly (10-Q) filings, takes the latest value per concept per filing.
    """
    if raw_fundamentals.empty:
        return pd.DataFrame()

    df = raw_fundamentals.copy()

    # Map concept names
    df["feature"] = df["concept"].map(_CONCEPT_MAP)
    df = df.dropna(subset=["feature", "filed", "value"])

    # Keep only 10-K and 10-Q
    if "form" in df.columns:
        df = df[df["form"].isin(["10-K", "10-Q"])]

    # Deduplicate: keep last value per (ticker, filed, feature)
    df = df.sort_values("period_end").drop_duplicates(
        subset=["ticker", "filed", "feature"], keep="last"
    )

    # Pivot
    pivoted = df.pivot_table(
        index=["ticker", "filed"],
        columns="feature",
        values="value",
        aggfunc="last",
    ).reset_index()

    pivoted = pivoted.rename(columns={"filed": "date"})
    pivoted["date"] = pd.to_datetime(pivoted["date"])
    return pivoted.sort_values(["ticker", "date"])


def compute_fundamental_features(pivoted: pd.DataFrame) -> pd.DataFrame:
    """Derive financial ratios from pivoted fundamental data.

    All ratios are computed from data that was available at the ``date``
    (= filing date), ensuring point-in-time correctness.
    """
    if pivoted.empty:
        return pivoted

    df = pivoted.copy()
    grouped = df.groupby("ticker")

    # Margins
    if "revenue" in df.columns and "gross_profit" in df.columns:
        df["gross_margin"] = df["gross_profit"] / df["revenue"].replace(0, np.nan)
    if "revenue" in df.columns and "operating_income" in df.columns:
        df["ebitda_margin"] = df["operating_income"] / df["revenue"].replace(0, np.nan)
    if "revenue" in df.columns and "net_income" in df.columns:
        df["net_margin"] = df["net_income"] / df["revenue"].replace(0, np.nan)

    # Returns
    if "net_income" in df.columns and "total_assets" in df.columns:
        df["roa"] = df["net_income"] / df["total_assets"].replace(0, np.nan)
    if "net_income" in df.columns and "equity" in df.columns:
        df["roe"] = df["net_income"] / df["equity"].replace(0, np.nan)

    # Leverage
    if "total_liabilities" in df.columns and "equity" in df.columns:
        df["leverage"] = df["total_liabilities"] / df["equity"].replace(0, np.nan)
    if "long_term_debt" in df.columns and "total_assets" in df.columns:
        df["debt_to_assets"] = df["long_term_debt"] / df["total_assets"].replace(0, np.nan)

    # Liquidity ratios
    if "current_assets" in df.columns and "current_liabilities" in df.columns:
        df["current_ratio"] = df["current_assets"] / df["current_liabilities"].replace(0, np.nan)
    if "cash" in df.columns and "current_liabilities" in df.columns:
        df["cash_ratio"] = df["cash"] / df["current_liabilities"].replace(0, np.nan)

    # Growth (yoy – 4 quarters back)
    if "revenue" in df.columns:
        df["revenue_growth_yoy"] = grouped["revenue"].pct_change(4)
    if "total_assets" in df.columns:
        df["asset_growth_yoy"] = grouped["total_assets"].pct_change(4)

    # Accruals (net income - operating CF) / total assets
    if all(c in df.columns for c in ["net_income", "operating_cf", "total_assets"]):
        df["accruals"] = (df["net_income"] - df["operating_cf"]) / df["total_assets"].replace(
            0,
            np.nan,
        )

    # Share issuance / dilution
    if "shares_outstanding" in df.columns:
        df["share_issuance_yoy"] = grouped["shares_outstanding"].pct_change(4)

    return df
