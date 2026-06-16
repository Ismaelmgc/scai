"""Short-interest features (price-orthogonal small-cap signal).

FINRA short interest is reported twice a month (settlement ~15th and month-end)
and predicts forward returns in the cross-section (high days-to-cover / rising
short interest → weaker returns; also squeeze fuel). It is orthogonal to the
price/volume/volatility features the model already uses.

POINT-IN-TIME SAFETY (critical): the data is keyed by `settlement_date`, but
FINRA disseminates it ~8 business days LATER. We therefore build an
`available_date = settlement_date + avail_lag_bdays` and merge_asof BACKWARD on
that — a feature row at date T only ever sees reports disseminated on/before T.
`si_change_2w` uses only the prior report (past data), so it is PIT-safe too.
A 45-day tolerance drops stale carry-forward when a name stops reporting.
"""
from __future__ import annotations

import pandas as pd

# Conservative buffer over FINRA's ~8-business-day dissemination lag.
AVAIL_LAG_BDAYS = 10
# Max age of a report before it's considered stale (bi-monthly cadence + lag).
STALE_TOLERANCE_DAYS = 45

SI_FEATURES = ["si_days_to_cover", "si_change_2w"]


def add_short_interest_features(
    features: pd.DataFrame,
    si: pd.DataFrame,
    avail_lag_bdays: int = AVAIL_LAG_BDAYS,
) -> tuple[pd.DataFrame, list[str]]:
    """Merge PIT-safe short-interest features into the feature matrix.

    Returns (features_with_si, new_feature_names).
    """
    si = si.copy()
    si["settlement_date"] = pd.to_datetime(si["settlement_date"])
    si = si.sort_values(["ticker", "settlement_date"])
    # Short-interest momentum vs the previous report (past-only → PIT-safe).
    si["si_change_2w"] = si.groupby("ticker")["short_interest"].pct_change()
    si = si.rename(columns={"days_to_cover": "si_days_to_cover"})
    # Availability = settlement + dissemination buffer. NEVER use settlement_date
    # as the as-of date (that would leak ~8 business days of future knowledge).
    si["available_date"] = (
        si["settlement_date"] + pd.tseries.offsets.BusinessDay(avail_lag_bdays)
    )
    si = si.sort_values("available_date")

    feat = features.copy()
    feat["date"] = pd.to_datetime(feat["date"])
    feat = feat.sort_values("date")

    merged = pd.merge_asof(
        feat,
        si[["available_date", "ticker", "si_days_to_cover", "si_change_2w"]],
        left_on="date",
        right_on="available_date",
        by="ticker",
        direction="backward",
        tolerance=pd.Timedelta(days=STALE_TOLERANCE_DAYS),
    ).drop(columns=["available_date"])

    return merged, list(SI_FEATURES)
