"""Feature pipeline – orchestrate all feature generators into a single matrix."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from app.features.cross_sectional import compute_cross_sectional_features
from app.features.liquidity import compute_liquidity_features
from app.features.microstructure import compute_microstructure_features
from app.features.momentum import compute_momentum_features
from app.features.price_action import compute_price_action_features
from app.features.sector import assign_sectors, compute_sector_features
from app.features.volatility import compute_volatility_features
from app.features.alpha_features import compute_alpha_features
from app.utils import get_logger
from app.utils.point_in_time import as_of

log = get_logger(__name__)


# ── Triple Barrier Labeling (vectorized) ────────────────────
def _triple_barrier_labels(
    df: pd.DataFrame,
    horizon: int = 5,
    pt_mult: float = 1.5,
    sl_mult: float = 1.5,
) -> pd.DataFrame:
    """Add triple-barrier labels per López de Prado (vectorized with NumPy).

    For each row, look forward up to ``horizon`` days:
      - If price hits take-profit first → label = 1
      - If price hits stop-loss first  → label = -1
      - If neither within horizon      → label = sign(return at horizon)

    Barriers are set at ``pt_mult`` / ``sl_mult`` times the daily vol.
    """
    df = df.sort_values(["ticker", "date"]).copy()

    vol_col = "realized_vol_20d"
    if vol_col not in df.columns:
        grouped = df.groupby("ticker")
        daily_ret = grouped["close"].pct_change()
        df["_daily_vol"] = daily_ret.groupby(df["ticker"]).transform(
            lambda x: x.rolling(20, min_periods=5).std()
        )
    else:
        df["_daily_vol"] = df[vol_col] / np.sqrt(252)

    labels = np.full(len(df), np.nan)
    barrier_types = np.full(len(df), "", dtype=object)

    for ticker, gdf in df.groupby("ticker"):
        closes = gdf["close"].values
        vols = gdf["_daily_vol"].values
        n = len(closes)
        if n < 2:
            continue

        # Build forward price matrix (n x horizon) using stride tricks
        # For each position i, forward_prices[i, j] = closes[i + j + 1]
        valid_mask = (~np.isnan(vols)) & (vols > 0)
        pt_barriers = closes * (1 + pt_mult * vols * np.sqrt(horizon))
        sl_barriers = closes * (1 - sl_mult * vols * np.sqrt(horizon))

        for i in range(n):
            if not valid_mask[i]:
                continue
            end = min(i + horizon, n - 1)
            if end <= i:
                continue

            fwd = closes[i + 1:end + 1]
            # Check take-profit hits
            pt_hits = np.where(fwd >= pt_barriers[i])[0]
            sl_hits = np.where(fwd <= sl_barriers[i])[0]

            first_pt = pt_hits[0] if len(pt_hits) > 0 else horizon + 1
            first_sl = sl_hits[0] if len(sl_hits) > 0 else horizon + 1

            iloc = gdf.index[i]
            pos = df.index.get_loc(iloc)

            if first_pt <= first_sl and first_pt < horizon + 1:
                labels[pos] = 1
                barrier_types[pos] = "tp"
            elif first_sl < first_pt and first_sl < horizon + 1:
                labels[pos] = -1
                barrier_types[pos] = "sl"
            else:
                ret = closes[end] / closes[i] - 1
                labels[pos] = 1 if ret > 0 else -1
                barrier_types[pos] = "timeout"

    df[f"tb_label_{horizon}d"] = labels
    df[f"tb_barrier_type_{horizon}d"] = barrier_types
    df[f"tb_label_{horizon}d_positive"] = (df[f"tb_label_{horizon}d"] > 0).astype(float)
    df.loc[df[f"tb_label_{horizon}d"].isna(), f"tb_label_{horizon}d_positive"] = np.nan
    df = df.drop(columns=["_daily_vol"], errors="ignore")
    return df


def _build_risk_adjusted_targets(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Add risk-adjusted forward-return labels.

    - ``fwd_ret_{h}d_risk_adj``: forward return / trailing realised vol
    - ``fwd_ret_{h}d_sector_rel``: forward return minus sector average
    - ``fwd_ret_{h}d_risk_adj_positive``: binary from risk-adjusted return
    """
    grouped = df.groupby("ticker")
    for h in horizons:
        fwd_col = f"fwd_ret_{h}d"
        if fwd_col not in df.columns:
            continue

        # Risk-adjusted: return / trailing vol (Sharpe-like)
        vol_col = "realized_vol_20d" if "realized_vol_20d" in df.columns else None
        if vol_col:
            trailing_vol = df[vol_col].replace(0, np.nan)
            # Annualised vol → daily vol for matching horizon scale
            daily_vol = trailing_vol / np.sqrt(252)
            horizon_vol = daily_vol * np.sqrt(h)
            df[f"fwd_ret_{h}d_risk_adj"] = df[fwd_col] / horizon_vol.replace(0, np.nan)
            df[f"fwd_ret_{h}d_risk_adj_positive"] = (
                df[f"fwd_ret_{h}d_risk_adj"] > 0
            ).astype(int)

        # Sector-relative return
        if "sector" in df.columns:
            sector_avg = df.groupby(["date", "sector"])[fwd_col].transform("mean")
            df[f"fwd_ret_{h}d_sector_rel"] = df[fwd_col] - sector_avg

    return df


def build_feature_matrix(
    ohlcv: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
    news: pd.DataFrame | None = None,
    market_df: pd.DataFrame | None = None,
    universe: list[dict] | pd.DataFrame | None = None,
    as_of_date: date | str | None = None,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Build the full feature matrix from raw data.

    Parameters
    ----------
    ohlcv : Daily OHLCV panel (ticker × date).
    fundamentals : Pivoted fundamentals (optional).
    news : News/headlines (optional).
    market_df : Market index OHLCV for regime features (optional).
    universe : List of dicts with 'ticker', 'sic_code' for sector assignment.
    as_of_date : If set, filter all data point-in-time before building features.
    horizons : Forward-return horizons for label generation.

    Returns
    -------
    DataFrame with all features and forward-return labels.
    """
    horizons = horizons or [1, 5, 10, 20]

    # Point-in-time filter
    if as_of_date:
        ohlcv = as_of(ohlcv, as_of_date)
        if fundamentals is not None:
            fundamentals = as_of(fundamentals, as_of_date)
        if news is not None:
            news = as_of(news, as_of_date)

    df = ohlcv.sort_values(["ticker", "date"]).copy()
    log.info("building_features", rows=len(df), tickers=df["ticker"].nunique())

    # 0. Assign sectors from universe metadata (SIC codes)
    df = assign_sectors(df, universe)

    # 1. Price action
    df = compute_price_action_features(df)

    # 2. Volatility
    market_ret = None
    if market_df is not None:
        from app.features.market_regime import compute_market_regime_features
        mkt = compute_market_regime_features(market_df)
        market_ret = mkt.set_index("date")["mkt_ret_1d"]
        # Merge market regime features
        mkt_cols = [c for c in mkt.columns if c.startswith("mkt_") or c == "vol_regime"]
        df = df.merge(mkt[["date"] + mkt_cols], on="date", how="left")

    df = compute_volatility_features(df, market_returns=market_ret)

    # 3. Liquidity
    df = compute_liquidity_features(df)

    # 4. Momentum
    df = compute_momentum_features(df)

    # 4b. Alpha features (statistical, interaction, high-value)
    df = compute_alpha_features(df)

    # 5. Microstructure features (VWAP, volume profile, spread proxies)
    df = compute_microstructure_features(df)

    # 6. Sector-relative features
    df = compute_sector_features(df)

    # 7. Fundamentals (left-join on ticker + most recent filing ≤ date)
    if fundamentals is not None and not fundamentals.empty:
        fund_cols = [c for c in fundamentals.columns if c not in ("ticker", "date")]
        fund = fundamentals[["ticker", "date"] + fund_cols].copy()
        fund = fund.rename(columns={"date": "fund_date"})
        df = pd.merge_asof(
            df.sort_values("date"),
            fund.sort_values("fund_date"),
            by="ticker",
            left_on="date",
            right_on="fund_date",
            direction="backward",
        )
        df = df.drop(columns=["fund_date"], errors="ignore")

    # 8. Cross-sectional ranks and normalisation
    df = compute_cross_sectional_features(df)

    # 9. Forward return labels (for training – will be NaN at inference edges)
    grouped = df.groupby("ticker")
    for h in horizons:
        df[f"fwd_ret_{h}d"] = grouped["close"].transform(
            lambda x: x.shift(-h) / x - 1
        )
        df[f"fwd_ret_{h}d_positive"] = (df[f"fwd_ret_{h}d"] > 0).astype(int)

        # Cross-sectional relative target: outperform the daily median
        date_median = df.groupby("date")[f"fwd_ret_{h}d"].transform("median")
        df[f"fwd_ret_{h}d_xsec_positive"] = (
            df[f"fwd_ret_{h}d"] > date_median
        ).astype(int)
        df.loc[df[f"fwd_ret_{h}d"].isna(), f"fwd_ret_{h}d_xsec_positive"] = np.nan

    # 9b. Downside-aware target: max drawdown within holding period
    # This captures what the P10 quantile model is predicting
    if 5 in horizons:
        def _fwd_max_dd(closes):
            """Compute max drawdown within next 5 days for each position."""
            result = np.full(len(closes), np.nan)
            vals = closes.values
            for i in range(len(vals) - 5):
                fwd_prices = vals[i+1:i+6]
                min_ret = np.min(fwd_prices / vals[i] - 1)
                result[i] = min_ret
            return pd.Series(result, index=closes.index)

        df["fwd_min_ret_5d"] = grouped["close"].transform(_fwd_max_dd)
        # Binary: survived without >5% drawdown within 5 days
        df["fwd_survived_5d"] = (df["fwd_min_ret_5d"] > -0.05).astype(int)
        df.loc[df["fwd_min_ret_5d"].isna(), "fwd_survived_5d"] = np.nan

    # 10. Risk-adjusted and sector-relative targets
    df = _build_risk_adjusted_targets(df, horizons)

    # 11. Triple barrier labels (better than simple positive/negative)
    primary_h = horizons[0] if horizons else 5
    df = _triple_barrier_labels(df, horizon=primary_h)

    # 12. Multi-horizon agreement features (consensus across timeframes)
    if len(horizons) >= 2:
        fwd_pos_cols = [f"fwd_ret_{h}d_positive" for h in horizons if f"fwd_ret_{h}d_positive" in df.columns]
        if len(fwd_pos_cols) >= 2:
            df["multi_horizon_agreement"] = df[fwd_pos_cols].mean(axis=1)
            df["multi_horizon_unanimous_up"] = (df[fwd_pos_cols].sum(axis=1) == len(fwd_pos_cols)).astype(int)

    log.info("features_built", rows=len(df), columns=len(df.columns))
    return df
