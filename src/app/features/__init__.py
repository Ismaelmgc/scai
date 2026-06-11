"""Feature layer – generate point-in-time features from raw data.

All feature generators receive a panel DataFrame (ticker × date) and
return it augmented with new columns.  Every generator is careful to
**never** use future information.
"""

from __future__ import annotations

from app.features.cross_sectional import compute_cross_sectional_features
from app.features.fundamentals import compute_fundamental_features
from app.features.liquidity import compute_liquidity_features
from app.features.market_regime import compute_market_regime_features
from app.features.momentum import compute_momentum_features
from app.features.pipeline import build_feature_matrix
from app.features.price_action import compute_price_action_features
from app.features.volatility import compute_volatility_features

__all__ = [
    "compute_price_action_features",
    "compute_volatility_features",
    "compute_liquidity_features",
    "compute_momentum_features",
    "compute_fundamental_features",
    "compute_market_regime_features",
    "compute_cross_sectional_features",
    "build_feature_matrix",
]
