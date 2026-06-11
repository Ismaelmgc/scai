"""Integration test – full pipeline from OHLCV to predictions."""

import pytest

from app.features.pipeline import build_feature_matrix
from app.models.tabular import TabularModel


@pytest.mark.integration
def test_full_pipeline(sample_ohlcv):
    """End-to-end pipeline: features → train → predict."""
    # 1. Build features
    features = build_feature_matrix(sample_ohlcv, horizons=[5])
    assert len(features) > 0
    assert "ret_1d" in features.columns
    assert "fwd_ret_5d" in features.columns

    # 2. Train
    clean = features.dropna(subset=["fwd_ret_5d_positive"])
    if len(clean) < 30:
        pytest.skip("Insufficient data")

    cls_model = TabularModel(horizon=5, task="classification")
    cls_model.train(clean)

    reg_model = TabularModel(horizon=5, task="regression")
    reg_model.train(clean)

    # 3. Predict
    cls_preds = cls_model.predict_df(clean)
    reg_preds = reg_model.predict_df(clean)

    merged = clean[["ticker", "date"]].copy()
    merged = merged.merge(cls_preds, on=["ticker", "date"])
    merged = merged.merge(reg_preds, on=["ticker", "date"])

    assert len(merged) > 0
    assert "pred_classification_5d" in merged.columns
    assert "pred_regression_5d" in merged.columns
