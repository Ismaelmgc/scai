"""Tests for the tabular model."""

import numpy as np
import pytest

from app.features.pipeline import build_feature_matrix
from app.models.tabular import TabularModel


def test_tabular_model_train_predict(sample_ohlcv, tmp_path):
    """Model should train and predict without errors on synthetic data."""
    features = build_feature_matrix(sample_ohlcv, horizons=[5])
    # Drop rows with NaN labels
    features = features.dropna(subset=["fwd_ret_5d_positive"])
    if len(features) < 50:
        pytest.skip("Not enough data for training")

    model = TabularModel(horizon=5, task="classification")
    metrics = model.train(features)

    assert "val_auc" in metrics
    assert "n_train" in metrics
    assert metrics["n_train"] > 0

    preds = model.predict(features)
    assert len(preds) == len(features)
    assert all(0 <= p <= 1 for p in preds)

    # Save and load
    model.save(tmp_path / "test_model.pkl")
    loaded = TabularModel.load(tmp_path / "test_model.pkl")
    preds2 = loaded.predict(features)
    np.testing.assert_array_almost_equal(preds, preds2)


def test_tabular_regression(sample_ohlcv):
    features = build_feature_matrix(sample_ohlcv, horizons=[5])
    features = features.dropna(subset=["fwd_ret_5d"])
    if len(features) < 50:
        pytest.skip("Not enough data")

    model = TabularModel(horizon=5, task="regression")
    metrics = model.train(features)
    assert "val_rmse" in metrics


def test_feature_importance(sample_ohlcv):
    features = build_feature_matrix(sample_ohlcv, horizons=[5])
    features = features.dropna(subset=["fwd_ret_5d_positive"])
    if len(features) < 50:
        pytest.skip("Not enough data")

    model = TabularModel(horizon=5, task="classification")
    model.train(features)
    fi = model.feature_importance()
    assert len(fi) > 0
    assert "feature" in fi.columns
    assert "importance" in fi.columns
