"""Tests for configuration system."""

import os
from unittest import mock

from app.config import AppMode, Settings


def test_default_settings():
    """Settings should load with sensible defaults."""
    s = Settings()
    assert s.mode == AppMode.DEMO
    assert s.seed == 42
    assert s.min_market_cap == 50_000_000
    assert s.exclude_otc is True
    assert s.exclude_adrs is True
    assert s.horizons == [1, 5, 10, 20]


def test_settings_paths():
    s = Settings(data_dir="/tmp/scai_test")
    assert str(s.raw_dir) == "/tmp/scai_test/raw"
    assert str(s.interim_dir) == "/tmp/scai_test/interim"
    assert str(s.processed_dir) == "/tmp/scai_test/processed"


def test_settings_from_env():
    with mock.patch.dict(os.environ, {"SCAI_SEED": "123", "SCAI_MODE": "production"}):
        s = Settings()
        assert s.seed == 123
        assert s.mode == AppMode.PRODUCTION
