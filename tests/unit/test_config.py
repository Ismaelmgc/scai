"""Tests for configuration system."""

import os
from pathlib import Path
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
    # Compare Path objects (not strings) so the test is OS-independent.
    base = Path("/tmp/scai_test")
    s = Settings(data_dir=base)
    assert s.raw_dir == base / "raw"
    assert s.interim_dir == base / "interim"
    assert s.processed_dir == base / "processed"


def test_settings_from_env():
    with mock.patch.dict(os.environ, {"SCAI_SEED": "123", "SCAI_MODE": "production"}):
        s = Settings()
        assert s.seed == 123
        assert s.mode == AppMode.PRODUCTION
