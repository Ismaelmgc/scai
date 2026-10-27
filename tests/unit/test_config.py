"""Tests for configuration system."""

import os
from pathlib import Path
from unittest import mock

from app.config import Settings


def test_default_settings():
    """Settings should load with sensible defaults."""
    s = Settings()
    assert s.seed == 42
    assert s.min_market_cap == 50_000_000
    assert s.max_market_cap == 2_000_000_000
    assert s.exclude_otc is True


def test_settings_paths():
    # Compare Path objects (not strings) so the test is OS-independent.
    base = Path("/tmp/scai_test")
    s = Settings(data_dir=base)
    assert s.processed_dir == base / "processed"


def test_settings_from_env():
    with mock.patch.dict(os.environ, {"SCAI_SEED": "123", "SCAI_POLYGON_API_KEY": "k"}):
        s = Settings()
        assert s.seed == 123
        assert s.polygon_api_key == "k"
