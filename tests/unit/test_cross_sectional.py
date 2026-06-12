"""Tests for cross-sectional feature transforms."""

import pandas as pd

from app.features.cross_sectional import (
    normalise_cross_section,
    rank_within_date,
    winsorise,
)


def test_rank_within_date():
    df = pd.DataFrame({
        "date": ["2024-01-01"] * 4,
        "val": [10, 20, 30, 40],
    })
    result = rank_within_date(df, ["val"])
    assert "val_rank" in result.columns
    assert result["val_rank"].iloc[0] == 0.25
    assert result["val_rank"].iloc[3] == 1.0


def test_winsorise():
    df = pd.DataFrame({
        "date": ["2024-01-01"] * 100,
        "val": list(range(100)),
    })
    result = winsorise(df, ["val"], limits=(0.05, 0.95))
    assert result["val"].min() >= 4
    assert result["val"].max() <= 95


def test_normalise_cross_section():
    df = pd.DataFrame({
        "date": ["2024-01-01"] * 4,
        "val": [10, 20, 30, 40],
    })
    result = normalise_cross_section(df, ["val"])
    assert abs(result["val"].mean()) < 1e-10
    # normalise_cross_section uses pandas std (ddof=1), so the normalised
    # column has unit sample std.
    assert abs(result["val"].std() - 1.0) < 0.01
