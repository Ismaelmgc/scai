"""Anti-leak gate: catches a leaked feature, tolerates a market-exposure feature."""
import numpy as np
import pandas as pd
import pytest

from app.features import leak_checks as lk


def _panel(n_dates=80, n_names=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.bdate_range("2024-01-01", periods=n_dates):
        mkt = rng.normal(0, 0.05)                      # the day's market move
        beta = rng.uniform(0.5, 1.5, n_names)
        target = beta * mkt + rng.normal(0, 0.05, n_names)
        rows.append(pd.DataFrame({"date": d, "ticker": [f"T{i}" for i in range(n_names)],
                                  "target": target, "beta": beta,
                                  "noise": rng.normal(0, 1, n_names),
                                  "leaky": target + rng.normal(0, 0.02, n_names)}))
    return pd.concat(rows, ignore_index=True)


def test_planted_leak_is_caught():
    errors = lk.verify(_panel(), ["leaky"], "target")
    assert any("leaky" in e for e in errors)


def test_market_exposure_and_noise_pass():
    # beta has a large daily |rho| with the target but its sign flips with the market
    assert lk.verify(_panel(), ["beta", "noise"], "target") == []


def test_label_names_and_target_rejected():
    errs = lk.check_names(["fwd_ret_5d", "x_positive", "target", "ok"], "target")
    assert len(errs) == 3


def test_degenerate_column_rejected():
    df = _panel(n_dates=5)
    df["const"] = 1.0
    assert lk.check_degenerate(df, ["const"]) == ["const: degenerate (< 2 distinct values)"]


def test_assert_no_leak_raises():
    with pytest.raises(RuntimeError, match="anti-leak gate FAILED"):
        lk.assert_no_leak(_panel(), ["leaky"], "target")
