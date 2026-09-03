import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momo.regime import compute_trend_state, current_regime, apply_regime_scaling


def _trending_series(n=250, direction=1):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    drift = 0.001 * direction
    prices = 100 * np.exp(np.cumsum(np.full(n, drift)))
    return pd.Series(prices, index=idx)


def test_uptrend_is_risk_on():
    prices = _trending_series(direction=1)
    result = current_regime(prices, sma_window=200)
    assert result["risk_on"] is True
    assert result["regime"] == "risk_on"


def test_downtrend_is_risk_off():
    prices = _trending_series(direction=-1)
    result = current_regime(prices, sma_window=200)
    assert result["risk_on"] is False
    assert result["regime"] == "risk_off"


def test_unknown_when_not_enough_history():
    prices = _trending_series(n=50, direction=1)  # shorter than the 200-day window
    result = current_regime(prices, sma_window=200)
    assert result["regime"] == "unknown"
    assert result["risk_on"] is None


def test_apply_regime_scaling_zeroes_weights_on_risk_off_days():
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    weights = pd.DataFrame({"AAPL": [0.5, 0.5, 0.5, 0.5, 0.5]}, index=idx)
    trend_state = pd.Series([True, True, False, False, True], index=idx)

    scaled = apply_regime_scaling(weights, trend_state, risk_off_scale=0.0)
    assert list(scaled["AAPL"]) == [0.5, 0.5, 0.0, 0.0, 0.5]


def test_apply_regime_scaling_partial_scale_on_risk_off():
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    weights = pd.DataFrame({"AAPL": [1.0, 1.0, 1.0]}, index=idx)
    trend_state = pd.Series([True, False, False], index=idx)

    scaled = apply_regime_scaling(weights, trend_state, risk_off_scale=0.3)
    assert list(scaled["AAPL"]) == [1.0, 0.3, 0.3]
