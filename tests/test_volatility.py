import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momo.volatility import forecast_volatility


def _simulate_garch11(n=600, omega=0.05, alpha=0.10, beta=0.85, seed=7):
    """Simulates a real GARCH(1,1) process (returns in percent, matching
    what forecast_volatility expects internally) with known parameters, so
    the fit can be checked against ground truth instead of just 'did it run'."""
    rng = np.random.default_rng(seed)
    returns = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)  # unconditional variance
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        returns[t] = rng.normal(0, np.sqrt(sigma2[t]))
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(returns / 100, index=idx)  # -> decimal scale, as real returns would be


def test_garch_recovers_persistence_from_known_process():
    # true persistence alpha+beta = 0.95 -- a real, high-persistence process
    returns = _simulate_garch11(alpha=0.10, beta=0.85)
    result = forecast_volatility(returns)
    assert result["converged"] is True
    # fitted persistence should land in the right ballpark of the true 0.95,
    # not exact (it's a maximum-likelihood estimate on one simulated path)
    assert 0.7 < result["persistence"] < 1.0


def test_garch_forecast_vol_is_positive_and_reasonable():
    returns = _simulate_garch11()
    result = forecast_volatility(returns)
    assert result["forecast_daily_vol_pct"] > 0
    assert result["forecast_annualized_vol_pct"] > result["forecast_daily_vol_pct"]


def test_garch_handles_insufficient_data():
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    returns = pd.Series(np.random.default_rng(1).normal(0, 0.01, 10), index=idx)
    result = forecast_volatility(returns)
    assert result["converged"] is False
    assert "observations" in result["reason"]


def test_low_persistence_process_reports_lower_persistence_than_high():
    low = forecast_volatility(_simulate_garch11(alpha=0.03, beta=0.10, seed=3))
    high = forecast_volatility(_simulate_garch11(alpha=0.10, beta=0.85, seed=3))
    assert low["persistence"] < high["persistence"]
