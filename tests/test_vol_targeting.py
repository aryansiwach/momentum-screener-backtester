import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momo.vol_targeting import compute_vol_scale, apply_vol_targeting


def _garch_returns(n=300, alpha=0.10, beta=0.85, omega=0.05, seed=1):
    rng = np.random.default_rng(seed)
    returns = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        returns[t] = rng.normal(0, np.sqrt(sigma2[t]))
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(returns / 100, index=idx)


def test_calm_market_scales_up_toward_target():
    # low-vol synthetic series -- scale should sit above 1.0 (target > forecast)
    calm = _garch_returns(omega=0.01, alpha=0.02, beta=0.10, seed=2)
    result = compute_vol_scale(calm, target_daily_vol_pct=1.5, min_scale=0.25, max_scale=2.0)
    assert result["scale"] > 1.0


def test_turbulent_market_scales_down():
    turbulent = _garch_returns(omega=0.5, alpha=0.15, beta=0.80, seed=3)
    result = compute_vol_scale(turbulent, target_daily_vol_pct=1.0, min_scale=0.25, max_scale=1.5)
    assert result["scale"] < 1.0


def test_scale_respects_min_and_max_bounds():
    turbulent = _garch_returns(omega=2.0, alpha=0.2, beta=0.7, seed=4)
    result = compute_vol_scale(turbulent, target_daily_vol_pct=0.1, min_scale=0.3, max_scale=1.5)
    assert result["scale"] >= 0.3

    calm = _garch_returns(omega=0.001, alpha=0.01, beta=0.05, seed=5)
    result = compute_vol_scale(calm, target_daily_vol_pct=5.0, min_scale=0.3, max_scale=1.5)
    assert result["scale"] <= 1.5


def test_insufficient_data_falls_back_to_scale_one():
    short = pd.Series(np.random.default_rng(1).normal(0, 0.01, 10))
    result = compute_vol_scale(short)
    assert result["scale"] == 1.0
    assert result["vol_model"] == "none"


def test_apply_vol_targeting_scales_weights_without_changing_shape():
    idx = pd.date_range("2023-01-01", periods=320, freq="B")
    tickers = ["A", "B"]
    rng = np.random.default_rng(1)
    prices = pd.DataFrame(
        {t: 100 * np.exp(rng.normal(0, 0.01, 320).cumsum()) for t in tickers}, index=idx
    )
    weights = pd.DataFrame(0.5, index=idx, columns=tickers)

    scaled = apply_vol_targeting(weights, prices, lookback=250, refit_every=10)
    assert scaled.shape == weights.shape
    # before the lookback warms up, scale defaults to 1.0 (unscaled)
    assert (scaled.iloc[:250] == weights.iloc[:250]).all().all()
