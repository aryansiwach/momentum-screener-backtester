import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momo.pairs_screen import screen_pairs


def _make_universe(n=300, seed=1):
    """Six tickers: three genuinely correlated pairs (A/B, C/D, E/F, each
    sharing their own random walk with small stationary spread noise) plus
    cross-group pairs that are NOT correlated -- screen_pairs should only
    backtest the correlated pairs, not all 15 combinations."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    prices = {}
    for group, (t1, t2) in enumerate([("A", "B"), ("C", "D"), ("E", "F")]):
        shared = rng.normal(0, 0.01, n).cumsum()
        noise = rng.normal(0, 0.003, n)
        prices[t1] = 100 * np.exp(shared)
        prices[t2] = 100 * np.exp(shared - noise)
    return pd.DataFrame(prices, index=idx)


def test_screen_pairs_only_tests_correlated_pairs():
    prices = _make_universe()
    result = screen_pairs(list(prices.columns), prices, min_correlation=0.6, lookback=40)
    tested_pairs = {(r["ticker_a"], r["ticker_b"]) for r in result["results"]}
    # cross-group pairs (e.g. A/C) should not have made it past the correlation filter
    assert ("A", "C") not in tested_pairs
    assert result["pairs_tested"] == 3


def test_screen_pairs_reports_pooled_result():
    prices = _make_universe()
    result = screen_pairs(list(prices.columns), prices, min_correlation=0.6, lookback=40)
    assert result["pooled"]["n_pairs_pooled"] == 3
    assert result["pooled"]["pooled_sharpe_annualized"] is not None


def test_screen_pairs_no_candidates_above_correlation_threshold():
    idx = pd.date_range("2023-01-01", periods=200, freq="B")
    rng = np.random.default_rng(9)
    prices = pd.DataFrame({
        "X": 100 * np.exp(rng.normal(0, 0.01, 200).cumsum()),
        "Y": 100 * np.exp(rng.normal(0, 0.01, 200).cumsum()),
    }, index=idx)
    result = screen_pairs(["X", "Y"], prices, min_correlation=0.95, lookback=40)
    assert result["pairs_tested"] == 0
    assert result["pooled"] is None


def test_corrected_alpha_shrinks_with_more_pairs_tested():
    prices = _make_universe()
    result = screen_pairs(list(prices.columns), prices, min_correlation=0.6, lookback=40)
    assert result["corrected_alpha"] == round(0.05 / 3, 5)
