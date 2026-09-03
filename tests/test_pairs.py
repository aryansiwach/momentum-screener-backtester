import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momo.pairs import (
    compute_correlation,
    compute_hedge_ratio,
    compute_spread_zscore,
    scan_pair,
    compute_hedge_position,
)


def _correlated_pair(n=120, seed=1, diverge_last=0):
    # B tracks A plus i.i.d. (not cumulative) noise on the spread itself --
    # that makes log(A) - log(B) stationary around zero, the way a real
    # cointegrated pair's spread behaves. Two independent random walks
    # summed into the spread (the first version of this fixture) makes the
    # spread itself a random walk with no stable mean to measure "diverged"
    # against, so its z-score wanders past 2 std devs by chance even with
    # no real divergence -- that's a bad null case, not a code bug.
    rng = np.random.default_rng(seed)
    shared_shock = rng.normal(0, 0.01, n).cumsum()
    spread_noise = rng.normal(0, 0.003, n)
    a = 100 * np.exp(shared_shock)
    b = 100 * np.exp(shared_shock - spread_noise)
    if diverge_last:
        # push the last `diverge_last` points of A up, away from B, on purpose
        a[-diverge_last:] *= np.linspace(1.0, 1.25, diverge_last)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series(a, index=idx), pd.Series(b, index=idx)


def test_highly_correlated_series_score_near_one():
    a, b = _correlated_pair()
    corr = compute_correlation(a, b, lookback=60)
    assert corr > 0.8


def test_uncorrelated_series_score_near_zero():
    rng = np.random.default_rng(2)
    idx = pd.date_range("2024-01-01", periods=120, freq="B")
    a = pd.Series(100 * np.exp(rng.normal(0, 0.01, 120).cumsum()), index=idx)
    b = pd.Series(100 * np.exp(rng.normal(0, 0.01, 120).cumsum()), index=idx)
    corr = compute_correlation(a, b, lookback=60)
    assert abs(corr) < 0.5


def test_zscore_is_near_zero_when_pair_tracks_its_own_history():
    a, b = _correlated_pair(diverge_last=0)
    z = compute_spread_zscore(a, b, lookback=60)
    assert abs(z.iloc[-1]) < 1.5


def test_zscore_spikes_when_pair_diverges():
    a, b = _correlated_pair(diverge_last=15)
    z = compute_spread_zscore(a, b, lookback=60)
    assert z.iloc[-1] > 2.0


def test_scan_pair_flags_divergence_and_correct_direction():
    a, b = _correlated_pair(diverge_last=15)
    result = scan_pair("A", a, "B", b, lookback=60, z_threshold=2.0, min_correlation=0.5)
    assert result["signal"] == "divergence"
    # A ran up relative to B (positive z) -> short the rich one (A), long the cheap one (B)
    assert result["direction"] == {"long": "B", "short": "A"}


def test_scan_pair_no_signal_when_pair_stays_in_line():
    a, b = _correlated_pair(diverge_last=0)
    result = scan_pair("A", a, "B", b, lookback=60, z_threshold=2.0, min_correlation=0.5)
    assert result["signal"] == "no_signal"
    assert result["direction"] is None


def test_scan_pair_insufficient_data_for_short_series():
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    a = pd.Series([100, 101, 99, 102, 103], index=idx)
    b = pd.Series([50, 50.5, 49.5, 51, 51.5], index=idx)
    result = scan_pair("A", a, "B", b, lookback=60)
    assert result["signal"] == "insufficient_data"


def test_hedge_ratio_matches_known_linear_relationship():
    # B moves exactly 2x A's return each period -> hedge ratio (beta of A on B) should be ~0.5
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    rng = np.random.default_rng(3)
    b_returns = rng.normal(0, 0.01, 100)
    a_returns = b_returns * 2
    a = pd.Series(100 * np.exp(np.cumsum(a_returns)), index=idx)
    b = pd.Series(100 * np.exp(np.cumsum(b_returns)), index=idx)
    ratio = compute_hedge_ratio(a, b, lookback=100)
    assert abs(ratio - 2.0) < 0.05


def test_compute_hedge_position_scales_by_ratio():
    assert compute_hedge_position(1000.0, 0.5) == 500.0
    assert compute_hedge_position(1000.0, 1.5) == 1500.0
