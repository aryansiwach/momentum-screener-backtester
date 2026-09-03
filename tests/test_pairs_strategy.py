import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momo.pairs_strategy import generate_pair_signals, backtest_pair


def _mean_reverting_pair(n=300, seed=5, n_cycles=6, amplitude=0.15, noise=0.003):
    """A pair whose spread genuinely oscillates (not a random walk) --
    the exact condition a pairs strategy is designed to exploit."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    shared_shock = rng.normal(0, 0.01, n).cumsum()
    spread = amplitude * np.sin(2 * np.pi * n_cycles * t / n) + rng.normal(0, noise, n)
    log_a = shared_shock
    log_b = shared_shock - spread
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(100 * np.exp(log_a), index=idx), pd.Series(100 * np.exp(log_b), index=idx)


def _diverging_pair(n=200, seed=6):
    """A pair that diverges once and never reverts -- the failure case a
    pairs trade's stop-loss (stop_z) exists to cut short."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    shared_shock = rng.normal(0, 0.01, n).cumsum()
    spread_noise = rng.normal(0, 0.003, n)
    a = 100 * np.exp(shared_shock)
    b = 100 * np.exp(shared_shock - spread_noise)
    # after day 80, A runs away from B and never comes back
    a[80:] *= np.linspace(1.0, 1.6, n - 80)
    return pd.Series(a, index=idx), pd.Series(b, index=idx)


def test_generate_pair_signals_produces_both_directions_on_oscillating_spread():
    a, b = _mean_reverting_pair()
    position = generate_pair_signals(a, b, lookback=40, entry_z=1.5, exit_z=0.5, stop_z=4.0)
    assert (position == 1).any()
    assert (position == -1).any()


def test_generate_pair_signals_flat_when_never_diverges():
    idx = pd.date_range("2023-01-01", periods=200, freq="B")
    rng = np.random.default_rng(1)
    flat_noise = rng.normal(0, 0.001, 200)
    a = pd.Series(100 + flat_noise, index=idx)
    b = pd.Series(100 + flat_noise, index=idx)  # identical -- spread never moves
    position = generate_pair_signals(a, b, lookback=40, entry_z=2.0)
    assert (position == 0).all()


def test_backtest_pair_profits_on_a_genuinely_mean_reverting_spread():
    a, b = _mean_reverting_pair()
    result = backtest_pair("A", a, "B", b, lookback=40, entry_z=1.5, exit_z=0.5, stop_z=4.0, costs_bps=1)
    assert result["stats"]["Num Trades"] > 0
    assert result["stats"]["CAGR"] > 0


def test_backtest_pair_stop_limits_loss_on_a_pair_that_never_reverts():
    a, b = _diverging_pair()
    with_stop = backtest_pair("A", a, "B", b, lookback=40, entry_z=1.5, stop_z=2.5, costs_bps=1)
    # a very wide/no stop should let the losing position ride out the full divergence
    without_stop = backtest_pair("A", a, "B", b, lookback=40, entry_z=1.5, stop_z=1000.0, costs_bps=1)
    assert with_stop["stats"]["Max Drawdown"] >= without_stop["stats"]["Max Drawdown"]


def test_backtest_pair_reports_zero_trades_when_never_diverges():
    idx = pd.date_range("2023-01-01", periods=200, freq="B")
    a = pd.Series(np.full(200, 100.0), index=idx)
    b = pd.Series(np.full(200, 100.0), index=idx)
    result = backtest_pair("A", a, "B", b, lookback=40)
    assert result["stats"]["Num Trades"] == 0
