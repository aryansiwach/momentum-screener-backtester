import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momo.indicators import sma, ema, rsi, macd, stochastic, momentum_return


def _prices(n=100, tickers=("A", "B")):
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    data = {t: 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n))) for t in tickers}
    return pd.DataFrame(data, index=idx)


def test_sma_matches_pandas_rolling_mean():
    prices = _prices()
    out = sma(prices, window=10)
    expected = prices.rolling(10).mean()
    pd.testing.assert_frame_equal(out, expected)


def test_rsi_is_bounded_0_100():
    prices = _prices()
    out = rsi(prices, window=14).dropna()
    assert (out >= 0).all().all()
    assert (out <= 100).all().all()


def test_macd_histogram_is_macd_minus_signal():
    prices = _prices()
    macd_line, signal_line, hist = macd(prices)
    pd.testing.assert_frame_equal(hist, macd_line - signal_line)


def test_stochastic_k_is_bounded_0_100():
    prices = _prices()
    k, d = stochastic(prices, window=14)
    k = k.dropna()
    # small epsilon: floating-point division can push an exact high/low tie
    # a hair past 0/100
    assert (k >= -1e-9).all().all()
    assert (k <= 100 + 1e-9).all().all()


def test_momentum_return_matches_pct_change():
    prices = _prices()
    out = momentum_return(prices, period=21)
    expected = prices.pct_change(periods=21)
    pd.testing.assert_frame_equal(out, expected)
