import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momo.quality import filter_gap_movers, filter_illiquid, historical_gap_mask


def _prices(series_by_ticker, n=70):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(series_by_ticker, index=idx)


def test_steady_climber_passes_filter():
    # +0.5%/day compounded for 70 days -- a real sustained trend, no day
    # anywhere close to the 25% single-day threshold.
    steady = 100 * (1.005 ** np.arange(70))
    prices = _prices({"STEADY": steady})
    mask = filter_gap_movers(prices, lookback_days=63, max_daily_return=0.25)
    assert mask["STEADY"]


def test_single_day_gap_is_excluded():
    # Flat, then one session gaps up 30x (like a reverse split / halt
    # reopen), then flat again -- the failure mode this filter exists for.
    flat = np.full(70, 1.0)
    flat[50] = 30.0
    flat[51:] = 30.0
    prices = _prices({"GAPPER": flat})
    mask = filter_gap_movers(prices, lookback_days=63, max_daily_return=0.25)
    assert not mask["GAPPER"]


def test_sawtooth_repeated_spikes_is_excluded():
    # Alternating flat/spike/flat/spike -- looks "high momentum" in
    # aggregate but every leg is a discontinuous jump, not a trend.
    vals = np.tile([1.0, 1.0, 20.0, 1.0], 18)[:70]
    prices = _prices({"SAWTOOTH": vals})
    mask = filter_gap_movers(prices, lookback_days=63, max_daily_return=0.25)
    assert not mask["SAWTOOTH"]


def test_insufficient_history_is_excluded_not_assumed_smooth():
    prices = _prices({"THIN": [np.nan] * 65 + [1.0, 1.0, 1.0, 1.0, 1.0]})
    mask = filter_gap_movers(prices, lookback_days=63, max_daily_return=0.25)
    assert not mask["THIN"]


def _ohlcv(close, volume, n=25):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"close": close, "volume": volume}, index=idx)


def test_liquid_name_passes_liquidity_filter():
    # $50 * 500k shares = $25M/day average, well above a $5M floor.
    ohlcv = {"LIQUID": _ohlcv([50.0] * 25, [500_000] * 25)}
    result = filter_illiquid(ohlcv, lookback_days=20, min_dollar_volume=5_000_000)
    assert result["LIQUID"]


def test_thin_name_fails_liquidity_filter():
    # $8 * 5k shares = $40k/day average -- exactly the microcap/thin-float
    # profile prone to a small order moving the price a lot.
    ohlcv = {"THIN": _ohlcv([8.0] * 25, [5_000] * 25)}
    result = filter_illiquid(ohlcv, lookback_days=20, min_dollar_volume=5_000_000)
    assert not result["THIN"]


def test_missing_ohlcv_fails_liquidity_filter():
    ohlcv = {"NODATA": pd.DataFrame(columns=["close", "volume"])}
    result = filter_illiquid(ohlcv, lookback_days=20, min_dollar_volume=5_000_000)
    assert not result["NODATA"]


def test_historical_gap_mask_matches_point_in_time_filter_gap_movers():
    # historical_gap_mask is the vectorized, whole-history version of
    # filter_gap_movers -- its value on the LAST date should agree with
    # calling filter_gap_movers on that same trailing window.
    steady = 100 * (1.005 ** np.arange(80))
    gapper = np.full(80, 1.0)
    gapper[50] = 30.0
    gapper[51:] = 30.0
    prices = _prices({"STEADY": steady, "GAPPER": gapper}, n=80)

    mask = historical_gap_mask(prices, lookback_days=63, max_daily_return=0.25)
    single_point = filter_gap_movers(prices, lookback_days=63, max_daily_return=0.25)

    assert bool(mask["STEADY"].iloc[-1]) == bool(single_point["STEADY"])
    assert bool(mask["GAPPER"].iloc[-1]) == bool(single_point["GAPPER"])


def test_historical_gap_mask_flips_false_only_after_the_gap_happens():
    gapper = np.full(80, 1.0)
    gapper[50] = 30.0
    gapper[51:] = 30.0
    prices = _prices({"GAPPER": gapper}, n=80)

    mask = historical_gap_mask(prices, lookback_days=63, max_daily_return=0.25)
    # Before the gap (day 49), nothing anomalous has happened yet in its
    # own trailing window -- should still read True (or unknown/NaN this
    # early, never a false positive).
    assert bool(mask["GAPPER"].iloc[49]) is True
    # The day the gap itself lands, the 25% threshold is breached.
    assert bool(mask["GAPPER"].iloc[50]) is False
    # It stays flagged while the gap day remains inside the trailing window.
    assert bool(mask["GAPPER"].iloc[60]) is False
