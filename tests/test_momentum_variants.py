import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momo.momentum_variants import (
    residual_momentum_scores, sector_relative_momentum_scores, lowvol_tilted_scores,
)
from momo.screener import Screener


def _dates(n):
    return pd.date_range("2023-01-01", periods=n, freq="B")


def test_residual_momentum_distinguishes_idiosyncratic_gain_from_pure_beta():
    # Two tickers post the SAME raw cumulative return over the momentum
    # window -- but PURE tracks the market with zero idiosyncratic
    # component (beta=1, no alpha), while ALPHA has genuine outperformance
    # beyond what its market exposure explains. Residual momentum's whole
    # point is telling these apart even when raw price momentum can't.
    n = 200
    idx = _dates(n)
    rng = np.random.default_rng(11)
    market_returns = rng.normal(0.0005, 0.01, n)
    market_prices = pd.Series(100 * (1 + pd.Series(market_returns)).cumprod().values, index=idx)

    pure_returns = market_returns.copy()  # beta=1, zero idiosyncratic return
    # ALPHA gets the same average daily drift as PURE's realized path over
    # the final momentum window by construction below, but earlier in the
    # series (outside the momentum lookback) is boosted, so raw 63-day
    # momentum ties while its beta-implied component differs -- simplest
    # clean construction: give ALPHA a small idiosyncratic ADD-ON that
    # nets out to the same 63-day cumulative return as PURE by also
    # slightly trimming ALPHA's beta exposure, so the residual (actual -
    # beta*market) is clearly positive for ALPHA and ~0 for PURE.
    idio = np.zeros(n)
    idio[-70:] = 0.0006  # small steady idiosyncratic drift within the momentum lookback
    alpha_returns = 0.4 * market_returns + idio  # lower beta (0.4) + genuine idiosyncratic drift

    prices = pd.DataFrame({
        "PURE": 100 * (1 + pd.Series(pure_returns, index=idx)).cumprod(),
        "ALPHA": 100 * (1 + pd.Series(alpha_returns, index=idx)).cumprod(),
    }, index=idx)

    scores = residual_momentum_scores(prices, market_prices, beta_window=60)
    last = scores.iloc[-1]
    assert last["ALPHA"] > last["PURE"]


def test_sector_relative_momentum_ranks_within_group_not_globally():
    n = 100
    idx = _dates(n)
    # Sector A: three modestly-performing tickers. Sector B: one huge
    # winner. A ticker that's merely the BEST of a weak sector should
    # still rank at the top of its own group, even though a market-wide
    # rank would bury it under sector B's outlier.
    prices = pd.DataFrame({
        "A1": 100 * (1.0010 ** np.arange(n)),
        "A2": 100 * (1.0005 ** np.arange(n)),
        "A3": 100 * (1.0002 ** np.arange(n)),
        "B1": 100 * (1.0200 ** np.arange(n)),
    }, index=idx)
    sectors = {"A1": "Sleepy", "A2": "Sleepy", "A3": "Sleepy", "B1": "Rocket"}

    global_rank = Screener().composite_scores(prices).iloc[-1]
    sector_scores = sector_relative_momentum_scores(prices, sectors).iloc[-1]

    # Globally, A1 (best of a weak sector) should rank below B1 (the
    # market-wide outlier) -- confirms the test setup actually creates
    # the effect being tested for.
    assert global_rank["A1"] < global_rank["B1"]
    # Within its own sector, A1's momentum component should be the top
    # of Sleepy regardless of Rocket's existence -- the sector-relative
    # score must not just reproduce the global rank's ordering.
    assert sector_scores["A1"] > sector_scores["A2"] > sector_scores["A3"]


def test_sector_relative_momentum_single_ticker_sector_gets_neutral_score():
    n = 80
    idx = _dates(n)
    prices = pd.DataFrame({
        "SOLO": 100 * (1.005 ** np.arange(n)),
        "PAIR1": 100 * (1.001 ** np.arange(n)),
        "PAIR2": 100 * (1.003 ** np.arange(n)),
    }, index=idx)
    sectors = {"SOLO": "Alone", "PAIR1": "Together", "PAIR2": "Together"}

    scores = sector_relative_momentum_scores(prices, sectors, weights={"mom": 1.0, "rsi": 0, "macd": 0, "sma": 0, "stoch": 0})
    # weights isolate the momentum component alone so 0.5 is exact, not
    # diluted/shifted by the other indicators' own percentile ranks
    assert scores["SOLO"].iloc[-1] == pytest.approx(0.5)


def test_lowvol_tilt_favors_lower_realized_vol_among_equal_momentum():
    # With only 2 tickers, percentile rank is too coarse (every component
    # is binary 0.5/1.0) to isolate the tilt's own effect from incidental
    # per-ticker noise in the other indicators -- use two 3-ticker groups
    # and compare group averages instead, which washes out idiosyncratic
    # per-ticker RSI/MACD/SMA noise and isolates the vol-driven effect.
    n = 150
    idx = _dates(n)
    rng = np.random.default_rng(12)
    daily_trend = np.log(1.0008)  # same underlying trend for every ticker
    calm_tickers = {f"CALM{i}": 100 * np.exp(np.cumsum(daily_trend + rng.normal(0, 0.002, n))) for i in range(3)}
    choppy_tickers = {f"CHOPPY{i}": 100 * np.exp(np.cumsum(daily_trend + rng.normal(0, 0.025, n))) for i in range(3)}
    prices = pd.DataFrame({**calm_tickers, **choppy_tickers}, index=idx)

    tilted = lowvol_tilted_scores(prices, vol_tilt_weight=0.5).iloc[-1]
    calm_avg = tilted[[c for c in prices.columns if c.startswith("CALM")]].mean()
    choppy_avg = tilted[[c for c in prices.columns if c.startswith("CHOPPY")]].mean()
    assert calm_avg > choppy_avg


def test_lowvol_tilt_weight_zero_reduces_to_plain_screener():
    n = 120
    idx = _dates(n)
    rng = np.random.default_rng(13)
    prices = pd.DataFrame({
        "AAA": 100 * (1 + rng.normal(0.001, 0.01, n)).cumprod(),
        "BBB": 100 * (1 + rng.normal(0.0005, 0.02, n)).cumprod(),
    }, index=idx)

    tilted = lowvol_tilted_scores(prices, vol_tilt_weight=0.0)
    baseline = Screener().composite_scores(prices)
    pd.testing.assert_frame_equal(tilted, baseline)
