"""Alternative momentum-score constructions, built to give the "Demonstrated
edge" grade (see the project report, Section 7) an honest, real shot rather
than leaving it untouched. Each variant tests a genuinely different economic
mechanism against the baseline (momo/screener.py:Screener) -- not a
re-tuning of the same score's weights, which would just be more of the same
multiple-testing risk the project already corrects for elsewhere.

Every variant here changes exactly ONE thing relative to the baseline, so
each economic claim is tested in isolation:

- residual_momentum_scores: momentum computed on returns residualized
  against a rolling single-factor market model (CAPM beta vs. a market
  proxy), not raw price returns. Blitz, Huij & Martens (2011), "Residual
  Momentum", Journal of Empirical Finance -- raw momentum partly reflects
  market-beta chasing, which is close to the mechanism Daniel & Moskowitz
  (2016) blame for momentum crashing hardest during bear-market rebounds
  (this project's own unresolved 2022 weakness). Stripping out beta before
  ranking should, if that story is right, reduce crash severity without
  necessarily giving up the underreaction-driven return.

- sector_relative_momentum_scores: the momentum component is percentile-
  ranked WITHIN each ticker's sector, not across the whole universe. A
  market-wide top-N list can just be one sector's macro move wearing a
  stock-picking costume -- momo.risk.check_sector_concentration already
  flags this after the fact on the live dashboard; this tests correcting
  for it at the scoring stage instead. Uses TODAY's sector classification
  applied across history -- a standard, defensible simplification (GICS
  sector reassignments are rare events, unlike fundamentals which change
  every quarter), not point-in-time data.

- lowvol_tilted_scores: the existing composite score, tilted toward lower
  trailing realized volatility. A price-based "quality" proxy, chosen
  specifically because it needs no fundamental data: OpenBB's free
  fundamental-metrics endpoint only returns TODAY's snapshot, not
  point-in-time history, so scoring a 2019 pick with today's P/E would be
  real look-ahead bias. Low-vol is computable from historical prices
  alone, sidestepping that trap entirely, and targets the same
  momentum-crash mechanism (high-vol names crash hardest) from a
  different, still-defensible angle (Frazzini & Pedersen 2014, "Betting
  Against Beta"; Baker, Bradley & Wurgler 2011 on the low-volatility
  anomaly).

All three return a wide score DataFrame in the same shape and "higher is
better" convention as Screener.composite_scores -- drop-in compatible
with PortfolioConstructor and Backtester, so they can run through the
exact same rolling-window methodology as the baseline for a fair,
apples-to-apples comparison."""

import pandas as pd

from momo.indicators import rsi, macd, sma, stochastic, momentum_return
from momo.screener import Screener


def _base_indicator_scores(prices: pd.DataFrame, lookbacks: dict):
    """The 4 non-momentum indicator components, exactly as Screener
    computes them -- factored out so each variant only overrides the
    momentum piece (or adds a tilt) instead of reimplementing RSI/MACD/
    SMA/stochastic scoring and risking it silently drifting from the
    baseline's own logic."""
    rsi_df = rsi(prices, lookbacks["rsi"]).rank(axis=1, pct=True)
    _, _, hist = macd(prices)
    macd_df = hist.rank(axis=1, pct=True)
    sma_df = (prices / sma(prices, lookbacks["sma"])).rank(axis=1, pct=True)
    stoch_k, _ = stochastic(prices, lookbacks["stoch"])
    stoch_df = stoch_k.rank(axis=1, pct=True)
    return rsi_df, macd_df, sma_df, stoch_df


def residual_momentum_scores(prices: pd.DataFrame, market_prices: pd.Series, weights=None, lookbacks=None,
                              beta_window: int = 126) -> pd.DataFrame:
    """market_prices: a single broad-market proxy (e.g. SPY), same date
    index as prices. beta_window: trailing window for the rolling
    single-factor beta -- 126 trading days (~6 months) is a standard,
    reasonably responsive choice, not fit to this data."""
    weights = weights or Screener().weights
    lookbacks = lookbacks or Screener().lookbacks

    stock_returns = prices.pct_change(fill_method=None)
    market_returns = market_prices.pct_change(fill_method=None)

    # Rolling beta per ticker, using only trailing data at every point --
    # no look-ahead. cov()/var() against a Series broadcast pairwise
    # across every column, matching pandas' documented rolling-cov
    # behavior with a Series argument.
    cov = stock_returns.rolling(beta_window).cov(market_returns)
    var = market_returns.rolling(beta_window).var()
    beta = cov.div(var, axis=0)
    residual_returns = stock_returns.sub(beta.mul(market_returns, axis=0))

    residual_cum = (1 + residual_returns).rolling(lookbacks["mom"]).apply(lambda x: x.prod() - 1, raw=True)
    mom = residual_cum.rank(axis=1, pct=True)

    rsi_df, macd_df, sma_df, stoch_df = _base_indicator_scores(prices, lookbacks)
    return (weights["mom"] * mom + weights["rsi"] * rsi_df + weights["macd"] * macd_df
            + weights["sma"] * sma_df + weights["stoch"] * stoch_df)


def sector_relative_momentum_scores(prices: pd.DataFrame, ticker_sectors: dict, weights=None, lookbacks=None) -> pd.DataFrame:
    weights = weights or Screener().weights
    lookbacks = lookbacks or Screener().lookbacks

    raw_mom = momentum_return(prices, lookbacks["mom"])
    sectors = pd.Series({t: ticker_sectors.get(t, "Unknown") for t in prices.columns})
    mom = raw_mom.copy()
    for sector in sectors.unique():
        cols = [c for c in sectors[sectors == sector].index if c in raw_mom.columns]
        if len(cols) < 2:
            # A sector group of one can't be meaningfully ranked against
            # itself -- a neutral 0.5 avoids handing it a spurious 1.0
            # (or 0.0) that a same-sized real group never could produce.
            mom[cols] = 0.5
            continue
        mom[cols] = raw_mom[cols].rank(axis=1, pct=True)

    rsi_df, macd_df, sma_df, stoch_df = _base_indicator_scores(prices, lookbacks)
    return (weights["mom"] * mom + weights["rsi"] * rsi_df + weights["macd"] * macd_df
            + weights["sma"] * sma_df + weights["stoch"] * stoch_df)


def lowvol_tilted_scores(prices: pd.DataFrame, weights=None, lookbacks=None, vol_window: int = 63,
                          vol_tilt_weight: float = 0.20) -> pd.DataFrame:
    """vol_tilt_weight is carved OUT of the existing composite score
    proportionally (not added on top of it), so the blend stays a genuine
    reallocation of belief between "momentum says buy" and "this name is
    calm enough not to be crash-prone" -- not free extra weight that lets
    the score exceed what either component alone could produce."""
    weights = weights or Screener().weights
    lookbacks = lookbacks or Screener().lookbacks

    base_score = Screener(weights=weights, lookbacks=lookbacks).composite_scores(prices)

    realized_vol = prices.pct_change(fill_method=None).rolling(vol_window).std()
    lowvol_rank = (-realized_vol).rank(axis=1, pct=True)  # lower vol -> higher rank

    return (1 - vol_tilt_weight) * base_score + vol_tilt_weight * lowvol_rank
