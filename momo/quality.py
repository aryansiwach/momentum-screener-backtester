"""Distinguish sustained trends from single-day gap events (halts, reverse
splits, short squeezes, buyout/news pops). A composite momentum score can't
tell "climbed steadily for two months" apart from "flat for two months, then
+3000% in one session" -- both end up with a similarly high score, but only
one represents a repeatable trend a momentum strategy is actually designed
to capture. This is standard practice in momentum research (e.g. excluding
extreme single-day return names), not a filter invented for one ticker.
"""

import pandas as pd


def filter_gap_movers(prices: pd.DataFrame, lookback_days: int = 63, max_daily_return: float = 0.25) -> pd.Series:
    """Boolean mask over prices.columns: True where the ticker's largest
    single-day |return| within the trailing `lookback_days` stays under
    `max_daily_return` -- i.e. it passes as a "smooth" mover. A ticker with
    too little history to evaluate is excluded (mask False), not assumed
    innocent -- pandas' skipna default would otherwise judge a mostly-NaN
    column "smooth" purely because there's nothing to compare."""
    recent = prices.tail(lookback_days)
    daily_returns = recent.pct_change()
    enough_history = daily_returns.notna().sum() >= max(5, int(lookback_days * 0.5))
    max_abs_daily_return = daily_returns.abs().max()
    return enough_history & (max_abs_daily_return <= max_daily_return)


def historical_gap_mask(prices: pd.DataFrame, lookback_days: int = 63, max_daily_return: float = 0.25) -> pd.DataFrame:
    """Point-in-time version of filter_gap_movers: a same-shaped boolean
    DataFrame where cell (date, ticker) is True if, using only data known
    as of that date, the ticker's trailing lookback_days had no single-day
    |return| over max_daily_return. Built for backtesting -- this is what
    lets a historical rebalance apply the same gap filter that's live in
    production today, instead of only ever checking "now"."""
    daily_returns = prices.pct_change()
    min_periods = max(5, int(lookback_days * 0.5))
    rolling_max_abs = daily_returns.abs().rolling(lookback_days, min_periods=min_periods).max()
    return rolling_max_abs <= max_daily_return


def filter_illiquid(ohlcv_by_ticker: dict, lookback_days: int = 20, min_dollar_volume: float = 5_000_000) -> dict:
    """ticker -> True if its trailing average daily dollar volume (close *
    volume) clears min_dollar_volume. Thin, low-volume names are exactly
    the ones prone to violent single-day gaps (a small number of shares
    moving the price a lot) and to slippage that makes a "$100 suggested
    allocation" harder to actually fill at the quoted price -- liquidity is
    a risk control, not just a convenience filter."""
    out = {}
    for ticker, df in ohlcv_by_ticker.items():
        recent = df.tail(lookback_days)
        if recent.empty:
            out[ticker] = False
            continue
        dollar_volume = (recent["close"] * recent["volume"]).mean()
        out[ticker] = bool(dollar_volume >= min_dollar_volume)
    return out
