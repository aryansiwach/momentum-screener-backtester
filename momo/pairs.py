"""Pairs relative-value scanning and hedge-ratio sizing. This is
statistical edge-seeking, not arbitrage and not a guarantee -- a wide
spread can widen further before it reverts, and a correlation that held
for 60 days can break down on day 61. The z-score threshold is a trigger
for looking closer, not a promise of profit."""

import numpy as np
import pandas as pd


def compute_correlation(price_a: pd.Series, price_b: pd.Series, lookback: int = 60) -> float:
    returns_a = price_a.pct_change().tail(lookback)
    returns_b = price_b.pct_change().tail(lookback)
    aligned = pd.concat([returns_a, returns_b], axis=1).dropna()
    if len(aligned) < 10:
        return float("nan")
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def compute_hedge_ratio(price_a: pd.Series, price_b: pd.Series, lookback: int = 60) -> float:
    """OLS beta of A's returns on B's -- how much of B's historical moves
    explain A's, not a 1:1 dollar hedge."""
    returns_a = price_a.pct_change().tail(lookback)
    returns_b = price_b.pct_change().tail(lookback)
    aligned = pd.concat([returns_a, returns_b], axis=1).dropna()
    if len(aligned) < 10:
        return float("nan")
    cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
    var = aligned.iloc[:, 1].var()
    if not var:
        return float("nan")
    return float(cov / var)


def compute_spread_zscore(price_a: pd.Series, price_b: pd.Series, lookback: int = 60) -> pd.Series:
    """Z-score of the log price ratio: how many standard deviations the
    pair's current relationship sits from its own recent historical norm."""
    log_ratio = np.log(price_a) - np.log(price_b)
    rolling_mean = log_ratio.rolling(lookback).mean()
    rolling_std = log_ratio.rolling(lookback).std()
    return (log_ratio - rolling_mean) / rolling_std


def scan_pair(ticker_a: str, price_a: pd.Series, ticker_b: str, price_b: pd.Series,
              lookback: int = 60, z_threshold: float = 2.0, min_correlation: float = 0.6) -> dict:
    correlation = compute_correlation(price_a, price_b, lookback)
    zscore_series = compute_spread_zscore(price_a, price_b, lookback)
    current_z = float(zscore_series.iloc[-1]) if len(zscore_series) else float("nan")
    hedge_ratio = compute_hedge_ratio(price_a, price_b, lookback)

    if pd.isna(current_z) or pd.isna(correlation):
        return {
            "ticker_a": ticker_a, "ticker_b": ticker_b,
            "signal": "insufficient_data", "correlation": None, "zscore": None,
            "hedge_ratio": None, "direction": None,
        }

    tradeable = correlation >= min_correlation and abs(current_z) >= z_threshold
    if tradeable:
        signal = "divergence"
        # positive z: A is rich relative to B -> short A, long B, betting the spread reverts
        direction = {"long": ticker_b, "short": ticker_a} if current_z > 0 else {"long": ticker_a, "short": ticker_b}
    else:
        signal = "no_signal"
        direction = None

    return {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "correlation": round(correlation, 3),
        "zscore": round(current_z, 3),
        "hedge_ratio": round(hedge_ratio, 3) if not pd.isna(hedge_ratio) else None,
        "signal": signal,
        "direction": direction,
    }


def compute_hedge_position(position_dollars: float, hedge_ratio: float) -> float:
    """Dollar size of the paired ticker to trade against a position, scaled
    by the historical hedge ratio. Reduces net exposure to the shared risk
    factor between the two -- it does not create profit, and a hedge sized
    on a stale ratio can under- or over-hedge the next move."""
    return round(position_dollars * hedge_ratio, 2)
