"""Vectorized technical indicators, each operating on a wide DataFrame of
prices (columns = tickers) and returning a same-shaped DataFrame."""

import pandas as pd

def sma(prices: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    return prices.rolling(window).mean()

def ema(prices: pd.DataFrame, span: int = 12) -> pd.DataFrame:
    return prices.ewm(span=span, adjust=False).mean()

def rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/window, adjust=False).mean()
    roll_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / roll_down
    return 100 - (100/(1+rs))

def macd(prices: pd.DataFrame, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def stochastic(prices: pd.DataFrame, window: int = 14):
    low = prices.rolling(window).min()
    high = prices.rolling(window).max()
    k = 100*(prices - low)/(high - low)
    d = k.rolling(3).mean()
    return k, d

def momentum_return(prices: pd.DataFrame, period: int = 63):
    return prices.pct_change(periods=period, fill_method=None)
