"""Vectorized backtest: turns a weights DataFrame into daily portfolio
returns net of turnover-based transaction costs, plus standard performance
stats (CAGR, Sharpe, Sortino, max drawdown)."""

import pandas as pd
import numpy as np

def _ann_vol(returns, ppy=252):
    return returns.std() * np.sqrt(ppy)

def _sharpe(returns, rf=0.0, ppy=252):
    ex = returns - (rf/ppy)
    vol = _ann_vol(ex, ppy)
    return np.nan if vol == 0 else ex.mean()*ppy/vol

def _sortino(returns, rf=0.0, ppy=252):
    ex = returns - (rf/ppy)
    downside = ex.copy()
    downside[downside > 0] = 0
    dstd = downside.std() * np.sqrt(ppy)
    return np.nan if dstd == 0 else ex.mean()*ppy/dstd

def _max_dd(equity):
    peak = equity.cummax()
    dd = equity/peak - 1.0
    return dd.min()

def _cagr(returns, ppy=252):
    cum = (1+returns).prod()
    years = len(returns)/ppy
    return np.nan if years == 0 else cum**(1/years) - 1

class Backtester:
    def __init__(self, costs_bps=5, periods_per_year=252):
        self.costs = costs_bps/10000.0
        self.ppy = periods_per_year

    def run(self, prices: pd.DataFrame, weights: pd.DataFrame):
        # daily asset returns
        rets = prices.pct_change().fillna(0)

        # apply yesterday's weights to today's returns
        w = weights.shift().fillna(0)
        port_gross = (w * rets).sum(axis=1)

        # turnover-based transaction costs
        turnover = w.diff().abs().sum(axis=1).fillna(0)
        tc = turnover * self.costs

        port = port_gross - tc
        equity = (1+port).cumprod()

        stats = {
            "CAGR": _cagr(port, self.ppy),
            "Volatility": _ann_vol(port, self.ppy),
            "Sharpe": _sharpe(port, 0.0, self.ppy),
            "Sortino": _sortino(port, 0.0, self.ppy),
            "Max Drawdown": _max_dd(equity),
            "Avg Daily Turnover": turnover.mean()
        }
        return port, equity, pd.Series(stats)
