"""Rolling-window out-of-sample scan -- the honest piece worth borrowing
from the "genetic algorithm breeds strategies, tests on unseen data"
approach, without the part that reintroduces the exact multiple-testing
bias this project corrects for elsewhere (see momo/pairs_screen.py's
Bonferroni correction, and tonight's pairs-screen result: a pooled
p=0.045 that lost every individual survivor once corrected).

The 5 labeled regimes in scripts/regime_backtest.py (2019 bull, 2020
COVID, 2022 bear, 2015-16 chop, 2023-24 recent) were hand-picked because
they're recognizable market periods -- which also means they could be
an accidentally favorable selection. This script doesn't hand-pick
anything: it runs the SAME fixed-weight strategy (no parameters are
fit to data here, so there's nothing to overfit by re-optimizing per
window) across every overlapping 12-month window in the full available
history, stepped quarterly, and reports the full distribution of
outcomes -- not 5 point estimates.

Uses free Yahoo Finance data -- no keys needed. Run it yourself:
    venv/Scripts/python.exe scripts/rolling_window_backtest.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from momo.orchestrator import MomentumPipeline
from momo.data_layer import YFDataLoader

WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "V",
    "UNH", "XOM", "MA", "HD", "COST", "PG", "NFLX", "AMD", "CRM", "ADBE",
    "BAC", "KO", "PEP", "TMO", "LIN", "WMT", "MCD", "ABT", "CSCO", "ORCL",
]

FULL_START = "2014-06-01"  # a few months before the first rolling window needs
FULL_END = None            # None -> today
WINDOW_MONTHS = 12
STEP_MONTHS = 3


def run_window(prices_full: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, top_n=5, costs_bps=5):
    from momo.screener import Screener
    from momo.portfolio import PortfolioConstructor
    from momo.backtest import Backtester

    window_prices = prices_full.loc[start:end].dropna(axis=1, how="any")
    if window_prices.shape[1] < 5 or window_prices.shape[0] < 60:
        return None  # not enough tickers/history survived this window to mean anything

    scores = Screener().composite_scores(window_prices)
    weights = PortfolioConstructor(top_n=top_n).construct_weights(scores)
    _, _, stats = Backtester(costs_bps=costs_bps).run(window_prices, weights)
    return {
        "window_start": start.date().isoformat(),
        "window_end": end.date().isoformat(),
        "tickers": window_prices.shape[1],
        "CAGR": round(float(stats["CAGR"]) * 100, 2),
        "Sharpe": round(float(stats["Sharpe"]), 2),
        "Max Drawdown": round(float(stats["Max Drawdown"]) * 100, 2),
    }


def main():
    end = pd.Timestamp(FULL_END) if FULL_END else pd.Timestamp.today().normalize()
    start = pd.Timestamp(FULL_START)
    print(f"Fetching full history {start.date()} to {end.date()} for {len(WATCHLIST)} tickers...")
    prices_full = YFDataLoader(WATCHLIST, start, end).load_adj_close()
    print(f"Got {prices_full.shape[0]} trading days x {prices_full.shape[1]} tickers.\n")

    first_window_start = start + pd.DateOffset(months=3)  # give the screener's 63-day lookback room to warm up
    window_starts = pd.date_range(first_window_start, end - pd.DateOffset(months=WINDOW_MONTHS), freq=f"{STEP_MONTHS}MS")

    print(f"Running {len(window_starts)} rolling {WINDOW_MONTHS}-month windows, stepped every {STEP_MONTHS} months...\n")
    rows = []
    for ws in window_starts:
        we = ws + pd.DateOffset(months=WINDOW_MONTHS)
        row = run_window(prices_full, ws, we)
        if row:
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid windows produced -- check the date range / data availability.")
        return

    print(df.to_string(index=False))

    print(f"\n=== Distribution across {len(df)} rolling windows ===")
    print(f"CAGR:   mean {df['CAGR'].mean():.2f}%, median {df['CAGR'].median():.2f}%, "
          f"min {df['CAGR'].min():.2f}%, max {df['CAGR'].max():.2f}%")
    print(f"Sharpe: mean {df['Sharpe'].mean():.2f}, median {df['Sharpe'].median():.2f}, "
          f"min {df['Sharpe'].min():.2f}, max {df['Sharpe'].max():.2f}")
    print(f"Windows with positive CAGR: {(df['CAGR'] > 0).sum()} / {len(df)} "
          f"({100 * (df['CAGR'] > 0).mean():.0f}%)")
    print(f"Windows with positive Sharpe: {(df['Sharpe'] > 0).sum()} / {len(df)} "
          f"({100 * (df['Sharpe'] > 0).mean():.0f}%)")
    print(f"Worst single-window max drawdown: {df['Max Drawdown'].min():.2f}%")


if __name__ == "__main__":
    main()
