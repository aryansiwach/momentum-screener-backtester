"""Runs the baseline momentum score (momo/screener.py) alongside three
genuinely different constructions (momo/momentum_variants.py) across the
IDENTICAL 45 rolling 12-month windows scripts/rolling_window_backtest.py
uses -- same watchlist, same date range, same step, same costs -- so the
four are directly comparable, not just individually plausible.

This is the "Demonstrated edge" research pass: each variant tests a real,
literature-backed economic mechanism (see momo/momentum_variants.py's
module docstring for the citations), not a re-tuned weight on the same
score. Every trial across all four constructions gets folded into ONE
multiple-testing correction (Deflated Sharpe Ratio, momo/significance.py)
against the honest full trial count -- not four separate, individually
easier tests.

Uses free Yahoo Finance data -- no keys needed. Run it yourself:
    venv/Scripts/python.exe scripts/momentum_variants_backtest.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from momo.data_layer import YFDataLoader

WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "V",
    "UNH", "XOM", "MA", "HD", "COST", "PG", "NFLX", "AMD", "CRM", "ADBE",
    "BAC", "KO", "PEP", "TMO", "LIN", "WMT", "MCD", "ABT", "CSCO", "ORCL",
]
MARKET_PROXY = "SPY"

FULL_START = "2014-06-01"
FULL_END = None
WINDOW_MONTHS = 12
STEP_MONTHS = 3


def run_window(prices_full, market_full, ticker_sectors, start, end, variant, top_n=5, costs_bps=5):
    from momo.screener import Screener
    from momo.momentum_variants import residual_momentum_scores, sector_relative_momentum_scores, lowvol_tilted_scores
    from momo.portfolio import PortfolioConstructor
    from momo.backtest import Backtester

    window_prices = prices_full.loc[start:end].dropna(axis=1, how="any")
    if window_prices.shape[1] < 5 or window_prices.shape[0] < 60:
        return None

    if variant == "baseline":
        scores = Screener().composite_scores(window_prices)
    elif variant == "residual_momentum":
        window_market = market_full.loc[start:end]
        scores = residual_momentum_scores(window_prices, window_market, beta_window=60)
    elif variant == "sector_relative":
        scores = sector_relative_momentum_scores(window_prices, ticker_sectors)
    elif variant == "lowvol_tilted":
        scores = lowvol_tilted_scores(window_prices)
    else:
        raise ValueError(variant)

    weights = PortfolioConstructor(top_n=top_n).construct_weights(scores)
    _, _, stats = Backtester(costs_bps=costs_bps).run(window_prices, weights)
    return {
        "variant": variant,
        "window_start": start.date().isoformat(),
        "window_end": end.date().isoformat(),
        "tickers": window_prices.shape[1],
        "CAGR": round(float(stats["CAGR"]) * 100, 2),
        "Sharpe": round(float(stats["Sharpe"]), 2),
        "Max Drawdown": round(float(stats["Max Drawdown"]) * 100, 2),
    }


def main():
    from momo.sectors import fetch_sectors

    end = pd.Timestamp(FULL_END) if FULL_END else pd.Timestamp.today().normalize()
    start = pd.Timestamp(FULL_START)
    print(f"Fetching full history {start.date()} to {end.date()} for {len(WATCHLIST)} tickers + {MARKET_PROXY}...")
    prices_full = YFDataLoader(WATCHLIST, start, end).load_adj_close()
    market_full = YFDataLoader([MARKET_PROXY], start, end).load_adj_close()[MARKET_PROXY]
    print(f"Got {prices_full.shape[0]} trading days x {prices_full.shape[1]} tickers.")

    print("Fetching sector classifications (current, applied across history -- see module docstring)...")
    ticker_sectors = fetch_sectors(WATCHLIST)
    print(f"Sectors resolved for {sum(1 for s in ticker_sectors.values() if s != 'Unknown')}/{len(WATCHLIST)} tickers.\n")

    first_window_start = start + pd.DateOffset(months=3)
    window_starts = pd.date_range(first_window_start, end - pd.DateOffset(months=WINDOW_MONTHS), freq=f"{STEP_MONTHS}MS")
    print(f"Running {len(window_starts)} rolling {WINDOW_MONTHS}-month windows x 4 constructions "
          f"(baseline, residual_momentum, sector_relative, lowvol_tilted)...\n")

    rows = []
    for variant in ["baseline", "residual_momentum", "sector_relative", "lowvol_tilted"]:
        for ws in window_starts:
            we = ws + pd.DateOffset(months=WINDOW_MONTHS)
            row = run_window(prices_full, market_full, ticker_sectors, ws, we, variant)
            if row:
                rows.append(row)
        print(f"  {variant}: done")

    df = pd.DataFrame(rows)
    out_path = os.path.join(os.path.dirname(__file__), "..", "progress", "momentum_variants_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}\n")

    print(df.to_string(index=False))
    print("\n=== Per-variant distribution across windows ===")
    for variant, g in df.groupby("variant"):
        print(f"{variant:20} CAGR mean {g['CAGR'].mean():6.2f}%  Sharpe mean {g['Sharpe'].mean():5.2f}  "
              f"median {g['Sharpe'].median():5.2f}  positive-Sharpe {100*(g['Sharpe']>0).mean():4.0f}%  "
              f"best {g['Sharpe'].max():5.2f}  worst-MaxDD {g['Max Drawdown'].min():6.2f}%")


if __name__ == "__main__":
    main()
