"""Re-runs scripts/regime_backtest.py's exact 5-regime comparison, but with
the gap-mover filter that's live in production tonight (momo.quality.
historical_gap_mask) applied at every historical rebalance date, not just
"now". This is the honest next test for whether tonight's production
change (excluding single-day-spike names) actually helps or hurts, versus
just assuming it does because it looked better on one live snapshot.

Caveat stated up front, not buried: the 30-name watchlist is all liquid
large-caps that essentially never trigger a >25% single-day gap, so this
is a low-power sanity check (does the filter regress performance on stable
names -- it shouldn't) rather than a real test of the filter's benefit,
which only shows up in the small/micro-cap names the full-market scan
actually surfaces. That test needs a much larger, slower backtest across
the full tradable universe across all 5 regimes -- not run here.

Uses free Yahoo Finance data -- no keys needed. Run it yourself:
    venv/Scripts/python.exe scripts/regime_backtest_gap_filtered.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from momo.data_layer import YFDataLoader
from momo.screener import Screener
from momo.portfolio import PortfolioConstructor
from momo.backtest import Backtester
from momo.quality import historical_gap_mask

WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "V",
    "UNH", "XOM", "MA", "HD", "COST", "PG", "NFLX", "AMD", "CRM", "ADBE",
    "BAC", "KO", "PEP", "TMO", "LIN", "WMT", "MCD", "ABT", "CSCO", "ORCL",
]

REGIMES = [
    ("2019 steady bull",        "2019-01-01", "2019-12-31"),
    ("2020 COVID crash+recovery", "2020-01-01", "2020-12-31"),
    ("2022 rate-hike bear",     "2022-01-01", "2022-12-31"),
    ("2015-16 choppy/sideways", "2015-01-01", "2016-06-30"),
    ("2023-24 recent",          "2023-01-01", "2024-12-31"),
]


def run_regime(name, start, end, top_n=5, costs_bps=5, gap_lookback_days=63, max_daily_return=0.25):
    loader = YFDataLoader(WATCHLIST, start, end)
    try:
        prices = loader.load_adj_close()
    except Exception as exc:
        return {"regime": name, "error": str(exc)}

    scores = Screener().composite_scores(prices)
    gap_mask = historical_gap_mask(prices, lookback_days=gap_lookback_days, max_daily_return=max_daily_return)
    filtered_scores = scores.where(gap_mask.reindex_like(scores), other=float("-inf"))

    weights = PortfolioConstructor(top_n=top_n).construct_weights(filtered_scores)
    returns, equity, stats = Backtester(costs_bps=costs_bps).run(prices, weights)
    win_rate = float((returns > 0).mean()) if len(returns) else float("nan")

    # The rolling window's warm-up period (before min_periods is reached)
    # reads as False for every ticker -- that's correct masking behavior
    # (no real gap-window data yet means don't trust the score), but it
    # would massively overstate a naive False-count as if it were all real
    # gap detections. Only count exclusions where the rolling window
    # actually had enough data to make a real call.
    daily_returns = prices.pct_change()
    min_periods = max(5, int(gap_lookback_days * 0.5))
    has_enough_data = daily_returns.abs().rolling(gap_lookback_days, min_periods=min_periods).count() >= min_periods
    genuine_exclusions = (~gap_mask) & has_enough_data
    excluded_count = int(genuine_exclusions.reindex(index=scores.index, columns=scores.columns, fill_value=False).sum().sum())

    return {
        "regime": name,
        "start": start,
        "end": end,
        "tickers_used": prices.shape[1],
        "CAGR": round(float(stats["CAGR"]) * 100, 2),
        "Sharpe": round(float(stats["Sharpe"]), 2),
        "Max Drawdown": round(float(stats["Max Drawdown"]) * 100, 2),
        "Win Rate": round(win_rate * 100, 1),
        "gap_exclusions": excluded_count,
    }


def main():
    print(f"Gap-filtered composite momentum (top-5, {len(WATCHLIST)}-ticker watchlist) "
          f"across {len(REGIMES)} regimes...\n")
    print("NOTE: this watchlist is all liquid large-caps -- expect near-zero gap")
    print("exclusions here. This checks the filter doesn't regress a clean universe,")
    print("not whether it helps where it actually matters (small/micro-caps).\n")

    rows = []
    for name, start, end in REGIMES:
        print(f"  {name} ({start} to {end})...")
        rows.append(run_regime(name, start, end))

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))

    if "CAGR" in df.columns:
        valid = df.dropna(subset=["CAGR"])
        print(f"\nRegimes with a positive CAGR: {(valid['CAGR'] > 0).sum()} / {len(valid)}")
        print(f"Worst max drawdown across regimes: {valid['Max Drawdown'].min()}%")
        print(f"Average Sharpe across regimes: {valid['Sharpe'].mean():.2f}")
        print(f"Total gap-driven exclusions across all regimes: {int(valid['gap_exclusions'].sum())}")


if __name__ == "__main__":
    main()
