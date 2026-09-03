"""Pairs relative-value screen across the entire tradable universe, not
one hand-picked pair or a 30-name watchlist -- the legitimate next test
after the single-pair and small-watchlist results came back weak/unproven."""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from momo.universe import fetch_tradable_universe
from momo.data_layer import ChunkedLoader, AlpacaDataLoader
from momo.pairs_screen import find_correlated_pairs, screen_pairs


def main():
    print("Fetching tradable universe...")
    universe = fetch_tradable_universe()
    print(f"Universe size: {len(universe)}")

    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=760)

    print("Pulling price history (this is the same funnel used for full_market_scan)...")
    t0 = time.time()
    loader = ChunkedLoader(universe, start, end, inner_loader_cls=AlpacaDataLoader, chunk_size=200)
    prices = loader.load_adj_close()
    print(f"Fetched {prices.shape[1]} tickers x {prices.shape[0]} days in {time.time()-t0:.1f}s")

    print("\nFinding correlated candidates (vectorized)...")
    t0 = time.time()
    candidates = find_correlated_pairs(prices, min_correlation=0.85)
    print(f"Found {len(candidates)} candidate pairs at corr>=0.85 in {time.time()-t0:.1f}s")

    print("\nBacktesting top 200 candidates by correlation...")
    t0 = time.time()
    result = screen_pairs(list(prices.columns), prices, min_correlation=0.85,
                           lookback=60, max_pairs_to_backtest=200)
    print(f"Backtested {result['pairs_tested']} pairs (truncated: {result['truncated_to_max_pairs']}) "
          f"in {time.time()-t0:.1f}s")
    print(f"Corrected alpha: {result['corrected_alpha']}")
    print(f"Survivors after correction: {len(result['survivors_after_correction'])}")
    if result["pooled"]:
        p = result["pooled"]
        print(f"\nPooled across {p['n_pairs_pooled']} pairs: "
              f"Sharpe(ann)={p['pooled_sharpe_annualized']}, p={p['pooled_p_value']}, "
              f"significant at 5%: {p['pooled_significant_at_5pct']}")

    print("\nTop 10 individual pairs by p-value:")
    for r in result["results"][:10]:
        print(f"  {r['ticker_a']}/{r['ticker_b']}: corr={r['correlation']}, "
              f"Sharpe={r['stats']['Sharpe']}, CAGR={r['stats']['CAGR']}%, "
              f"trades={r['stats']['Num Trades']}, p={r['p_value']}")

    if result["survivors_after_correction"]:
        print("\nPairs surviving Bonferroni correction:")
        for r in result["survivors_after_correction"]:
            print(f"  {r['ticker_a']}/{r['ticker_b']}: p={r['p_value']}, stats={r['stats']}")


if __name__ == "__main__":
    main()
