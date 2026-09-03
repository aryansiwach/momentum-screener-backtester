"""Honestly tests whether the two proposed fixes actually help, against
real data -- vol-targeting on the 2022 bear regime (where the binary
regime filter failed), and pairs screening across a real multi-ticker
universe instead of one hand-picked pair."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from momo.orchestrator import MomentumPipeline
from momo.data_layer import YFDataLoader
from momo.backtest import Backtester
from momo.vol_targeting import apply_vol_targeting
from momo.pairs_screen import screen_pairs

WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "V",
    "UNH", "XOM", "MA", "HD", "COST", "PG", "NFLX", "AMD", "CRM", "ADBE",
    "BAC", "KO", "PEP", "TMO", "LIN", "WMT", "MCD", "ABT", "CSCO", "ORCL",
]


def main():
    print("=" * 70)
    print("Fix 1: GARCH vol-targeting vs unfiltered, 2022 bear regime")
    print("=" * 70)
    pipeline = MomentumPipeline(WATCHLIST, "2022-01-01", "2022-12-31", top_n=5,
                                 data_loader_cls=YFDataLoader)
    results = pipeline.run()
    unfiltered_stats = results["stats"]

    scaled_weights = apply_vol_targeting(results["weights"], results["prices"],
                                          target_daily_vol_pct=1.0, lookback=200, refit_every=10)
    _, _, scaled_stats = Backtester(costs_bps=5).run(results["prices"], scaled_weights)

    print(f"Unfiltered:     CAGR {unfiltered_stats['CAGR']*100:.2f}%  Sharpe {unfiltered_stats['Sharpe']:.2f}  "
          f"MaxDD {unfiltered_stats['Max Drawdown']*100:.2f}%")
    print(f"Vol-targeted:   CAGR {scaled_stats['CAGR']*100:.2f}%  Sharpe {scaled_stats['Sharpe']:.2f}  "
          f"MaxDD {scaled_stats['Max Drawdown']*100:.2f}%")

    print("\n" + "=" * 70)
    print("Fix 2: Pairs screening across the real watchlist (not one pair)")
    print("=" * 70)
    prices = YFDataLoader(WATCHLIST, "2023-01-01", "2024-12-31").load_adj_close()
    screen = screen_pairs(list(prices.columns), prices, min_correlation=0.75, lookback=60)
    print(f"Pairs tested: {screen['pairs_tested']}, corrected alpha: {screen.get('corrected_alpha')}")
    print(f"Survivors after correction: {len(screen.get('survivors_after_correction', []))}")
    if screen["pooled"]:
        print(f"Pooled Sharpe (annualized): {screen['pooled']['pooled_sharpe_annualized']}, "
              f"p={screen['pooled']['pooled_p_value']}, "
              f"significant at 5%: {screen['pooled']['pooled_significant_at_5pct']}")
    print("\nTop 5 individual pairs by p-value:")
    for r in screen["results"][:5]:
        print(f"  {r['ticker_a']}/{r['ticker_b']}: corr={r['correlation']}, "
              f"Sharpe={r['stats']['Sharpe']}, p={r['p_value']}")


if __name__ == "__main__":
    main()
