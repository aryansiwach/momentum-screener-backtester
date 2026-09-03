"""Runs the new econometric tools against real market data and reports
what they actually find -- not what a demo would want them to find."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from momo.data_layer import YFDataLoader
from momo.volatility import forecast_volatility
from momo.significance import sharpe_significance_test
from momo.timeseries import evaluate_return_predictability
from momo.pairs_strategy import backtest_pair


def main():
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=800)

    print("=" * 70)
    print("1) GARCH(1,1) vs historical-std volatility -- AAPL")
    print("=" * 70)
    prices = YFDataLoader(["AAPL"], start, end).load_adj_close()["AAPL"]
    returns = prices.pct_change(fill_method=None).dropna()
    garch = forecast_volatility(returns)
    hist_daily_vol = returns.std()
    print(f"Historical daily vol:     {hist_daily_vol*100:.3f}%  ({hist_daily_vol*(252**0.5)*100:.1f}% annualized)")
    print(f"GARCH(1,1) forecast vol:  {garch['forecast_daily_vol_pct']:.3f}%  "
          f"({garch['forecast_annualized_vol_pct']:.1f}% annualized)")
    print(f"Persistence (alpha+beta): {garch['persistence']} "
          f"({'slow-decaying, real clustering' if garch['persistence'] and garch['persistence'] > 0.9 else 'fast-decaying'})")

    print("\n" + "=" * 70)
    print("2) Return predictability (ARIMA) -- SPY")
    print("=" * 70)
    spy = YFDataLoader(["SPY"], start, end).load_adj_close()["SPY"]
    spy_returns = spy.pct_change(fill_method=None).dropna()
    pred = evaluate_return_predictability(spy_returns)
    print(f"Best order: {pred.get('best_order')}, predictable: {pred.get('predictable')}")
    print(f"Conclusion: {pred.get('conclusion')}")

    print("\n" + "=" * 70)
    print("3) Sharpe significance test -- momentum strategy, 2023-24 regime")
    print("=" * 70)
    from momo.orchestrator import MomentumPipeline
    watchlist = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "V"]
    pipeline = MomentumPipeline(watchlist, "2023-01-01", "2024-12-31", top_n=5, data_loader_cls=YFDataLoader)
    results = pipeline.run()
    sig = sharpe_significance_test(results["returns"])
    print(f"N={sig['n_observations']}, Sharpe(ann)={sig['sharpe_annualized']}, "
          f"t-stat={sig['t_stat']}, p={sig['p_value']}, significant at 5%: {sig['significant_at_5pct']}")

    print("\n" + "=" * 70)
    print("4) Pairs mean-reversion backtest -- V / MA")
    print("=" * 70)
    pair_prices = YFDataLoader(["V", "MA"], start, end).load_adj_close()
    pair_result = backtest_pair("V", pair_prices["V"], "MA", pair_prices["MA"], lookback=60)
    print(pair_result["stats"])
    pair_sig = sharpe_significance_test(pair_result["returns"])
    print(f"Pairs strategy significance: t-stat={pair_sig.get('t_stat')}, p={pair_sig.get('p_value')}, "
          f"significant at 5%: {pair_sig.get('significant_at_5pct')}")


if __name__ == "__main__":
    main()
