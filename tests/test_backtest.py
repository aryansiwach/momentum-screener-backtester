import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momo.backtest import Backtester
from momo.portfolio import PortfolioConstructor


def _toy_prices():
    idx = pd.date_range("2023-01-02", periods=10, freq="B")
    # A goes up steadily, B is flat
    a = 100 * (1.01 ** np.arange(10))
    b = np.full(10, 100.0)
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def test_zero_weights_produce_flat_equity():
    prices = _toy_prices()
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    bt = Backtester(costs_bps=0)
    port, equity, stats = bt.run(prices, weights)
    assert (port == 0).all()
    assert (equity == 1.0).all()


def test_all_in_winner_grows_equity_more_than_flat_ticker():
    prices = _toy_prices()
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights["A"] = 1.0
    bt = Backtester(costs_bps=0)
    _, equity, _ = bt.run(prices, weights)
    assert equity.iloc[-1] > 1.0


def test_transaction_costs_reduce_returns_when_turnover_is_positive():
    prices = _toy_prices()
    # weights that flip every period generate turnover
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights.loc[weights.index[::2], "A"] = 1.0
    weights.loc[weights.index[1::2], "B"] = 1.0

    bt_no_cost = Backtester(costs_bps=0)
    bt_with_cost = Backtester(costs_bps=50)

    _, equity_no_cost, _ = bt_no_cost.run(prices, weights)
    _, equity_with_cost, _ = bt_with_cost.run(prices, weights)

    assert equity_with_cost.iloc[-1] <= equity_no_cost.iloc[-1]


def test_per_ticker_costs_charge_the_thin_name_more():
    prices = _toy_prices()
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights.loc[weights.index[::2], "A"] = 1.0
    weights.loc[weights.index[1::2], "B"] = 1.0

    # A is cheap to trade (5bps), B is expensive (100bps) -- same turnover
    # pattern, but B's leg of the round-trips should cost noticeably more.
    bt_cheap_a = Backtester(costs_bps={"A": 5, "B": 5})
    bt_expensive_b = Backtester(costs_bps={"A": 5, "B": 100})

    _, equity_cheap, _ = bt_cheap_a.run(prices, weights)
    _, equity_expensive, _ = bt_expensive_b.run(prices, weights)

    assert equity_expensive.iloc[-1] < equity_cheap.iloc[-1]


def test_per_ticker_costs_match_flat_costs_when_uniform():
    prices = _toy_prices()
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights.loc[weights.index[::2], "A"] = 1.0
    weights.loc[weights.index[1::2], "B"] = 1.0

    bt_flat = Backtester(costs_bps=25)
    bt_dict = Backtester(costs_bps={"A": 25, "B": 25})

    port_flat, _, _ = bt_flat.run(prices, weights)
    port_dict, _, _ = bt_dict.run(prices, weights)

    pd.testing.assert_series_equal(port_flat, port_dict)


def test_missing_ticker_in_cost_dict_falls_back_to_default():
    prices = _toy_prices()
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights.loc[weights.index[::2], "A"] = 1.0
    weights.loc[weights.index[1::2], "B"] = 1.0

    # Only A specified -- B should fall back to the 5bps default, not error
    # or silently charge nothing.
    bt = Backtester(costs_bps={"A": 5})
    port, equity, stats = bt.run(prices, weights)
    assert not port.isna().any()


def test_portfolio_constructor_long_only_weights_sum_to_one_at_rebalance():
    prices = _toy_prices()
    score = pd.DataFrame(
        np.random.default_rng(1).normal(size=(10, 2)), index=prices.index, columns=["A", "B"]
    )
    pc = PortfolioConstructor(top_n=2, short_n=0, rebalance="W")
    weights = pc.construct_weights(score)
    nonzero_rows = weights[(weights.abs().sum(axis=1) > 0)]
    assert not nonzero_rows.empty
    np.testing.assert_allclose(nonzero_rows.sum(axis=1).values, 1.0, atol=1e-9)
