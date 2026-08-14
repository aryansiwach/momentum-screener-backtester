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
