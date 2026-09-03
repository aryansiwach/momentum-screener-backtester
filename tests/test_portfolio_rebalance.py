import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momo.portfolio import PortfolioConstructor


def test_dropped_ticker_weight_resets_to_zero_at_next_rebalance():
    # A is best in week 1, C is best in week 2 -- with top_n=1 that must mean
    # A's weight goes back to 0 once C takes over, not stay at its old value
    # forever (the bug: ffill couldn't tell "explicitly zero" from "never
    # touched", so any ticker ever picked stayed nonzero for the rest of the
    # series once a later rebalance touched a different column).
    idx = pd.date_range("2024-01-01", periods=14, freq="B")
    score = pd.DataFrame(0.0, index=idx, columns=["A", "B", "C"])
    score.loc[idx[:7], "A"] = 1.0    # A wins the first week
    score.loc[idx[7:], "C"] = 1.0    # C wins the second week

    pc = PortfolioConstructor(top_n=1, short_n=0, rebalance="W")
    weights = pc.construct_weights(score)

    last_row = weights.iloc[-1]
    assert last_row["C"] == 1.0
    assert last_row["A"] == 0.0
    assert last_row["B"] == 0.0


def test_every_row_sums_to_one_or_zero_across_multiple_rebalances():
    # With a universe bigger than top_n across several rebalances, no row
    # should ever hold more names than top_n allows (the bug let stale picks
    # from earlier rebalances silently accumulate alongside new ones).
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    rng = np.random.default_rng(3)
    score = pd.DataFrame(rng.normal(size=(40, 6)), index=idx, columns=list("ABCDEF"))

    pc = PortfolioConstructor(top_n=2, short_n=0, rebalance="W")
    weights = pc.construct_weights(score)

    nonzero_counts = (weights > 0).sum(axis=1)
    active_rows = nonzero_counts[nonzero_counts > 0]
    assert (active_rows == 2).all()
    row_sums = weights[weights.index.isin(active_rows.index)].sum(axis=1)
    np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-9)
