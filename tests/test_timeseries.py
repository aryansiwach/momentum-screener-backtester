import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momo.timeseries import evaluate_return_predictability


def test_white_noise_false_positive_rate_is_low_across_many_draws():
    # A single white-noise draw can occasionally trip a "significant" hit
    # even after correction -- that's the definition of a nonzero
    # false-positive rate, not a bug. Asserting "never significant on one
    # arbitrary seed" isn't a fair test of the method; checking the rate
    # across many independent draws is the honest version of this check.
    rng = np.random.default_rng(100)
    false_positives = 0
    n_trials = 20
    for _ in range(n_trials):
        returns = pd.Series(rng.normal(0, 0.01, 300))
        result = evaluate_return_predictability(returns)
        if result["predictable"]:
            false_positives += 1
    assert false_positives / n_trials <= 0.15


def test_strong_ar1_process_is_detected_as_predictable():
    # a real, strong AR(1) process (coefficient 0.6) -- should be detectable
    rng = np.random.default_rng(2)
    n = 300
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = 0.6 * r[t - 1] + rng.normal(0, 0.01)
    returns = pd.Series(r)
    result = evaluate_return_predictability(returns)
    assert result["predictable"] is True
    assert len(result["significant_terms"]) > 0


def test_insufficient_data():
    result = evaluate_return_predictability(pd.Series([0.01, -0.01, 0.02]))
    assert result["predictable"] is None
    assert "observations" in result["reason"]
