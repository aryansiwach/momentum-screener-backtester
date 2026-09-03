import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momo.significance import sharpe_significance_test, mean_return_ttest


def test_sharpe_test_flags_strong_positive_edge_as_significant():
    rng = np.random.default_rng(1)
    # daily mean 0.3%, std 1% -- a strong, real edge, large sample
    returns = pd.Series(rng.normal(0.003, 0.01, 500))
    result = sharpe_significance_test(returns)
    assert result["significant_at_5pct"] is True
    assert result["p_value"] < 0.05
    assert result["sharpe_annualized"] > 0


def test_sharpe_test_does_not_flag_pure_noise_as_significant():
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0.0, 0.01, 500))
    result = sharpe_significance_test(returns)
    assert result["significant_at_5pct"] is False


def test_sharpe_test_insufficient_data():
    result = sharpe_significance_test(pd.Series([0.01, -0.005, 0.002]))
    assert result["significant_at_5pct"] is None
    assert "observations" in result["reason"]


def test_mean_return_ttest_flags_real_positive_mean():
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(0.005, 0.01, 300))
    result = mean_return_ttest(returns)
    assert result["significant_at_5pct"] is True
    assert result["mean_return"] > 0


def test_mean_return_ttest_does_not_flag_zero_mean_noise():
    rng = np.random.default_rng(4)
    returns = pd.Series(rng.normal(0.0, 0.01, 300))
    result = mean_return_ttest(returns)
    assert result["significant_at_5pct"] is False
