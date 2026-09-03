import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momo.significance import (
    sharpe_significance_test, mean_return_ttest,
    probabilistic_sharpe_ratio, probabilistic_sharpe_ratio_from_returns, deflated_sharpe_ratio,
)


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


def test_psr_flags_strong_positive_edge_as_significant():
    rng = np.random.default_rng(5)
    returns = pd.Series(rng.normal(0.003, 0.01, 500))
    result = probabilistic_sharpe_ratio_from_returns(returns)
    assert result["psr"] > 0.95
    assert result["significant_at_5pct"] is True


def test_psr_does_not_flag_pure_noise_as_significant():
    rng = np.random.default_rng(6)
    returns = pd.Series(rng.normal(0.0, 0.01, 500))
    result = probabilistic_sharpe_ratio_from_returns(returns)
    # zero true edge means the estimated Sharpe should sit right around
    # the sr_benchmark=0 default -- psr near 0.5, not near 1.
    assert 0.2 < result["psr"] < 0.8
    assert result["significant_at_5pct"] is False


def test_psr_insufficient_data():
    result = probabilistic_sharpe_ratio(sharpe_hat=1.0, n_obs=10)
    assert result["psr"] is None
    assert "observations" in result["reason"]


def test_psr_from_returns_matches_manually_computed_moments():
    from scipy import stats as scipy_stats
    rng = np.random.default_rng(7)
    # skewed, fat-tailed synthetic returns -- exactly the case a plain
    # normal-IID Sharpe test (Lo 2002) is not built to handle correctly.
    returns = pd.Series(rng.standard_t(df=3, size=400) * 0.01 + 0.001)
    via_wrapper = probabilistic_sharpe_ratio_from_returns(returns)

    sharpe_hat = float(returns.mean() / returns.std(ddof=1))
    skew = float(scipy_stats.skew(returns))
    kurtosis = float(scipy_stats.kurtosis(returns, fisher=False))
    via_manual = probabilistic_sharpe_ratio(sharpe_hat, len(returns), skew, kurtosis)

    assert via_wrapper["psr"] == via_manual["psr"]


def test_dsr_is_stricter_than_psr_at_zero_benchmark_once_multiple_trials_exist():
    # Same evaluated result (sharpe_hat, n_obs), but DSR additionally
    # prices in "this was the best of several trials" -- it should never
    # be easier to pass than the plain PSR-vs-zero test.
    sharpe_hat, n_obs = 0.15, 252
    psr_vs_zero = probabilistic_sharpe_ratio(sharpe_hat, n_obs)
    trial_sharpes = [0.15, -0.05, 0.02, -0.10, 0.08, 0.01, -0.03]
    dsr = deflated_sharpe_ratio(sharpe_hat, n_obs, trial_sharpes)
    assert dsr["sr_benchmark"] > 0
    assert dsr["dsr"] < psr_vs_zero["psr"]


def test_dsr_gets_stricter_as_more_trials_are_tried():
    # Same evaluated trial, same underlying spread of trial outcomes, but
    # more of them -- the expected best-by-chance Sharpe rises with
    # log(n_trials), so DSR should fall (or at least not rise).
    rng = np.random.default_rng(8)
    sharpe_hat, n_obs = 0.20, 252
    few_trials = list(rng.normal(0.0, 0.05, 5))
    many_trials = list(rng.normal(0.0, 0.05, 200))
    dsr_few = deflated_sharpe_ratio(sharpe_hat, n_obs, few_trials)
    dsr_many = deflated_sharpe_ratio(sharpe_hat, n_obs, many_trials)
    assert dsr_many["sr_benchmark"] > dsr_few["sr_benchmark"]
    assert dsr_many["dsr"] <= dsr_few["dsr"]


def test_dsr_requires_at_least_two_trials():
    result = deflated_sharpe_ratio(0.5, 252, [0.5])
    assert result["dsr"] is None
    assert "at least 2 trials" in result["reason"]


def test_dsr_zero_variance_across_trials():
    result = deflated_sharpe_ratio(0.5, 252, [0.5, 0.5, 0.5])
    assert result["dsr"] is None
    assert "variance" in result["reason"]
