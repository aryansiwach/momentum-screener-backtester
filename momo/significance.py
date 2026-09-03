"""Statistical significance testing for strategy returns. A backtested
Sharpe ratio is a sample statistic, not a fact -- these tests ask whether
it's distinguishable from zero (no skill) at the sample size actually
available, per Lo, A.W. (2002), "The Statistics of Sharpe Ratios",
Financial Analysts Journal 58(4). Small samples routinely produce Sharpe
ratios that look good and are not statistically different from noise."""

import numpy as np
import pandas as pd
from scipy import stats


def sharpe_significance_test(returns: pd.Series, periods_per_year: int = 252) -> dict:
    """H0: the true (population) Sharpe ratio is 0. Uses Lo (2002)'s IID
    asymptotic standard error, SE(SR) ~= sqrt(1/N) for the per-period
    Sharpe. This is the textbook approximation -- it assumes returns are
    IID, which real return series only roughly satisfy."""
    r = returns.dropna()
    n = len(r)
    if n < 30:
        return {"significant_at_5pct": None, "reason": "fewer than 30 observations -- not enough for a reliable test"}

    std = r.std(ddof=1)
    if std == 0:
        return {"significant_at_5pct": None, "reason": "zero variance in returns"}

    sharpe_period = float(r.mean() / std)
    se = np.sqrt(1.0 / n)
    t_stat = sharpe_period / se
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    return {
        "n_observations": n,
        "sharpe_period": round(sharpe_period, 4),
        "sharpe_annualized": round(sharpe_period * np.sqrt(periods_per_year), 4),
        "t_stat": round(float(t_stat), 3),
        "p_value": round(float(p_value), 4),
        "significant_at_5pct": bool(p_value < 0.05),
        "reason": None,
    }


def mean_return_ttest(returns: pd.Series) -> dict:
    """H0: the mean return is 0. A plain one-sample t-test -- the most
    basic check of whether a strategy's average return could plausibly
    just be zero plus noise."""
    r = returns.dropna()
    if len(r) < 2:
        return {"significant_at_5pct": None, "reason": "fewer than 2 observations"}

    t_stat, p_value = stats.ttest_1samp(r, popmean=0.0)
    return {
        "n_observations": len(r),
        "mean_return": round(float(r.mean()), 6),
        "t_stat": round(float(t_stat), 3),
        "p_value": round(float(p_value), 4),
        "significant_at_5pct": bool(p_value < 0.05),
    }
