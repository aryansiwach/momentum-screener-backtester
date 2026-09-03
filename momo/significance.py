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


_EULER_MASCHERONI = 0.5772156649015329


def probabilistic_sharpe_ratio(sharpe_hat: float, n_obs: int, skew: float = 0.0, kurtosis: float = 3.0,
                                sr_benchmark: float = 0.0) -> dict:
    """P(true Sharpe ratio > sr_benchmark), per Bailey & Lopez de Prado
    (2012), "The Sharpe Ratio Efficient Frontier", Journal of Risk 15(2).
    Unlike sharpe_significance_test above (which assumes normal IID
    returns via Lo (2002)'s asymptotic SE), this corrects the standard
    error of the Sharpe estimate for skewness and kurtosis, so it stays
    valid for the fat-tailed, skewed return series a real trading
    strategy actually produces.

    All inputs are PER-PERIOD (e.g. daily), not annualized -- annualizing
    first changes what n_obs means relative to the returns that produced
    the estimate, and silently invalidates the standard error below.
    skew/kurtosis default to 0/3 (normal distribution, kurtosis is RAW
    not excess) when the caller only has summary statistics, not the raw
    return series -- pass the actual moments when available via
    probabilistic_sharpe_ratio_from_returns."""
    if n_obs < 30:
        return {"psr": None, "reason": "fewer than 30 observations -- not enough for a reliable test"}

    variance_term = 1 + 0.5 * sharpe_hat ** 2 - skew * sharpe_hat + (kurtosis - 3) / 4 * sharpe_hat ** 2
    if variance_term <= 0:
        return {"psr": None, "reason": "non-positive variance term -- skew/kurtosis input is too extreme relative to sharpe_hat to trust"}

    se = np.sqrt(variance_term / (n_obs - 1))
    psr = float(stats.norm.cdf((sharpe_hat - sr_benchmark) / se))
    return {
        "psr": round(psr, 4),
        "sharpe_hat": round(sharpe_hat, 4),
        "sr_benchmark": round(sr_benchmark, 4),
        "n_obs": n_obs,
        "significant_at_5pct": bool(psr > 0.95),
        "reason": None,
    }


def probabilistic_sharpe_ratio_from_returns(returns: pd.Series, sr_benchmark: float = 0.0) -> dict:
    """Convenience wrapper: computes sharpe_hat, skew, and kurtosis directly
    from a per-period returns series instead of requiring the caller to
    supply them, then calls probabilistic_sharpe_ratio. Use this whenever
    the raw return series is actually available; it's strictly more
    accurate than the skew=0/kurtosis=3 default."""
    r = returns.dropna()
    n = len(r)
    if n < 30:
        return {"psr": None, "reason": "fewer than 30 observations -- not enough for a reliable test"}
    std = r.std(ddof=1)
    if std == 0:
        return {"psr": None, "reason": "zero variance in returns"}

    sharpe_hat = float(r.mean() / std)
    skew = float(stats.skew(r))
    kurtosis = float(stats.kurtosis(r, fisher=False))  # raw kurtosis (normal == 3), matches the formula's (kurtosis-3) term
    return probabilistic_sharpe_ratio(sharpe_hat, n, skew, kurtosis, sr_benchmark)


def deflated_sharpe_ratio(sharpe_hat: float, n_obs: int, trial_sharpes, skew: float = 0.0, kurtosis: float = 3.0) -> dict:
    """The Probabilistic Sharpe Ratio above answers "is this one result
    distinguishable from luck." It does not answer the harder, more
    honest question this project's audit specifically calls for: if
    several strategy variants were tried and the best one is what gets
    reported, is THAT one still distinguishable from the best of several
    equally-skill-less variants? Bailey & Lopez de Prado (2014), "The
    Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
    Overfitting, and Non-Normality", Journal of Portfolio Management
    40(5) -- the generalization of multiple-hypothesis correction (already
    used for the pairs screen's per-pair p-values, see momo/pairs_screen.py)
    to a Sharpe ratio benchmark instead of a p-value threshold.

    sharpe_hat/n_obs/skew/kurtosis describe the ONE trial being evaluated
    (typically the best-performing one), all PER-PERIOD, same convention
    as probabilistic_sharpe_ratio. trial_sharpes is the per-period Sharpe
    ratio actually achieved by EVERY trial run (including the one being
    evaluated) -- its variance sets how much of the best result should be
    attributed to chance ("with this many tries, how good would the best
    one look even with zero true skill"), so it must be the honest full
    set of trials actually attempted, not a cherry-picked subset."""
    n_trials = len(trial_sharpes)
    if n_trials < 2:
        return {"dsr": None, "reason": "need at least 2 trials to estimate a by-chance benchmark"}

    var_trials = float(np.var(trial_sharpes, ddof=1))
    if var_trials == 0:
        return {"dsr": None, "reason": "zero variance across trial Sharpe ratios"}

    # Expected maximum Sharpe ratio achievable by chance across n_trials
    # independent, zero-skill trials (the "False Strategy Theorem" bound).
    sr_benchmark = float(np.sqrt(var_trials) * (
        (1 - _EULER_MASCHERONI) * stats.norm.ppf(1 - 1.0 / n_trials)
        + _EULER_MASCHERONI * stats.norm.ppf(1 - 1.0 / n_trials * np.exp(-1))
    ))

    result = probabilistic_sharpe_ratio(sharpe_hat, n_obs, skew, kurtosis, sr_benchmark)
    if result["psr"] is None:
        return {"dsr": None, "reason": result["reason"], "sr_benchmark": round(sr_benchmark, 4), "n_trials": n_trials}
    result["dsr"] = result.pop("psr")
    result["n_trials"] = n_trials
    return result


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
