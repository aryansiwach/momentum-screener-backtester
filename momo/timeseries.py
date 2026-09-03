"""ARIMA on daily returns (Box & Jenkins, 1970) -- tests whether a return
series has linear autocorrelation structure, i.e. whether past returns
have any power to predict future ones. For a genuinely liquid US equity,
the honest expectation under the efficient-market hypothesis is "no" --
finding no significant structure is the correct, useful result here, not
a failed model. This tests predictability, not whether to trade."""

import numpy as np
import pandas as pd


def fit_best_arima(returns: pd.Series, max_p: int = 3, max_q: int = 3):
    """Grid search over AR/MA orders (fixed d=0, applied to already-
    stationary returns, not prices), keeping the model with the lowest
    AIC. Orders that fail to converge are silently skipped, not scored.
    Returns the winning (order, result) plus the total number of
    coefficients tested across every order that *did* converge -- needed
    to correct for the multiple-comparison problem the search itself
    creates (see evaluate_return_predictability)."""
    from statsmodels.tsa.arima.model import ARIMA

    r = returns.dropna()
    best_aic = np.inf
    best_order = None
    best_result = None
    total_coefs_tested = 0
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                result = ARIMA(r, order=(p, 0, q)).fit()
            except Exception:
                continue
            # A non-converged MLE fit's coefficients and p-values aren't
            # trustworthy statistics at all -- don't let one win the AIC
            # comparison or count toward the multiple-testing correction.
            if not result.mle_retvals.get("converged", True):
                continue
            total_coefs_tested += sum(1 for k in result.pvalues.index if k not in ("const", "sigma2"))
            if result.aic < best_aic:
                best_aic = result.aic
                best_order = (p, 0, q)
                best_result = result
    return best_order, best_result, total_coefs_tested


def evaluate_return_predictability(returns: pd.Series, max_p: int = 3, max_q: int = 3) -> dict:
    """H0: no linear autocorrelation structure in the return series.
    Searching many (p, q) orders and then reading off the winning model's
    coefficient p-values is itself a form of data-snooping -- across enough
    attempted models, a false "significant" hit on pure noise is expected
    by chance even at the 5% level. Applies a Bonferroni correction across
    every coefficient actually tested during the search, not just the
    winning model's, so "predictable" reflects a real finding rather than
    a false positive from trying enough specifications."""
    r = returns.dropna()
    if len(r) < 60:
        return {"predictable": None, "reason": "fewer than 60 observations"}

    order, result, total_coefs_tested = fit_best_arima(r, max_p=max_p, max_q=max_q)
    if result is None:
        return {"predictable": None, "reason": "no ARIMA order converged"}

    corrected_alpha = 0.05 / max(1, total_coefs_tested)
    pvalues = result.pvalues
    coef_pvalues = {k: round(float(v), 4) for k, v in pvalues.items() if k not in ("const", "sigma2")}
    significant_terms = {k: v for k, v in coef_pvalues.items() if v < corrected_alpha}

    return {
        "best_order": order,
        "aic": round(float(result.aic), 2),
        "coefficient_pvalues": coef_pvalues,
        "coefficients_tested_across_search": total_coefs_tested,
        "bonferroni_corrected_alpha": round(corrected_alpha, 6),
        "significant_terms": significant_terms,
        "predictable": len(significant_terms) > 0,
        "conclusion": (
            "Found autocorrelation structure that survives correction for testing many model "
            "specifications -- worth investigating further."
            if significant_terms else
            "No structure survives correction for the number of specifications searched -- consistent "
            "with the return series behaving like a random walk, the expected result for a liquid US equity."
        ),
    }
