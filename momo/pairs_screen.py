"""Screens many candidate pairs at once rather than judging the strategy
by one hand-picked pair -- picking whichever pair happens to look
significant after the fact is data-snooping across pairs, the same
mistake already caught and fixed in the ARIMA order search. This applies
the same discipline: test many, correct for how many were tested, and
report the pooled result honestly, including if nothing survives."""

import pandas as pd
import numpy as np

from momo.pairs_strategy import backtest_pair
from momo.significance import sharpe_significance_test


def find_correlated_pairs(prices: pd.DataFrame, min_correlation: float = 0.6) -> list:
    """Vectorized candidate search over the full return-correlation matrix.
    A per-pair Python loop (the original implementation) is fine for a few
    hundred tickers but becomes minutes-to-hours of work at market-wide
    scale (~2,400 tickers is ~2.9M possible pairs) -- pandas'/numpy's
    BLAS-backed .corr() computes the whole matrix in a few seconds instead.
    Uses full-sample correlation for candidate selection (a stronger read
    on "is this pair genuinely related" than one short trailing window);
    the actual trading signal inside backtest_pair still uses a proper
    rolling window, so this doesn't introduce look-ahead into the
    backtested returns themselves -- it only decides what to consider."""
    returns = prices.pct_change(fill_method=None)
    corr_matrix = returns.corr()

    tickers = corr_matrix.columns.to_numpy()
    values = corr_matrix.to_numpy()
    n = len(tickers)
    iu, ju = np.triu_indices(n, k=1)
    pair_corrs = values[iu, ju]

    mask = pair_corrs >= min_correlation
    return [
        (str(tickers[i]), str(tickers[j]), float(c))
        for i, j, c in zip(iu[mask], ju[mask], pair_corrs[mask])
        if not np.isnan(c)
    ]


def screen_pairs(tickers: list, prices: pd.DataFrame, min_correlation: float = 0.6,
                  lookback: int = 60, entry_z: float = 2.0, exit_z: float = 0.5,
                  stop_z: float = 3.5, costs_bps: float = 5.0, max_pairs_to_backtest: int = 200) -> dict:
    """Backtests correlation-qualifying pairs in `tickers`, applies a
    Bonferroni correction across the number of pairs actually tested (not
    just the number that happened to look good), and reports both the
    individual results and the pooled average -- the pooled number is the
    more honest answer to "does pairs trading have an edge here", since
    any single pair's result is a small sample on its own. If more than
    max_pairs_to_backtest qualify, keeps the highest-correlation ones --
    chosen before any backtest result is seen, not cherry-picked after."""
    candidates = find_correlated_pairs(prices[tickers], min_correlation=min_correlation)
    candidates.sort(key=lambda c: c[2], reverse=True)
    truncated = len(candidates) > max_pairs_to_backtest
    candidates = candidates[:max_pairs_to_backtest]

    if not candidates:
        return {"pairs_tested": 0, "results": [], "pooled": None,
                "reason": f"no pair met the min_correlation={min_correlation} threshold"}

    results = []
    for ticker_a, ticker_b, corr in candidates:
        bt = backtest_pair(ticker_a, prices[ticker_a], ticker_b, prices[ticker_b],
                            lookback=lookback, entry_z=entry_z, exit_z=exit_z,
                            stop_z=stop_z, costs_bps=costs_bps)
        sig = sharpe_significance_test(bt["returns"])
        results.append({
            "ticker_a": ticker_a, "ticker_b": ticker_b, "correlation": round(corr, 3),
            "stats": bt["stats"], "p_value": sig.get("p_value"),
            "returns": bt["returns"],
        })

    n_tested = len(results)
    corrected_alpha = 0.05 / n_tested
    survivors = [r for r in results if r["p_value"] is not None and r["p_value"] < corrected_alpha]

    all_returns = pd.concat([r["returns"] for r in results], axis=1).mean(axis=1)
    pooled_sig = sharpe_significance_test(all_returns)

    for r in results:
        r.pop("returns")

    return {
        "pairs_tested": n_tested,
        "truncated_to_max_pairs": truncated,
        "corrected_alpha": round(corrected_alpha, 5),
        "results": sorted(results, key=lambda r: (r["p_value"] is None, r["p_value"])),
        "survivors_after_correction": survivors,
        "pooled": {
            "n_pairs_pooled": n_tested,
            "pooled_sharpe_annualized": pooled_sig.get("sharpe_annualized"),
            "pooled_p_value": pooled_sig.get("p_value"),
            "pooled_significant_at_5pct": pooled_sig.get("significant_at_5pct"),
        },
    }
