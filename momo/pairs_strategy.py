"""A tradeable mean-reversion strategy over a pair, built on the z-score
scanner in momo/pairs.py: it enters when the spread diverges, exits on
reversion or a further-divergence stop, and can be backtested with the
same honesty standard as the momentum strategy in momo/backtest.py.
This is a genuinely different signal family from momentum -- momentum
bets a trend continues, this bets a historical relationship reasserts
itself -- which is the actual point of adding it: the two tend to fail
in different regimes, not the same one.

STRUCTURAL HYPOTHESIS: two tickers correlated because they share a common
economic driver (same sector, same input costs, same customer base) trade
at a stable price ratio absent new information specific to just one of
them. A temporary dislocation in that ratio -- caused by short-term
liquidity/flow pressure on one leg, not a change in the underlying
relationship -- gets arbitraged back toward the historical ratio, which
is what the z-score entry/exit is actually betting on.

What would falsify this: a pair whose spread never reverts within the
holding-period assumptions (the relationship broke, not just dislocated --
e.g. one company's fundamentals genuinely diverged from the other's,
which no z-score threshold can distinguish from a temporary dislocation
ahead of time). This project's own full-market pairs screen already found
the honest result: pooled Sharpe looked significant (p=0.045) but zero
individual pairs survived Bonferroni correction across 200 tested pairs --
meaning this hypothesis has not yet cleared its own bar, and that result
should stand until a genuinely new, pre-registered test says otherwise --
not be re-tested with new parameters until one clears it."""

import numpy as np
import pandas as pd

from momo.pairs import compute_spread_zscore


def generate_pair_signals(price_a: pd.Series, price_b: pd.Series, lookback: int = 60,
                           entry_z: float = 2.0, exit_z: float = 0.5, stop_z: float = 3.5) -> pd.Series:
    """Position state per day: +1 = long A/short B, -1 = short A/long B,
    0 = flat. Enters on divergence past entry_z, exits on reversion past
    exit_z, or cuts the position if the spread keeps widening past stop_z
    instead of reverting -- a pairs trade's version of a stop-loss."""
    z = compute_spread_zscore(price_a, price_b, lookback)
    position = pd.Series(0, index=z.index, dtype=int)
    current = 0
    for i in range(len(z)):
        zi = z.iloc[i]
        if pd.isna(zi):
            position.iloc[i] = current
            continue
        if current == 0:
            if zi >= entry_z:
                current = -1  # A rich vs B -> short A, long B
            elif zi <= -entry_z:
                current = 1   # A cheap vs B -> long A, short B
        else:
            if abs(zi) <= exit_z or abs(zi) >= stop_z:
                current = 0
        position.iloc[i] = current
    return position


def backtest_pair(ticker_a: str, price_a: pd.Series, ticker_b: str, price_b: pd.Series,
                   lookback: int = 60, entry_z: float = 2.0, exit_z: float = 0.5,
                   stop_z: float = 3.5, costs_bps: float = 5.0, periods_per_year: int = 252) -> dict:
    """Applies yesterday's position to today's spread return (no look-ahead,
    same discipline as the single-asset Backtester), nets out turnover-based
    transaction costs on both legs, and reports the same stats vocabulary
    as the momentum backtest so the two are directly comparable."""
    position = generate_pair_signals(price_a, price_b, lookback, entry_z, exit_z, stop_z)

    ret_a = price_a.pct_change(fill_method=None).fillna(0)
    ret_b = price_b.pct_change(fill_method=None).fillna(0)
    pos_shifted = position.shift().fillna(0)

    spread_return = pos_shifted * (ret_a - ret_b)
    turnover = pos_shifted.diff().abs().fillna(0)  # two legs move together, so this is per-unit-of-pair-notional
    costs = turnover * (costs_bps / 10000.0) * 2  # both legs incur costs
    net_return = spread_return - costs
    equity = (1 + net_return).cumprod()

    num_trades = int((position.diff().fillna(0) != 0).sum())
    win_rate = float((net_return[position.shift().fillna(0) != 0] > 0).mean()) if num_trades else float("nan")

    years = len(net_return) / periods_per_year
    cagr = float((1 + net_return).prod() ** (1 / years) - 1) if years > 0 else float("nan")
    ann_vol = float(net_return.std() * np.sqrt(periods_per_year))
    sharpe = float(net_return.mean() * periods_per_year / ann_vol) if ann_vol else float("nan")
    peak = equity.cummax()
    max_dd = float((equity / peak - 1).min())

    return {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "returns": net_return,
        "equity": equity,
        "stats": {
            "CAGR": round(cagr * 100, 2) if not np.isnan(cagr) else None,
            "Sharpe": round(sharpe, 2) if not np.isnan(sharpe) else None,
            "Max Drawdown": round(max_dd * 100, 2),
            "Win Rate": round(win_rate * 100, 1) if not np.isnan(win_rate) else None,
            "Num Trades": num_trades,
        },
    }
