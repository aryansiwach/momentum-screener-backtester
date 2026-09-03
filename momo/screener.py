"""STRUCTURAL HYPOTHESIS (stated up front, not inferred after the fact --
see the "don't backtest for p-hacking, backtest to validate a hypothesis"
discipline this project holds itself to):

Claim: recent relative price strength (63-day return) and technical
confirmation (RSI/MACD/SMA/stochastic agreeing) predict near-term
continuation, not reversal.

Proposed economic mechanism -- underreaction, not overreaction: investors
and analysts revise views on new information gradually rather than
instantly (attention constraints, anchoring on prior estimates), and the
disposition effect causes early profit-taking that dampens full price
discovery on day one. Institutional rebalancing flows (index funds,
factor funds) also lag the information event itself. All three predict
the SAME thing: a real move keeps drifting for weeks, not that it snaps
back.

What would falsify this: if adding a mean-reversion filter (buying recent
losers) outperformed net of costs, or if the strategy's own live-vs-paper
track record shows no persistence, the underreaction story is wrong for
this universe/period -- that's a reason to stop trusting the score, not a
cue to re-tune the weights until a backtest looks better again.

Known counter-evidence already found this session, reported honestly:
Daniel & Moskowitz (2016) show momentum crashes hardest exactly during
bear-market rebounds (up-market beta roughly double down-market beta),
matching this project's own unresolved 2022 regime weakness -- the
underreaction story doesn't explain why momentum reverses violently in
exactly that state, and two attempted fixes (a regime filter, GARCH
vol-targeting) both failed to address it."""

import pandas as pd
import numpy as np
from momo.indicators import rsi, macd, sma, stochastic, momentum_return

class Screener:
    """
    Compute composite momentum scores and ranks.
    You can tune the weights and lookbacks as you experiment -- but see the
    module-level STRUCTURAL HYPOTHESIS docstring above first: a change here
    should trace back to a specific claim about *why* the new weighting
    reflects the underreaction mechanism better, not just to a backtest
    number moving in the right direction.
    """
    def __init__(self, weights=None, lookbacks=None):
        self.weights = weights or {
            "mom": 0.35,
            "rsi": 0.15,
            "macd": 0.25,
            "sma": 0.15,
            "stoch": 0.10
        }
        self.lookbacks = lookbacks or {
            "mom": 63,
            "sma": 50,
            "rsi": 14,
            "stoch": 14
        }

    def composite_scores(self, prices: pd.DataFrame) -> pd.DataFrame:
        # individual indicator scores scaled 0–1 (percentile rank each day)
        mom = momentum_return(prices, self.lookbacks["mom"]).rank(axis=1, pct=True)
        rsi_df = rsi(prices, self.lookbacks["rsi"]).rank(axis=1, pct=True)
        macd_line, signal, hist = macd(prices)
        macd_df = hist.rank(axis=1, pct=True)
        sma_df = (prices / sma(prices, self.lookbacks["sma"])).rank(axis=1, pct=True)
        stoch_k, stoch_d = stochastic(prices, self.lookbacks["stoch"])
        stoch_df = stoch_k.rank(axis=1, pct=True)

        score = (
            self.weights["mom"]   * mom +
            self.weights["rsi"]   * rsi_df +
            self.weights["macd"]  * macd_df +
            self.weights["sma"]   * sma_df +
            self.weights["stoch"] * stoch_df
        )
        return score

    def top_n(self, prices: pd.DataFrame, n=3):
        score = self.composite_scores(prices)
        picks = score.apply(lambda row: row.nlargest(n).index.tolist(), axis=1)
        return score, picks
