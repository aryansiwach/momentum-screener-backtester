
import pandas as pd
import numpy as np
from momo.indicators import rsi, macd, sma, stochastic, momentum_return

class Screener:
    """
    Compute composite momentum scores and ranks.
    You can tune the weights and lookbacks as you experiment.
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
