
import pandas as pd
import numpy as np

class PortfolioConstructor:
    """
    Builds portfolios given momentum scores.
    Supports:
      - long-only (top_n)
      - long-short (top_n longs, bottom_n shorts)
    Rebalances on given frequency (ME=month-end, W=weekly).
    """
    def __init__(self, top_n=3, short_n=0, rebalance='ME'):
        self.top_n = top_n
        self.short_n = short_n
        self.rebalance = rebalance

    def construct_weights(self, score: pd.DataFrame) -> pd.DataFrame:
        # rebalance dates (end of each period)
        rebal_dates = score.resample(self.rebalance).last().index
        weights = pd.DataFrame(0.0, index=score.index, columns=score.columns)

        for dt in rebal_dates:
            row = score.loc[:dt].iloc[-1]
            longs = row.nlargest(self.top_n).index
            shorts = row.nsmallest(self.short_n).index if self.short_n > 0 else []
            n_l, n_s = len(longs), len(shorts)

            w_long = 1.0 / (n_l + n_s) if n_s == 0 else 0.5 / n_l
            w_short = -0.5 / n_s if n_s > 0 else 0

            weights.loc[dt, longs] = w_long
            if n_s > 0:
                weights.loc[dt, shorts] = w_short

        # forward-fill until next rebalance
        weights = weights.replace(0, np.nan).ffill().fillna(0)
        return weights
