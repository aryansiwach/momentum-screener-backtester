
import pandas as pd
from momo.data_layer import DataLoader
from momo.screener import Screener
from momo.portfolio import PortfolioConstructor
from momo.backtest import Backtester
from momo.visualize import Visualizer

class MomentumPipeline:
    def __init__(
        self,
        tickers,
        start,
        end,
        top_n=3,
        short_n=0,
        rebalance='ME',
        costs_bps=5,
        screener_kwargs=None,
        data_loader_cls=DataLoader,     # <-- NEW
        data_loader_kwargs=None         # <-- NEW
    ):
        self.tickers = list(tickers)
        self.start = start
        self.end = end
        self.scr = Screener(**(screener_kwargs or {}))
        self.pf  = PortfolioConstructor(top_n=top_n, short_n=short_n, rebalance=rebalance)
        self.bt  = Backtester(costs_bps=costs_bps)
        self.data_loader_cls = data_loader_cls
        self.data_loader_kwargs = data_loader_kwargs or {}

    def run(self):
        # 1) prices
        loader = self.data_loader_cls(self.tickers, self.start, self.end, **self.data_loader_kwargs)
        prices = loader.load_adj_close()

        # 2) scores
        scores, _ = self.scr.top_n(prices, n=max(1, self.pf.top_n))

        # 3) weights
        weights = self.pf.construct_weights(scores)

        # 4) backtest
        returns, equity, stats = self.bt.run(prices, weights)

        return {"prices": prices, "scores": scores, "weights": weights,
                "returns": returns, "equity": equity, "stats": stats}

    def report(self, results, benchmark_ticker=None):
        prices = results["prices"]
        equity = results["equity"]
        returns = results["returns"]
        scores  = results["scores"]

        bench = None
        if benchmark_ticker and benchmark_ticker in prices.columns:
            bench = (1 + prices[benchmark_ticker].pct_change().fillna(0)).cumprod()

        Visualizer.equity_curve(equity, bench)
        Visualizer.rolling_sharpe(returns)
        Visualizer.score_heatmap(scores, n_last=150)
