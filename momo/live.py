import pandas as pd
from momo.orchestrator import MomentumPipeline
from momo.data_layer import AlpacaDataLoader
from momo.execution import AlpacaBroker
from momo.reporting import log_progress_snapshot


class LiveMomentumTrader:
    def __init__(self, tickers, lookback_days=400, top_n=3, short_n=0,
                 screener_kwargs=None, paper=True):
        self.tickers = list(tickers)
        self.lookback_days = lookback_days
        self.top_n = top_n
        self.short_n = short_n
        self.screener_kwargs = screener_kwargs or {}
        self.broker = AlpacaBroker(paper=paper)

    def latest_target_weights(self):
        end = pd.Timestamp.today().normalize()
        start = end - pd.Timedelta(days=self.lookback_days)
        pipeline = MomentumPipeline(
            self.tickers, start, end,
            top_n=self.top_n, short_n=self.short_n,
            screener_kwargs=self.screener_kwargs,
            data_loader_cls=AlpacaDataLoader,
        )
        results = pipeline.run()
        weights = results["weights"].iloc[-1].to_dict()
        prices = results["prices"].iloc[-1].to_dict()
        return weights, prices

    def rebalance(self, dry_run=True):
        weights, prices = self.latest_target_weights()
        ticker_sectors = None
        if not dry_run:
            from momo.sectors import fetch_sectors
            ticker_sectors = fetch_sectors([t for t, w in weights.items() if w != 0])
        orders = self.broker.rebalance_to(weights, prices, dry_run=dry_run, ticker_sectors=ticker_sectors)
        if not dry_run:
            log_progress_snapshot(self.broker)
        return orders
