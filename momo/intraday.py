"""Intraday day-trading loop: minute-bar momentum entries, the 3%/1.5%
day-trade exit rules, a PDT guard, and a hard flatten-by-close. Separate
from momo/live.py, which is the daily/swing rebalancing path -- the two
run on different data cadences and shouldn't share a rebalance schedule."""

import pandas as pd

from momo.orchestrator import MomentumPipeline
from momo.data_layer import AlpacaDataLoader
from momo.execution import AlpacaBroker
from momo.risk import check_pdt_compliance
from momo.session import can_open_new_position, should_flatten
from momo.reporting import log_progress_snapshot

# Daily lookbacks (63/50/14/14) don't mean anything on minute bars -- these
# are the same indicators, scaled to an intraday session instead of a
# multi-month trend.
DEFAULT_INTRADAY_SCREENER_KWARGS = {
    "lookbacks": {"mom": 30, "sma": 20, "rsi": 14, "stoch": 14},
}


class IntradayTrader:
    def __init__(self, tickers, lookback_minutes=240, top_n=3,
                 screener_kwargs=None, paper=True, min_pdt_equity=25000.0):
        self.tickers = list(tickers)
        self.lookback_minutes = lookback_minutes
        self.top_n = top_n
        self.screener_kwargs = screener_kwargs or DEFAULT_INTRADAY_SCREENER_KWARGS
        self.broker = AlpacaBroker(paper=paper)
        self.min_pdt_equity = min_pdt_equity

    def latest_target_weights(self):
        end = pd.Timestamp.utcnow()
        start = end - pd.Timedelta(minutes=self.lookback_minutes)
        pipeline = MomentumPipeline(
            self.tickers, start, end,
            top_n=self.top_n, screener_kwargs=self.screener_kwargs,
            data_loader_cls=AlpacaDataLoader,
            data_loader_kwargs={"timeframe_amount": 1, "timeframe_unit": "Minute"},
        )
        results = pipeline.run()
        weights = results["weights"].iloc[-1].to_dict()
        prices = results["prices"].iloc[-1].to_dict()
        return weights, prices

    def run_once(self, day_trades_so_far=0, dry_run=True):
        """One decision cycle: flatten if near the close, else check PDT
        and session-timing guards, then rebalance toward the current
        intraday momentum picks. Call this on a short interval (e.g. every
        few minutes) during market hours -- it does not loop internally."""
        now = pd.Timestamp.utcnow()

        if should_flatten(now):
            positions = self.broker.get_positions()
            flatten_orders = {ticker: -qty for ticker, qty in positions.items() if qty != 0}
            if not dry_run:
                for ticker, qty in flatten_orders.items():
                    self.broker._submit(ticker, int(qty))
            return {"action": "flatten", "orders": flatten_orders}

        if not can_open_new_position(now):
            return {"action": "hold", "reason": "too close to the close to open new positions"}

        equity = self.broker.get_equity()
        pdt = check_pdt_compliance(equity, day_trades_so_far, low_equity_warning=self.min_pdt_equity)

        weights, prices = self.latest_target_weights()
        # Sector lookup is a real network call per ticker -- only worth
        # paying for when a real order is actually about to go out, since
        # that's the only time rebalance_to's risk gate consults it.
        ticker_sectors = None
        if not dry_run:
            from momo.sectors import fetch_sectors
            ticker_sectors = fetch_sectors([t for t, w in weights.items() if w != 0])
        orders = self.broker.rebalance_to(weights, prices, dry_run=dry_run, ticker_sectors=ticker_sectors)
        if not dry_run:
            log_progress_snapshot(self.broker)
        result = {"action": "rebalance", "orders": orders}
        if pdt["warning"]:
            result["warning"] = pdt["warning"]
        return result
