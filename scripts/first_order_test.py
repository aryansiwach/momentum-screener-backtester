"""One real (paper) order, end to end: pulls a live price, sizes a small
position, and submits it via AlpacaBroker with dry_run=False -- the code
path that has never actually executed until now. Small on purpose: this is
a mechanism test, not a strategy call."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from momo.execution import AlpacaBroker

TICKER = "AAPL"
TARGET_WEIGHT = 0.05  # ~5% of equity -- small and bounded on purpose


def main():
    broker = AlpacaBroker(paper=True)

    equity = broker.get_equity()
    positions_before = broker.get_positions()
    print(f"Equity before: ${equity:,.2f}")
    print(f"Positions before: {positions_before}")

    # Alpaca's own latest trade price, not yfinance -- consistent with what
    # the broker will actually execute against.
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest

    client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    trade = client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=TICKER))[TICKER]
    price = float(trade.price)
    print(f"Latest {TICKER} price: ${price:.2f}")

    print(f"\nSubmitting real (paper) order: target weight {TARGET_WEIGHT:.0%} of equity in {TICKER}...")
    orders = broker.rebalance_to(
        target_weights={TICKER: TARGET_WEIGHT},
        prices={TICKER: price},
        dry_run=False,
    )
    print(f"Orders submitted: {orders}")

    positions_after = broker.get_positions()
    equity_after = broker.get_equity()
    print(f"\nPositions after: {positions_after}")
    print(f"Equity after: ${equity_after:,.2f}")


if __name__ == "__main__":
    main()
