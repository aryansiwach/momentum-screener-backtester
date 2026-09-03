"""Run the momentum screener over a watchlist and print today's picks with
suggested dollar allocation. Data source and equity figure are swappable --
defaults to free Yahoo Finance data and no dollar sizing (pass --equity to
size against a real account balance, or --alpaca to use live Alpaca data
and your real account equity)."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momo.reporting import morning_momentum_report, full_market_scan


def main():
    parser = argparse.ArgumentParser(description="Daily momentum screener report")
    parser.add_argument("tickers", nargs="*", help="Watchlist, e.g. AAPL MSFT NVDA TSLA")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--equity", type=float, default=None,
                         help="Account equity to size positions against")
    parser.add_argument("--alpaca", action="store_true",
                         help="Use Alpaca data + live account equity instead of Yahoo Finance")
    parser.add_argument("--full-market", action="store_true",
                         help="Scan the whole tradable US equity universe instead of a fixed watchlist "
                              "(ignores positional tickers; pulls the universe from Alpaca)")
    parser.add_argument("--min-price", type=float, default=5.0,
                         help="Full-market scan only: drop tickers trading below this price")
    parser.add_argument("--news", action="store_true",
                         help="Attach news sentiment to the shortlisted picks (requires Alpaca News access)")
    args = parser.parse_args()

    if not args.full_market and not args.tickers:
        parser.error("provide a ticker watchlist, or pass --full-market to scan the whole universe")

    equity = args.equity
    kwargs = {}
    if args.alpaca or args.full_market:
        from momo.data_layer import AlpacaDataLoader
        from momo.execution import AlpacaBroker
        kwargs["inner_loader_cls" if args.full_market else "data_loader_cls"] = AlpacaDataLoader
        if equity is None:
            equity = AlpacaBroker(paper=True).get_equity()

    if args.full_market:
        report = full_market_scan(
            lookback_days=args.lookback_days, top_n=args.top_n, min_price=args.min_price,
            equity=equity, with_news=args.news, **kwargs,
        )
    else:
        report = morning_momentum_report(
            args.tickers, lookback_days=args.lookback_days, top_n=args.top_n,
            equity=equity, **kwargs,
        )
    if report.empty:
        print("No positive-momentum picks today.")
        return
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
