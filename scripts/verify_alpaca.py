"""Verifies the Alpaca paper-trading connection end to end: account access,
a daily price pull, a minute-bar pull, and a dry-run intraday scan. Reads
keys from .env -- never prints them."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from momo.execution import AlpacaBroker
from momo.data_layer import AlpacaDataLoader
import pandas as pd


def main():
    if not os.environ.get("ALPACA_API_KEY") or not os.environ.get("ALPACA_SECRET_KEY"):
        print("ALPACA_API_KEY / ALPACA_SECRET_KEY not found in the environment -- "
              "check that .env is saved and has real values, not the placeholders.")
        sys.exit(1)

    print("1) Account connection...")
    broker = AlpacaBroker(paper=True)
    equity = broker.get_equity()
    positions = broker.get_positions()
    print(f"   OK -- paper account equity: ${equity:,.2f}, open positions: {len(positions)}")

    print("2) Daily bar pull (AAPL, last 10 days)...")
    end = pd.Timestamp.today().normalize()
    daily = AlpacaDataLoader(["AAPL"], end - pd.Timedelta(days=15), end).load_adj_close()
    print(f"   OK -- {len(daily)} daily bars, latest close ${daily['AAPL'].iloc[-1]:.2f}")

    print("3) Minute bar pull (AAPL, most recent trading session)...")
    # widen to 5 calendar days so this works even run on a weekend/holiday,
    # when "the last few hours" has no session in it at all
    minute = AlpacaDataLoader(
        ["AAPL"], pd.Timestamp.utcnow() - pd.Timedelta(days=5), pd.Timestamp.utcnow(),
        timeframe_amount=1, timeframe_unit="Minute",
    ).load_adj_close()
    print(f"   OK -- {len(minute)} minute bars pulled, most recent: {minute.index[-1]}")

    print("4) Dry-run intraday scan...")
    from momo.session import is_market_open
    from momo.intraday import IntradayTrader
    if not is_market_open(pd.Timestamp.utcnow()):
        print("   SKIPPED -- market is closed right now, so there's no live intraday "
              "window to scan. This step only produces real output during NYSE hours "
              "(9:30am-4pm ET, Mon-Fri). Everything else above already confirms Alpaca "
              "is correctly wired up.")
    else:
        trader = IntradayTrader(["AAPL", "MSFT", "NVDA", "TSLA", "META"], top_n=2, paper=True)
        weights, prices = trader.latest_target_weights()
        picks = {t: w for t, w in weights.items() if w > 0}
        print(f"   OK -- current intraday picks: {picks}")

    print("\nAll checks passed. Alpaca is fully wired up.")


if __name__ == "__main__":
    main()
