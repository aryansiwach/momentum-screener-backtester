"""Runs the morning momentum report and prints the top picks. Meant to be
run by a scheduled Claude Code task, which reports the output back to the
user directly -- this script itself just fetches and prints, it never
places orders."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from momo.reporting import morning_momentum_report
from momo.data_layer import AlpacaDataLoader
from momo.execution import AlpacaBroker


def main():
    try:
        equity = AlpacaBroker(paper=True).get_equity()
        report = morning_momentum_report(
            ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "V",
             "UNH", "XOM", "MA", "HD", "COST", "PG", "NFLX", "AMD", "CRM", "ADBE",
             "BAC", "KO", "PEP", "TMO", "LIN", "WMT", "MCD", "ABT", "CSCO", "ORCL"],
            top_n=3, equity=equity, data_loader_cls=AlpacaDataLoader,
        )
        if report.empty:
            title = "Morning Momentum"
            body = "No positive-momentum picks today."
        else:
            title = "Morning Momentum — Top Picks"
            lines = [
                f"{row.ticker}: score {row.momentum_score:.2f}, ${row.suggested_dollars:.0f}"
                for row in report.itertuples()
            ]
            body = "\n".join(lines)
    except Exception as exc:
        title = "Morning Momentum — Error"
        body = f"Report failed: {exc}"

    print(title)
    print(body)


if __name__ == "__main__":
    main()
