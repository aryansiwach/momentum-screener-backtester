"""Backtests the actual composite momentum strategy (not just its
components) across distinct historical market regimes: a steady bull, the
COVID crash+recovery, a rate-hike bear market, a genuinely choppy/sideways
stretch, and the most recent two years. This is the test that was missing:
every indicator is unit-tested for correctness, but nobody had checked
whether the strategy, as a whole, would have made money.

Uses free Yahoo Finance data -- no keys needed. Run it yourself:
    venv/Scripts/python.exe scripts/regime_backtest.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from momo.orchestrator import MomentumPipeline
from momo.data_layer import YFDataLoader

WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "V",
    "UNH", "XOM", "MA", "HD", "COST", "PG", "NFLX", "AMD", "CRM", "ADBE",
    "BAC", "KO", "PEP", "TMO", "LIN", "WMT", "MCD", "ABT", "CSCO", "ORCL",
]

REGIMES = [
    ("2019 steady bull",        "2019-01-01", "2019-12-31"),
    ("2020 COVID crash+recovery", "2020-01-01", "2020-12-31"),
    ("2022 rate-hike bear",     "2022-01-01", "2022-12-31"),
    ("2015-16 choppy/sideways", "2015-01-01", "2016-06-30"),
    ("2023-24 recent",          "2023-01-01", "2024-12-31"),
]


def run_regime(name, start, end, top_n=5, costs_bps=5):
    pipeline = MomentumPipeline(
        WATCHLIST, start, end, top_n=top_n, costs_bps=costs_bps,
        data_loader_cls=YFDataLoader,
    )
    try:
        results = pipeline.run()
    except Exception as exc:
        return {"regime": name, "error": str(exc)}

    port_returns = results["returns"]
    stats = results["stats"]
    win_rate = float((port_returns > 0).mean()) if len(port_returns) else float("nan")

    return {
        "regime": name,
        "start": start,
        "end": end,
        "tickers_used": results["prices"].shape[1],
        "CAGR": round(float(stats["CAGR"]) * 100, 2),
        "Sharpe": round(float(stats["Sharpe"]), 2),
        "Max Drawdown": round(float(stats["Max Drawdown"]) * 100, 2),
        "Win Rate": round(win_rate * 100, 1),
    }


def main():
    print(f"Backtesting composite momentum (top-5, {len(WATCHLIST)}-ticker universe) "
          f"across {len(REGIMES)} regimes...\n")

    rows = []
    for name, start, end in REGIMES:
        print(f"  {name} ({start} to {end})...")
        rows.append(run_regime(name, start, end))

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))

    if "CAGR" in df.columns:
        valid = df.dropna(subset=["CAGR"])
        print(f"\nRegimes with a positive CAGR: {(valid['CAGR'] > 0).sum()} / {len(valid)}")
        print(f"Worst max drawdown across regimes: {valid['Max Drawdown'].min()}%")
        print(f"Average Sharpe across regimes: {valid['Sharpe'].mean():.2f}")


if __name__ == "__main__":
    main()
