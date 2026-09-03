"""Answers the exact critique from the comment thread that prompted this
script: "most of these strategies will fall apart in live trading once
real-world spreads and slippage are taken into account." Re-runs the same
5-regime backtest (scripts/regime_backtest.py) at three cost assumptions
instead of one -- 5bps (the original optimistic default), 15bps (a more
realistic retail assumption), and 30bps (conservative, closer to what a
thinner name might actually cost) -- to show how much the results actually
degrade, honestly, rather than asserting an answer either way.

Uses free Yahoo Finance data -- no keys needed. Run it yourself:
    venv/Scripts/python.exe scripts/regime_cost_sensitivity.py
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

COST_SCENARIOS = [
    ("5bps (optimistic)", 5),
    ("15bps (realistic retail)", 15),
    ("30bps (conservative, thin name)", 30),
]


def run_regime(name, start, end, costs_bps, top_n=5):
    pipeline = MomentumPipeline(
        WATCHLIST, start, end, top_n=top_n, costs_bps=costs_bps,
        data_loader_cls=YFDataLoader,
    )
    try:
        results = pipeline.run()
    except Exception as exc:
        return {"regime": name, "error": str(exc)}

    stats = results["stats"]
    return {
        "regime": name,
        "CAGR": round(float(stats["CAGR"]) * 100, 2),
        "Sharpe": round(float(stats["Sharpe"]), 2),
        "Max Drawdown": round(float(stats["Max Drawdown"]) * 100, 2),
    }


def main():
    print(f"Cost sensitivity: same {len(WATCHLIST)}-ticker watchlist, same 5 regimes, "
          f"varying only the transaction cost assumption.\n")

    all_rows = []
    for scenario_name, bps in COST_SCENARIOS:
        print(f"=== {scenario_name} ===")
        for name, start, end in REGIMES:
            row = run_regime(name, start, end, bps)
            row["scenario"] = scenario_name
            row["bps"] = bps
            all_rows.append(row)
        df = pd.DataFrame([r for r in all_rows if r["scenario"] == scenario_name])
        print(df[["regime", "CAGR", "Sharpe", "Max Drawdown"]].to_string(index=False))
        print()

    full = pd.DataFrame(all_rows)
    print("=== Sharpe by regime across cost scenarios ===")
    pivot = full.pivot(index="regime", columns="scenario", values="Sharpe")
    pivot = pivot[[s for s, _ in COST_SCENARIOS]]
    print(pivot.to_string())

    print("\n=== Average Sharpe across all 5 regimes, per cost scenario ===")
    for scenario_name, _ in COST_SCENARIOS:
        avg = full[full["scenario"] == scenario_name]["Sharpe"].mean()
        print(f"  {scenario_name}: {avg:.2f}")


if __name__ == "__main__":
    main()
