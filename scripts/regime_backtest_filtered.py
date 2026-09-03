"""Same regime backtest, but with the SPY-trend regime filter applied --
answers honestly whether it actually helps, rather than assuming it does.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from momo.orchestrator import MomentumPipeline
from momo.data_layer import YFDataLoader
from momo.regime import compute_trend_state, apply_regime_scaling
from momo.backtest import Backtester
from scripts.regime_backtest import WATCHLIST, REGIMES


def run_regime_filtered(name, start, end, top_n=5, costs_bps=5, risk_off_scale=0.0):
    pipeline = MomentumPipeline(
        WATCHLIST, start, end, top_n=top_n, costs_bps=costs_bps,
        data_loader_cls=YFDataLoader,
    )
    try:
        results = pipeline.run()
    except Exception as exc:
        return {"regime": name, "error": str(exc)}

    # A 200-day SMA needs ~200 trading days of real history *before* the
    # regime window starts, or the first ~200 days read as risk-off simply
    # because the SMA hasn't warmed up yet (NaN comparisons are False) --
    # not because the market was actually trending down. YFDataLoader pads
    # its own fetch internally but trims the returned frame back to
    # [start, end], so that padding isn't visible here; fetch further back
    # explicitly instead.
    padded_start = pd.Timestamp(start) - pd.Timedelta(days=400)
    spy = YFDataLoader(["SPY"], padded_start, end).load_adj_close()["SPY"]
    trend_state = compute_trend_state(spy, sma_window=200)

    filtered_weights = apply_regime_scaling(results["weights"], trend_state, risk_off_scale=risk_off_scale)
    port_returns, equity, stats = Backtester(costs_bps=costs_bps).run(results["prices"], filtered_weights)
    win_rate = float((port_returns > 0).mean()) if len(port_returns) else float("nan")

    return {
        "regime": name,
        "CAGR": round(float(stats["CAGR"]) * 100, 2),
        "Sharpe": round(float(stats["Sharpe"]), 2),
        "Max Drawdown": round(float(stats["Max Drawdown"]) * 100, 2),
        "Win Rate": round(win_rate * 100, 1),
        "Days Risk-Off": int((~trend_state.reindex(results["prices"].index).ffill().fillna(True)).sum()),
    }


def main():
    print("Regime-filtered backtest (SPY 200-day trend filter, risk-off = fully flat)...\n")
    rows = [run_regime_filtered(name, start, end) for name, start, end in REGIMES]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
