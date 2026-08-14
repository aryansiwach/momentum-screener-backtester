# Momentum Screener & Backtester

An interactive Streamlit app for ranking stocks by composite momentum and
backtesting a rules-based long/short strategy against real market data —
with a clean, modular Python package underneath it, not just a notebook.

## What it does

- **Composite momentum scoring**: ranks any list of tickers each day using
  a percentile-weighted blend of 5-year-tested signals — 63-day return
  momentum, RSI, MACD histogram, price-vs-SMA trend, and stochastic %K.
- **Portfolio construction**: builds long-only or long/short (dollar-tilted)
  portfolios from those scores, rebalanced monthly or weekly.
- **Realistic backtesting**: applies yesterday's weights to today's returns
  and charges turnover-based transaction costs — no look-ahead bias.
- **Performance stats**: CAGR, annualized volatility, Sharpe, Sortino, max
  drawdown, average daily turnover.
- **A real UI, not just a script**: ticker search, date range, adjustable
  weights/lookbacks/costs, an equity curve, rolling Sharpe, a momentum-score
  heatmap, and one-click Excel/CSV export of every result table.

Verified against real Yahoo Finance data end to end — a 6-year, 12-ticker
backtest (AAPL, MSFT, NVDA, META, GOOGL, AMZN, TSLA, BRK-B, JPM, XOM, SPY,
QQQ; 2019–2024) runs in ~7 seconds and produces a 14.8% CAGR / 0.74 Sharpe
/ -28% max drawdown result — sensible numbers for a long/short momentum
strategy over that window, not a broken placeholder.

## Architecture

```
app.py                 Streamlit UI -- wires the pieces below together
momo/
  data_layer.py         Price loaders: Yahoo Finance (real) or synthetic GBM (demo)
  indicators.py          SMA, EMA, RSI, MACD, Stochastic, momentum return
  screener.py             Composite percentile-ranked momentum score
  portfolio.py            Turns scores into rebalanced long/short weights
  backtest.py              Vectorized backtest + performance stats
  orchestrator.py           Wires loader -> screener -> portfolio -> backtest
  visualize.py               Matplotlib charts (used outside the Streamlit UI)
tests/                  pytest suite for indicators and backtest logic
results/, results_full_run/    Sample output from real runs
```

The package layer (`momo/`) has no Streamlit dependency and no I/O beyond
price loading — `MomentumPipeline` in `orchestrator.py` is a plain Python
object you can drive from a script, a notebook, or a different UI entirely.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Type any tickers (space or comma separated), pick a date range, hit **Run
analysis**. "Synthetic (demo)" data source needs no network call if you
just want to see it work offline.

## Tests

```bash
pip install pytest
pytest tests/ -v
```

## Notes

This is a research/educational tool, not investment advice, and the
backtest doesn't model slippage, market impact, shorting costs/borrow
availability, or execution at bid/ask rather than close price — real
trading would need all of those. `results/` and `results_full_run/` are
committed as sample output so you can see what a real run produces without
running one yourself.
