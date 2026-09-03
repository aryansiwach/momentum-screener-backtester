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
  data_layer.py         Price loaders: Yahoo Finance, Alpaca (live), or synthetic GBM (demo)
  indicators.py          SMA, EMA, RSI, MACD, Stochastic, momentum return
  screener.py             Composite percentile-ranked momentum score
  portfolio.py            Turns scores into rebalanced long/short weights
  backtest.py              Vectorized backtest + performance stats
  orchestrator.py           Wires loader -> screener -> portfolio -> backtest
  execution.py               Target weights -> broker orders (Alpaca), pure sizing logic
  live.py                     Wires the pipeline + Alpaca data + execution together
  reporting.py                 Morning momentum report + equity/position progress log
  visualize.py                  Matplotlib charts (used outside the Streamlit UI)
scripts/
  morning_report.py       CLI: today's momentum picks + suggested $ sizing
tests/                  pytest suite for indicators, backtest, and order-sizing logic
results/, results_full_run/    Sample output from real runs
```

The package layer (`momo/`) has no Streamlit dependency and no I/O beyond
price loading — `MomentumPipeline` in `orchestrator.py` is a plain Python
object you can drive from a script, a notebook, or a different UI entirely.

## Live / paper trading (Alpaca)

1. Create a free Alpaca account and generate **paper trading** API keys at
   alpaca.markets.
2. Copy `.env.example` to `.env` and fill in `ALPACA_API_KEY` /
   `ALPACA_SECRET_KEY`. `.env` is gitignored — never commit real keys.
3. Install the extra dependencies: `pip install -r requirements.txt`.
4. Dry-run a rebalance (computes orders, submits nothing):

   ```python
   from dotenv import load_dotenv; load_dotenv()
   from momo.live import LiveMomentumTrader

   trader = LiveMomentumTrader(["AAPL","MSFT","NVDA","META","GOOGL"], top_n=3, paper=True)
   print(trader.rebalance(dry_run=True))
   ```

5. Once the dry-run output looks right, call `trader.rebalance(dry_run=False)`
   to actually submit paper orders. `paper=True` is the default everywhere —
   switching to a live (real-money) account requires deliberately passing
   `paper=False`, on purpose, so it can't happen by accident.
6. Every non-dry-run rebalance appends an equity/position snapshot to
   `progress/equity_log.csv` (gitignored) via `momo.reporting.log_progress_snapshot`
   — read it back with `momo.reporting.load_progress_log()`.

For a daily momentum readout without trading:

```bash
python scripts/morning_report.py AAPL MSFT NVDA META GOOGL TSLA --top-n 5 --equity 10000
```

Add `--alpaca` to pull live Alpaca data and size against your real paper/live
account equity instead of a manually-supplied number.

**Start on paper, not live money.** Nothing here models slippage, partial
fills, or PDT-rule/margin constraints the way a real live account would
enforce them — validate the whole loop on paper first.

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
