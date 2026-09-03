# Team structure

Each "team" below is one deterministic module with one job — not an
autonomous agent, on purpose (see the reasoning in the project history:
a system trading real money should be auditable, not self-modifying).
Running "multiple teams in parallel" means running these modules with
different configs side by side, which the architecture already supports.

| Team | Module | Job |
|---|---|---|
| Scout | `momo/universe.py` | Lists every tradable US equity (Alpaca) |
| Data | `momo/data_layer.py` | Pulls prices — Yahoo (free), Alpaca (live), WRDS (deep backtest) |
| Analyst | `momo/indicators.py`, `momo/screener.py` | Scores momentum per ticker |
| Treasury | `momo/portfolio.py`, `momo/execution.py` | Turns scores into sized orders |
| Sentinel | `momo/risk.py` | Stop-loss / take-profit / trailing-stop, position preview |
| Newsroom | `momo/news.py` | Sentiment on the day's shortlist |
| Herald | `momo/reporting.py`, `api.py` | Morning report, dashboard API, P&L tracking |

To run a "second team" on a different strategy (e.g. a slower swing
config alongside the day-trade config), instantiate the same modules with
different parameters — no new infrastructure needed.
