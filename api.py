"""HTTP bridge between the momo package and the web dashboard. Two tiers,
deliberately kept honest and separate:

- /momentum/watchlist works today with zero credentials (Yahoo Finance).
- /momentum/full-market and /account need real Alpaca keys (universe listing
  and account equity aren't available any other way) -- missing keys return
  a clear typed error instead of a stack trace.
"""

import os
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Must run before anything below reads os.environ -- without this the
# server process never sees ALPACA_API_KEY/ALPACA_SECRET_KEY from .env at
# all (they're only in the file, not the process environment), and every
# Alpaca-backed endpoint 503s even though the same code works fine in a
# script that calls load_dotenv() itself.
load_dotenv()

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware

from momo.reporting import morning_momentum_report, full_market_scan, performance_summary
from momo.data_layer import YFDataLoader
from momo.risk import position_preview, DAY_TRADE_STOP_LOSS_PCT, DAY_TRADE_TAKE_PROFIT_PCT
from momo.session import is_market_open, minutes_to_close, should_flatten, can_open_new_position
from momo.pairs import scan_pair, compute_hedge_position
from momo.pairs_strategy import backtest_pair
from momo.volatility import forecast_volatility
from momo.significance import sharpe_significance_test

app = FastAPI(title="Momentum Trading API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

DEFAULT_WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "V",
    "UNH", "XOM", "MA", "HD", "COST", "PG", "NFLX", "AMD", "CRM", "ADBE",
    "BAC", "KO", "PEP", "TMO", "LIN", "WMT", "MCD", "ABT", "CSCO", "ORCL",
]

# Background full-market scanner: /momentum/watchlist only ever ranks the 30
# names above, so a stronger mover anywhere else in the market (e.g. a name
# outside this list) never surfaces there no matter how it scores. This
# loop re-scans the *entire* tradable universe continuously, back-to-back
# with no artificial delay -- the real floor is how long the data pull
# takes (measured at ~65s for ~2,400 tickers earlier in this session), not
# a number we can just turn down. The dashboard polls the cheap in-memory
# cache below every 30s; the scan itself refreshes it as fast as it
# physically can.
_full_market_cache = {
    "picks": None,
    "universe_size": None,
    "updated_at": None,
    "scanning": False,
    "error": None,
    "sector_concentration": None,
}
_full_market_lock = threading.Lock()

# Persisted scan history: the in-memory cache above is lost on every server
# restart, leaving the dashboard empty for a full ~90s cycle and with no way
# to look back at what a past scan actually found. This is a plain
# append-only JSON-lines log, same spirit as reporting.py's equity_log.csv.
_SCAN_LOG_PATH = Path(__file__).resolve().parent / "progress" / "full_market_scan_log.jsonl"
_SCAN_LOG_MAX_ENTRIES = 500


def _append_scan_log(entry: dict):
    try:
        _SCAN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_SCAN_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        with open(_SCAN_LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Trim with headroom past the cap rather than every single write --
        # keeps this an occasional O(n) rewrite, not a per-scan one.
        if len(lines) > _SCAN_LOG_MAX_ENTRIES * 1.2:
            with open(_SCAN_LOG_PATH, "w", encoding="utf-8") as f:
                f.writelines(lines[-_SCAN_LOG_MAX_ENTRIES:])
    except Exception:
        pass  # persistence is a nice-to-have -- a write failure shouldn't take down the scan loop


def _read_scan_log(limit: int = 20) -> list:
    if not _SCAN_LOG_PATH.exists():
        return []
    try:
        with open(_SCAN_LOG_PATH, "r", encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
    except Exception:
        return []
    entries = []
    for line in lines[-limit:]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    entries.reverse()  # most recent first
    return entries


def _load_last_scan_into_cache():
    """Called at startup so a server restart doesn't show an empty
    dashboard for a full scan cycle -- preloads whatever the last completed
    scan before this process started actually found."""
    recent = _read_scan_log(limit=1)
    if not recent:
        return
    last = recent[0]
    with _full_market_lock:
        _full_market_cache["picks"] = last.get("picks")
        _full_market_cache["universe_size"] = last.get("universe_size")
        _full_market_cache["updated_at"] = last.get("updated_at")
        _full_market_cache["sector_concentration"] = last.get("sector_concentration")


def _full_market_scan_loop():
    from momo.universe import fetch_tradable_universe
    from momo.data_layer import AlpacaDataLoader
    from momo.risk import check_sector_concentration
    from momo.sectors import fetch_sectors as _fetch_sectors

    while True:
        if not _alpaca_keys_present():
            time.sleep(30)
            continue
        with _full_market_lock:
            _full_market_cache["scanning"] = True
        try:
            universe = fetch_tradable_universe()
            report = full_market_scan(
                top_n=10, min_price=5.0, universe=universe, inner_loader_cls=AlpacaDataLoader,
                exclude_gap_movers=True, exclude_news_risk=True, exclude_illiquid=True,
            )
            sector_concentration = None
            if not report.empty:
                tickers = report["ticker"].tolist()
                weights = dict(zip(report["ticker"], report["target_weight"]))
                sectors = _fetch_sectors(tickers)
                sector_concentration = check_sector_concentration(sectors, weights, max_sector_weight=0.4)
                sector_concentration["ticker_sectors"] = sectors
            picks = report.to_dict(orient="records")
            updated_at = pd.Timestamp.utcnow().isoformat()
            with _full_market_lock:
                _full_market_cache["picks"] = picks
                _full_market_cache["universe_size"] = len(universe)
                _full_market_cache["updated_at"] = updated_at
                _full_market_cache["error"] = None
                _full_market_cache["sector_concentration"] = sector_concentration
            _append_scan_log({
                "picks": picks,
                "universe_size": len(universe),
                "updated_at": updated_at,
                "sector_concentration": sector_concentration,
            })
        except Exception as exc:
            with _full_market_lock:
                _full_market_cache["error"] = str(exc)
        finally:
            with _full_market_lock:
                _full_market_cache["scanning"] = False


# Position exit monitor: momo/monitor.py and AlpacaBroker.check_exits /
# monitor_and_exit exist and are tested, but nothing was ever running them
# continuously -- a position's stop-loss/take-profit/trailing-stop was only
# ever a preview number in the dashboard, never actually watched. This
# loop checks open positions every 30s during market hours. It defaults to
# dry-run (reports what it WOULD sell, submits nothing) unless
# MOMO_LIVE_EXIT_MONITOR=true is explicitly set in the environment -- this
# is the function that can auto-liquidate a real position, so silence is
# the safe default, not the surprising one.
_LIVE_EXIT_MONITOR_ENABLED = os.environ.get("MOMO_LIVE_EXIT_MONITOR", "false").lower() == "true"
_exit_monitor_cache = {"verdicts": [], "updated_at": None, "live": _LIVE_EXIT_MONITOR_ENABLED, "error": None}
_exit_monitor_lock = threading.Lock()


def _exit_monitor_loop():
    from momo.execution import AlpacaBroker
    from momo.session import is_market_open

    while True:
        if not _alpaca_keys_present():
            time.sleep(30)
            continue
        if not is_market_open(pd.Timestamp.utcnow()):
            time.sleep(60)
            continue
        try:
            broker = AlpacaBroker(paper=True)
            verdicts = broker.monitor_and_exit(dry_run=not _LIVE_EXIT_MONITOR_ENABLED)
            with _exit_monitor_lock:
                _exit_monitor_cache["verdicts"] = verdicts
                _exit_monitor_cache["updated_at"] = pd.Timestamp.utcnow().isoformat()
                _exit_monitor_cache["error"] = None
        except Exception as exc:
            with _exit_monitor_lock:
                _exit_monitor_cache["error"] = str(exc)
        time.sleep(30)


@app.get("/positions/exits")
def positions_exits():
    """Read-only view of the exit monitor's latest pass -- what it found
    for each open position, and whether it's actually allowed to sell
    (`live`) or only reporting what it would do."""
    _require_alpaca_keys()
    with _exit_monitor_lock:
        return dict(_exit_monitor_cache)


# Pre-market / current-session movers: a genuinely different question from
# the 63-day daily momentum score -- "what's actually moving right now, and
# is there a real reason." Alpaca's free market-movers screener is
# unfiltered junk (warrants, sub-$1 instruments) by default; this loop
# filters it through the same clean-universe standard used everywhere else
# in this project, then attaches recent news so a mover comes with a "why",
# not just a number.
_premarket_cache = {"gainers": [], "losers": [], "intraday_leaders": [], "updated_at": None, "scanning": False, "error": None}
_premarket_lock = threading.Lock()


def _premarket_scan_loop():
    from momo.universe import fetch_tradable_universe
    from momo.premarket import fetch_market_movers, filter_clean_movers, fetch_intraday_trend, rank_intraday_leaders
    from momo.news import fetch_recent_news, detect_risk_flags, score_sentiment

    while True:
        if not _alpaca_keys_present():
            time.sleep(30)
            continue
        with _premarket_lock:
            _premarket_cache["scanning"] = True
        try:
            clean_universe = set(fetch_tradable_universe())
            movers = fetch_market_movers(top=50)
            gainers = filter_clean_movers(movers["gainers"], clean_universe, min_price=3.0)[:15]
            losers = filter_clean_movers(movers["losers"], clean_universe, min_price=3.0)[:15]

            candidates = [m["symbol"] for m in gainers + losers]
            # NewsRequest's limit is a total cap across every symbol in the
            # request combined, not per-ticker -- too low here would starve
            # most of the ~30 candidates of any headline at all.
            headlines = fetch_recent_news(candidates, lookback_days=2, limit=50) if candidates else {}
            flags_by_ticker = detect_risk_flags(headlines)
            sentiment_by_ticker = score_sentiment(headlines)

            def _fix_mojibake(text):
                # Some Benzinga-sourced headlines arrive with UTF-8 bytes
                # already mis-decoded as Latin-1 upstream (an apostrophe
                # comes through as "â€™") -- re-encoding as
                # Latin-1 and decoding as UTF-8 undoes that specific
                # mangling; anything not actually mangled just fails the
                # round-trip and is left alone.
                if not text:
                    return text
                try:
                    return text.encode("latin-1").decode("utf-8")
                except (UnicodeDecodeError, UnicodeEncodeError):
                    return text

            def _enrich(m):
                sym = m["symbol"]
                own_headlines = headlines.get(sym) or []
                return {
                    **m,
                    "headline": _fix_mojibake(own_headlines[0]) if own_headlines else None,
                    "news_flags": flags_by_ticker.get(sym, []),
                    "news_sentiment": sentiment_by_ticker.get(sym),
                }

            enriched_gainers = [_enrich(m) for m in gainers]
            enriched_losers = [_enrich(m) for m in losers]

            # Confirmation layer on top of the raw gainers: a mover being
            # up already is descriptive, not predictive -- this checks
            # each gainer's CURRENT minute-bar technicals (MACD, RSI) and
            # keeps only the ones that haven't shown exhaustion signals
            # yet, ranked by how strong that confirmation is. One extra
            # Alpaca call per gainer (already capped at 15), not per raw
            # mover -- losers are excluded since "leader" implies up.
            gainers_with_trend = [
                {**g, "intraday_trend": fetch_intraday_trend(g["symbol"])} for g in enriched_gainers
            ]
            intraday_leaders = rank_intraday_leaders(gainers_with_trend)[:10]

            with _premarket_lock:
                _premarket_cache["gainers"] = enriched_gainers
                _premarket_cache["losers"] = enriched_losers
                _premarket_cache["intraday_leaders"] = intraday_leaders
                _premarket_cache["updated_at"] = pd.Timestamp.utcnow().isoformat()
                _premarket_cache["error"] = None
        except Exception as exc:
            with _premarket_lock:
                _premarket_cache["error"] = str(exc)
        finally:
            with _premarket_lock:
                _premarket_cache["scanning"] = False
        time.sleep(60)


@app.get("/premarket/watch")
def premarket_watch():
    """Read-only view of the latest clean, news-attached movers pass --
    reflects whatever session is live when the background loop last ran:
    pre-market moves before the bell, regular-session moves during market
    hours. Not a prediction of what will keep moving."""
    _require_alpaca_keys()
    with _premarket_lock:
        return dict(_premarket_cache)


@app.on_event("startup")
def _start_background_workers():
    _load_last_scan_into_cache()
    if os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY"):
        threading.Thread(target=_full_market_scan_loop, daemon=True).start()
        threading.Thread(target=_exit_monitor_loop, daemon=True).start()
        threading.Thread(target=_premarket_scan_loop, daemon=True).start()


def _alpaca_keys_present() -> bool:
    return bool(os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY"))


def _require_alpaca_keys():
    if not _alpaca_keys_present():
        raise HTTPException(
            status_code=503,
            detail="Alpaca API keys not configured. Set ALPACA_API_KEY and "
                   "ALPACA_SECRET_KEY (see .env.example) to enable this endpoint.",
        )


@contextmanager
def _fetch_guard(message: str):
    """try/except -> HTTPException(502, "<message>: <exc>") as a context
    manager. This exact 4-line try/except block (an HTTPException from
    inside the block always passes through unwrapped; anything else
    becomes the standard 502) used to be copy-pasted at every external
    data-fetch call site in this file -- `with _fetch_guard("..."):`
    keeps the identical status code and message, written once."""
    try:
        yield
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"{message}: {exc}")


@app.get("/")
def health():
    return {"status": "ok", "service": "Momentum Trading API"}


@app.get("/quotes")
def quotes(tickers: str):
    """Near-real-time last-traded prices, decoupled from the daily momentum
    score -- this is the part of the dashboard that should visibly move
    between refreshes; the momentum score itself is a 63-day trend signal
    and correctly does not."""
    import yfinance as yf

    ticker_list = tickers.split(",")
    with _fetch_guard("Quote fetch failed"):
        yf_tickers = yf.Tickers(" ".join(ticker_list))

    out = {}
    for symbol, t in yf_tickers.tickers.items():
        try:
            out[symbol] = round(float(t.fast_info["lastPrice"]), 2)
        except Exception:
            out[symbol] = None
    return {"quotes": out}


@app.get("/momentum/watchlist")
def momentum_watchlist(
    tickers: Optional[str] = Query(None, description="Comma-separated tickers; defaults to a large-cap watchlist"),
    top_n: int = 10,
    lookback_days: int = 400,
    equity: Optional[float] = None,
):
    ticker_list = tickers.split(",") if tickers else DEFAULT_WATCHLIST
    with _fetch_guard("Data fetch failed"):
        report = morning_momentum_report(
            ticker_list, lookback_days=lookback_days, top_n=top_n,
            equity=equity, data_loader_cls=YFDataLoader,
        )
    return {"source": "yahoo_finance", "picks": report.to_dict(orient="records")}


@app.get("/momentum/score")
def momentum_score(ticker: str, lookback_days: int = 400):
    """Score a single arbitrary ticker (e.g. one typed into search that isn't
    in the default watchlist). The composite score is a cross-sectional
    percentile rank (see Screener.composite_scores), so it's only meaningful
    relative to a reference universe -- scoring the ticker completely alone
    would trivially rank it 1.0. Score it against the same DEFAULT_WATCHLIST
    used everywhere else on the dashboard so the number is comparable to
    what's already shown in the table."""
    from momo.screener import Screener

    ticker = ticker.upper()
    universe = DEFAULT_WATCHLIST if ticker in DEFAULT_WATCHLIST else DEFAULT_WATCHLIST + [ticker]
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=lookback_days)
    with _fetch_guard("Score computation failed"):
        loader = YFDataLoader(universe, start, end)
        prices = loader.load_adj_close()
        scores = Screener().composite_scores(prices)
        latest = scores.iloc[-1]
    if ticker not in latest.index or pd.isna(latest[ticker]):
        raise HTTPException(status_code=404, detail=f"No momentum score available for {ticker}")
    return {"ticker": ticker, "momentum_score": round(float(latest[ticker]), 4), "universe_size": len(universe)}


@app.get("/ticker/analysis")
def ticker_analysis(ticker: str, lookback_days: int = 180):
    """The 'notes' panel under the momentum gauge: trend read (RSI/MACD/SMA/
    stochastic zones), a volatility-based projected range for two horizons
    (not a forecast -- see momo.risk.estimate_price_range's own docstring),
    recent candlestick patterns, and the same gap/liquidity risk flags used
    to filter the full-market scan. Works for any ticker via Yahoo Finance,
    no Alpaca keys required; the news-headline flag is best-effort and only
    populates when Alpaca keys are configured (headlines aren't available
    from Yahoo here)."""
    import yfinance as yf
    from momo.technicals import classify_trend, build_narrative, build_bull_bear_case
    from momo.candlesticks import detect_patterns
    from momo.quality import filter_gap_movers, filter_illiquid
    from momo.risk import estimate_price_range

    ticker = ticker.upper()
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=lookback_days)
    with _fetch_guard("Price history fetch failed"):
        hist = yf.Ticker(ticker).history(start=start, end=end + pd.Timedelta(days=1))
    if hist.empty or len(hist) < 30:
        raise HTTPException(status_code=404, detail=f"Not enough price history for {ticker} to analyze")

    hist = hist.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    close = hist["close"]
    daily_returns = close.pct_change().dropna()

    trend = classify_trend(close)

    raw_patterns = detect_patterns(hist, lookback=10)
    patterns = [
        {"date": p["date"].strftime("%Y-%m-%d"), "pattern": p["pattern"], "bias": p["bias"]}
        for p in raw_patterns
    ]

    gap_smooth = filter_gap_movers(close.to_frame(ticker), lookback_days=63, max_daily_return=0.25)
    liquid = filter_illiquid({ticker: hist}, lookback_days=20, min_dollar_volume=5_000_000)

    news_flags = []
    news_sentiment = None
    if os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY"):
        try:
            from momo.news import fetch_recent_news, detect_risk_flags, score_sentiment
            headlines = fetch_recent_news([ticker])
            news_flags = detect_risk_flags(headlines).get(ticker, [])
            # Already-fetched headlines, scored a second way -- VADER
            # sentiment (already used for the full-market scan's optional
            # news=true param) wasn't previously surfaced per-ticker here.
            news_sentiment = score_sentiment(headlines).get(ticker)
        except Exception:
            pass  # news screen is best-effort here -- the rest of the analysis still stands without it

    # OpenBB, no API key needed for either the yfinance or sec providers --
    # short interest is a genuine risk signal (heavily shorted names are
    # exactly the squeeze-prone profile BYND turned out to be), insider
    # trading is context only, reported as-is.
    ownership = None
    try:
        from openbb import obb
        from momo.ownership import short_interest_pct, high_short_interest, summarize_insider_trading

        profile = obb.equity.profile(ticker, provider="yfinance").to_df()
        short_pct = None
        if not profile.empty:
            row = profile.iloc[0]
            short_pct = short_interest_pct(row.get("shares_short"), row.get("shares_float"))

        insider_summary = None
        try:
            insider_df = obb.equity.ownership.insider_trading(ticker, provider="sec", limit=20).to_df()
            insider_summary = summarize_insider_trading(insider_df.to_dict(orient="records"), lookback_days=90)
        except Exception:
            pass  # insider filings can be sparse/missing for a given ticker -- not fatal

        ownership = {"short_interest_pct": short_pct, "insider_summary": insider_summary}
    except Exception:
        pass  # OpenBB is a nice-to-have enrichment -- the rest of the analysis still stands without it

    risk_flags = {
        "gap_risk": not bool(gap_smooth.get(ticker, False)),
        "illiquid": not bool(liquid.get(ticker, False)),
        "high_short_interest": high_short_interest(ownership["short_interest_pct"]) if ownership else False,
        "news_flags": news_flags,
    }

    # A GARCH/historical-std volatility estimate fit on a series that still
    # contains the actual gap day is dominated by that one outlier -- BYND's
    # real 32x single-day jump alone pushed its "daily vol" past 600%, which
    # isn't a usable range for anything. Winsorize at the same threshold the
    # gap filter uses so the estimate reflects the stock's normal trading
    # behavior; risk_flags.gap_risk (computed above, on the raw returns)
    # still tells the caller a gap actually happened.
    vol_returns = daily_returns.clip(lower=-0.25, upper=0.25)
    projected_range = [
        estimate_price_range(vol_returns, horizon_days=5, use_garch=True),
        estimate_price_range(vol_returns, horizon_days=20, use_garch=True),
    ]

    return {
        "ticker": ticker,
        "trend": trend,
        "projected_range": projected_range,
        "patterns": patterns,
        "risk_flags": risk_flags,
        "ownership": ownership,
        "news_sentiment": news_sentiment,
        "bull_bear_case": build_bull_bear_case(trend, raw_patterns, risk_flags, ownership),
        "summary": build_narrative(trend, raw_patterns, risk_flags, ownership),
    }


@app.get("/momentum/full-market")
def momentum_full_market(
    top_n: int = 10,
    min_price: float = 5.0,
    lookback_days: int = 400,
    equity: Optional[float] = None,
    news: bool = False,
):
    _require_alpaca_keys()
    from momo.data_layer import AlpacaDataLoader
    with _fetch_guard("Full-market scan failed"):
        report = full_market_scan(
            lookback_days=lookback_days, top_n=top_n, min_price=min_price,
            equity=equity, with_news=news, inner_loader_cls=AlpacaDataLoader,
        )
    return {"source": "alpaca", "picks": report.to_dict(orient="records")}


@app.get("/momentum/full-market/cached")
def momentum_full_market_cached(equity: Optional[float] = None):
    """Instant read of the background scanner's most recent completed pass
    over the whole tradable universe (see _full_market_scan_loop) -- this is
    what the dashboard polls every 30s. Never triggers a scan itself, so it
    stays cheap regardless of how often the frontend calls it; suggested
    dollar amounts are recomputed here from the cached weight so a changed
    equity value doesn't require a rescan."""
    _require_alpaca_keys()
    with _full_market_lock:
        picks = _full_market_cache["picks"]
        universe_size = _full_market_cache["universe_size"]
        updated_at = _full_market_cache["updated_at"]
        scanning = _full_market_cache["scanning"]
        error = _full_market_cache["error"]
        sector_concentration = _full_market_cache["sector_concentration"]

    if picks is not None and equity is not None:
        picks = [
            {**row, "suggested_dollars": round(row["target_weight"] * equity, 2)}
            for row in picks
        ]

    return {
        "source": "alpaca_full_market",
        "picks": picks or [],
        "universe_size": universe_size,
        "updated_at": updated_at,
        "scanning": scanning,
        "error": error,
        "sector_concentration": sector_concentration,
    }


@app.get("/momentum/full-market/history")
def momentum_full_market_history(limit: int = 20):
    """The audit trail the in-memory cache alone doesn't have -- past
    completed scans (see _append_scan_log), most recent first. Read
    straight from disk, independent of whatever's currently in memory."""
    limit = max(1, min(limit, _SCAN_LOG_MAX_ENTRIES))
    return {"scans": _read_scan_log(limit=limit)}


@app.get("/briefing/audio")
def briefing_audio(equity: Optional[float] = None):
    """Spoken audio of the current Top 3 momentum picks, via ElevenLabs
    text-to-speech (momo/voice.py) -- read-only, just narrates numbers the
    dashboard already computed. Requires ELEVENLABS_API_KEY; returns a
    clear 503 if it's not configured, same pattern as the Alpaca-gated
    endpoints, instead of a bare failure."""
    if not os.environ.get("ELEVENLABS_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="ELEVENLABS_API_KEY is not configured. Get a free key at elevenlabs.io "
                   "(Join free -> profile icon -> API Keys) and add it to .env to enable this.",
        )

    from momo.voice import build_briefing_text, synthesize_speech

    with _full_market_lock:
        picks = _full_market_cache["picks"]
    source = "alpaca_full_market"

    if not picks:
        # Full-market cache isn't warmed up yet (or Alpaca isn't
        # configured) -- fall back to the free watchlist so the briefing
        # still works.
        with _fetch_guard("No picks available to brief"):
            report = morning_momentum_report(
                DEFAULT_WATCHLIST, top_n=10, equity=equity, data_loader_cls=YFDataLoader,
            )
            picks = report.to_dict(orient="records")
            source = "yahoo_finance"
    elif equity is not None:
        picks = [{**p, "suggested_dollars": round(p["target_weight"] * equity, 2)} for p in picks]

    text = build_briefing_text(picks, source=source)
    try:
        audio = synthesize_speech(text)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return Response(content=audio, media_type="audio/mpeg")


@app.get("/account")
def account():
    _require_alpaca_keys()
    from momo.execution import AlpacaBroker
    with _fetch_guard("Alpaca account fetch failed"):
        broker = AlpacaBroker(paper=True)
        return {"equity": broker.get_equity(), "positions": broker.get_positions()}


@app.get("/session/status")
def session_status():
    now = pd.Timestamp.utcnow()
    open_now = is_market_open(now)
    return {
        "market_open": open_now,
        "minutes_to_close": minutes_to_close(now) if open_now else None,
        "should_flatten": should_flatten(now),
        "can_open_new_position": can_open_new_position(now),
    }


@app.get("/intraday/scan")
def intraday_scan(top_n: int = 3, lookback_minutes: int = 240):
    """Read-only preview of the current intraday momentum picks -- does not
    place orders. Needs Alpaca keys because Yahoo Finance doesn't offer
    minute-bar history."""
    _require_alpaca_keys()
    from momo.intraday import IntradayTrader

    with _fetch_guard("Intraday scan failed"):
        trader = IntradayTrader(DEFAULT_WATCHLIST, lookback_minutes=lookback_minutes, top_n=top_n, paper=True)
        weights, prices = trader.latest_target_weights()

    picks = [
        {"ticker": t, "target_weight": w, "price": prices.get(t)}
        for t, w in weights.items() if w > 0
    ]
    return {"source": "alpaca_minute_bars", "picks": picks}


@app.get("/company/info")
def company_info(ticker: str):
    import yfinance as yf
    with _fetch_guard("Company info fetch failed"):
        info = yf.Ticker(ticker).info

    name = info.get("longName") or info.get("shortName")
    if not name:
        raise HTTPException(status_code=404, detail=f"No company info for {ticker}")

    return {
        "ticker": ticker,
        "name": name,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
    }


@app.get("/ticker/intraday-analysis")
def ticker_intraday_analysis(ticker: str):
    """The gap this endpoint closes: the momentum score and Technical Read
    are both computed from DAILY bars (63-day lookback) -- for intraday
    trading, a ticker can score 0.9+ on that signal while actively
    declining right now, today, and the dashboard had no way to say so.
    This runs the same classify_trend logic (momo/technicals.py) on TODAY's
    minute bars with intraday-scaled windows (rsi=14 bars, sma=20 bars,
    stoch=14 bars -- minutes, not days, matching momo/intraday.py's
    DEFAULT_INTRADAY_SCREENER_KWARGS), so there's an actual signal for
    "what has this ticker done in today's session," not silence."""
    _require_alpaca_keys()
    from momo.data_layer import AlpacaDataLoader
    from momo.technicals import classify_trend

    ticker = ticker.upper()
    today = pd.Timestamp.today().normalize()
    start = today - pd.Timedelta(days=5)  # pad past a weekend/holiday to reach the last real session
    with _fetch_guard("Minute-bar fetch failed"):
        loader = AlpacaDataLoader([ticker], start, pd.Timestamp.utcnow(), timeframe_amount=1, timeframe_unit="Minute")
        prices = loader.load_adj_close()
    if ticker not in prices.columns:
        raise HTTPException(status_code=404, detail=f"No minute-bar history for {ticker}")

    series = prices[ticker].dropna()
    if series.empty:
        raise HTTPException(status_code=404, detail=f"No minute-bar history for {ticker}")
    last_session_date = series.index[-1].date()
    session = series[series.index.date == last_session_date]

    min_bars = 20
    if len(session) < min_bars:
        return {
            "ticker": ticker,
            "available": False,
            "reason": f"Only {len(session)} minute bars in today's session so far -- need at least {min_bars} "
                      f"for a real intraday read (this fills in through the trading day).",
        }

    trend = classify_trend(session, rsi_window=14, sma_window=min(20, len(session) - 1), stoch_window=14)
    session_change_pct = round((float(session.iloc[-1]) / float(session.iloc[0]) - 1) * 100, 2)

    direction = "up" if session_change_pct >= 0 else "down"
    summary = (
        f"Today's session ({last_session_date.isoformat()}, {len(session)} minute bars): "
        f"{ticker} is {direction} {abs(session_change_pct):.2f}% from the session open. "
        f"Intraday RSI is {trend['rsi']:.0f} ({trend['rsi_zone']}), "
        f"MACD is {'bullish' if trend['macd_state'].startswith('bullish') else 'bearish'} "
        f"({'building' if trend['macd_state'].endswith('accelerating') else 'fading'}). "
        f"This is today's session only -- a separate read from the Momentum Score above, which is a "
        f"63-day daily signal and can legitimately disagree with what's happening intraday."
    )

    return {
        "ticker": ticker,
        "available": True,
        "session_date": last_session_date.isoformat(),
        "bars": len(session),
        "session_change_pct": session_change_pct,
        "trend": trend,
        "summary": summary,
    }


_RANGE_CONFIG = {
    "1D": {"granularity": "minute", "days": 1},
    "1W": {"granularity": "minute", "days": 7},
    "3M": {"granularity": "daily", "days": 90},
    "YTD": {"granularity": "daily", "days": None},
    "1Y": {"granularity": "daily", "days": 365},
    "ALL": {"granularity": "daily", "days": 365 * 5},
}


@app.get("/momentum/history")
def momentum_history(ticker: str, time_range: str = Query("3M", alias="range")):
    """Intraday ranges (1D, 1W) use real Alpaca minute bars and need keys --
    they're "live" in the sense of using the freshest granularity available,
    refreshed on the dashboard's normal 30s poll, not a persistent
    tick-by-tick stream. Longer ranges use free daily Yahoo data."""
    time_range = time_range.upper()
    if time_range not in _RANGE_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unknown range '{time_range}'. Valid: {list(_RANGE_CONFIG)}")

    config = _RANGE_CONFIG[time_range]
    today = pd.Timestamp.today().normalize()

    if config["granularity"] == "minute":
        _require_alpaca_keys()
        from momo.data_layer import AlpacaDataLoader

        start = today - pd.Timedelta(days=config["days"] + 4)  # pad past weekends/holidays
        with _fetch_guard("Data fetch failed"):
            loader = AlpacaDataLoader(
                [ticker], start, pd.Timestamp.utcnow(), timeframe_amount=1, timeframe_unit="Minute",
            )
            prices = loader.load_adj_close()
        if ticker not in prices.columns:
            raise HTTPException(status_code=404, detail=f"No minute-bar history for {ticker}")

        series = prices[ticker].dropna()
        if time_range == "1D" and not series.empty:
            last_session_date = series.index[-1].date()
            series = series[series.index.date == last_session_date]
        points = [
            {"date": idx.strftime("%Y-%m-%d %H:%M"), "close": round(float(val), 2)}
            for idx, val in series.items()
        ]
    else:
        start = pd.Timestamp(f"{today.year}-01-01") if time_range == "YTD" else today - pd.Timedelta(days=config["days"])
        with _fetch_guard("Data fetch failed"):
            loader = YFDataLoader([ticker], start - pd.Timedelta(days=30), today, min_non_na=30)
            prices = loader.load_adj_close()
        if ticker not in prices.columns:
            raise HTTPException(status_code=404, detail=f"No price history for {ticker}")

        series = prices[ticker].dropna()
        series = series[series.index >= start]
        points = [
            {"date": idx.strftime("%Y-%m-%d"), "close": round(float(val), 2)}
            for idx, val in series.items()
        ]

    return {"ticker": ticker, "range": time_range, "granularity": config["granularity"], "points": points}


@app.get("/position/preview")
def position_preview_endpoint(
    ticker: str,
    amount: float = Query(..., gt=0, description="Dollar amount to preview"),
    horizon_days: int = 1,
    day_trade: bool = True,
    vol_lookback_days: int = 90,
):
    today = pd.Timestamp.today().normalize()
    # min_non_na must stay below the window itself -- the 200-trading-day
    # default assumes a long backtest window, not a short recent-vol read.
    with _fetch_guard("Data fetch failed"):
        loader = YFDataLoader(
            [ticker], today - pd.Timedelta(days=vol_lookback_days + 30), today,
            min_non_na=min(30, vol_lookback_days),
        )
        prices = loader.load_adj_close()

    if ticker not in prices.columns or prices[ticker].dropna().shape[0] < 2:
        raise HTTPException(status_code=404, detail=f"No usable price history for {ticker}")

    series = prices[ticker].dropna().tail(vol_lookback_days)
    entry_price = float(series.iloc[-1])
    daily_returns = series.pct_change().dropna()

    stop_loss_pct = DAY_TRADE_STOP_LOSS_PCT if day_trade else 0.08
    take_profit_pct = DAY_TRADE_TAKE_PROFIT_PCT if day_trade else 0.20
    trailing_stop_pct = 0.01 if day_trade else 0.10

    preview = position_preview(
        entry_price=entry_price, dollar_amount=amount, daily_returns=daily_returns,
        horizon_days=horizon_days, stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct, trailing_stop_pct=trailing_stop_pct,
    )
    preview["ticker"] = ticker
    preview["mode"] = "day_trade" if day_trade else "swing"
    preview["disclaimer"] = (
        "historical_range is a statistical estimate from trailing realized "
        "volatility, not a prediction -- actual moves can and do exceed it."
    )
    return preview


def _fetch_pair_prices(ticker_a: str, ticker_b: str, lookback_days: int):
    today = pd.Timestamp.today().normalize()
    # padded well past min_non_na's 200-trading-day floor plus the lookback
    # window itself, same reasoning as the other short-window endpoints
    start = today - pd.Timedelta(days=lookback_days + 260)
    loader = YFDataLoader([ticker_a, ticker_b], start, today, min_non_na=min(30, lookback_days))
    prices = loader.load_adj_close()
    missing = [t for t in (ticker_a, ticker_b) if t not in prices.columns]
    if missing:
        raise HTTPException(status_code=404, detail=f"No usable price history for {missing}")
    return prices[ticker_a].dropna(), prices[ticker_b].dropna()


@app.get("/pairs/scan")
def pairs_scan(ticker_a: str, ticker_b: str, lookback_days: int = 60,
               z_threshold: float = 2.0, min_correlation: float = 0.6):
    """Statistical relative-value scan, not arbitrage. A high |z-score| means
    this pair's spread is unusually wide relative to its own recent history --
    a probabilistic signal, not a guarantee it reverts."""
    with _fetch_guard("Data fetch failed"):
        price_a, price_b = _fetch_pair_prices(ticker_a, ticker_b, lookback_days)

    result = scan_pair(ticker_a, price_a, ticker_b, price_b,
                        lookback=lookback_days, z_threshold=z_threshold, min_correlation=min_correlation)
    result["disclaimer"] = (
        "Statistical relative-value signal based on historical correlation, not arbitrage "
        "and not a guarantee -- correlations break down and spreads can widen further before reverting."
    )
    return result


@app.get("/pairs/hedge")
def pairs_hedge(ticker_a: str, ticker_b: str, position_dollars: float = Query(..., gt=0),
                 lookback_days: int = 60):
    """Sizes a hedge in ticker_b against a position in ticker_a using the
    historical hedge ratio. Reduces exposure to the shared risk factor
    between the two -- it does not create profit, and a ratio computed on
    past data can be wrong for the next move."""
    with _fetch_guard("Data fetch failed"):
        price_a, price_b = _fetch_pair_prices(ticker_a, ticker_b, lookback_days)

    from momo.pairs import compute_hedge_ratio
    ratio = compute_hedge_ratio(price_a, price_b, lookback=lookback_days)
    if pd.isna(ratio):
        raise HTTPException(status_code=422, detail="Not enough overlapping data to compute a hedge ratio")

    hedge_dollars = compute_hedge_position(position_dollars, ratio)
    return {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "position_dollars": position_dollars,
        "hedge_ratio": round(ratio, 3),
        "hedge_dollars": hedge_dollars,
        "action": f"Short ~${abs(hedge_dollars):,.2f} of {ticker_b}" if hedge_dollars >= 0
                   else f"Long ~${abs(hedge_dollars):,.2f} of {ticker_b}",
        "disclaimer": "Risk reduction, not profit -- sized on a historical ratio that can change.",
    }


@app.get("/volatility/garch")
def volatility_garch(ticker: str, lookback_days: int = 250, horizon_days: int = 1):
    """GARCH(1,1) volatility forecast (Bollerslev 1986) alongside the plain
    historical estimate, so the two can be compared directly."""
    today = pd.Timestamp.today().normalize()
    with _fetch_guard("Data fetch failed"):
        prices = YFDataLoader([ticker], today - pd.Timedelta(days=lookback_days + 30), today,
                               min_non_na=min(50, lookback_days)).load_adj_close()
    if ticker not in prices.columns:
        raise HTTPException(status_code=404, detail=f"No usable price history for {ticker}")

    returns = prices[ticker].pct_change(fill_method=None).dropna()
    garch = forecast_volatility(returns, horizon_days=horizon_days)
    garch["ticker"] = ticker
    garch["historical_daily_vol_pct"] = round(float(returns.std()) * 100, 3)
    return garch


@app.get("/pairs/backtest")
def pairs_backtest(ticker_a: str, ticker_b: str, lookback_days: int = 60,
                    entry_z: float = 2.0, exit_z: float = 0.5, stop_z: float = 3.5,
                    window_days: int = 500):
    """Backtests the actual pairs mean-reversion strategy (not just a
    current snapshot), plus a significance test on the resulting returns."""
    today = pd.Timestamp.today().normalize()
    with _fetch_guard("Data fetch failed"):
        prices = YFDataLoader([ticker_a, ticker_b], today - pd.Timedelta(days=window_days + 260), today,
                               min_non_na=min(60, window_days)).load_adj_close()
    missing = [t for t in (ticker_a, ticker_b) if t not in prices.columns]
    if missing:
        raise HTTPException(status_code=404, detail=f"No usable price history for {missing}")

    result = backtest_pair(ticker_a, prices[ticker_a], ticker_b, prices[ticker_b],
                            lookback=lookback_days, entry_z=entry_z, exit_z=exit_z, stop_z=stop_z)
    significance = sharpe_significance_test(result["returns"])
    return {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "stats": result["stats"],
        "significance": significance,
        "disclaimer": "A backtest, not a live track record -- past reversion is not a guarantee of future reversion.",
    }


@app.get("/performance")
def performance():
    summary = performance_summary()
    daily = summary.pop("daily_returns")
    # The first row's daily_return_pct is NaN (pct_change has nothing prior
    # to compare against) -- Starlette's JSONResponse uses allow_nan=False,
    # so a bare NaN 500s the whole response. None is the correct JSON null
    # for "no prior value to compare," not a workaround. Must cast to
    # object dtype first: .where() on a float64 column silently coerces
    # None back to NaN to preserve the column's dtype.
    daily = daily.astype(object).where(pd.notna(daily), None)
    summary["daily_returns"] = daily.to_dict(orient="records") if not daily.empty else []
    return summary
