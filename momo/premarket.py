"""Pre-market / current-session movers -- built for the "what might move
before the bell, and why" use case, which is a genuinely different question
from the 63-day daily momentum score. Alpaca's free market-movers screener
returns whatever's moving right now (pre-market hours in, pre-market moves
out; regular hours in, regular-session moves out), but unfiltered it's
almost entirely warrants and sub-$1 instruments -- exactly the junk this
project's universe filter already excludes elsewhere. This module applies
that same standard, plus a liquidity floor, before anything here is called
a "mover" worth looking at.

Deliberately does NOT apply the gap-mover filter used for the daily scan:
that filter exists to catch a stale, no-longer-relevant single-day spike
dominating a 63-day score. A pre-market mover's whole premise is "moved a
lot, right now" -- excluding big moves here would exclude the entire
feature. The substitute check is a real one, not a weaker one: does a
legitimate liquid instrument explain the move, and does recent news say why."""


def fetch_market_movers(api_key=None, secret_key=None, top=50):
    """{"gainers": [...], "losers": [...]}, each item {"symbol", "price",
    "percent_change", "change"}. Reflects whatever session is active when
    called -- pre-market before the bell, regular-session move during
    market hours."""
    import os
    from alpaca.data.historical.screener import ScreenerClient
    from alpaca.data.requests import MarketMoversRequest

    api_key = api_key or os.environ["ALPACA_API_KEY"]
    secret_key = secret_key or os.environ["ALPACA_SECRET_KEY"]
    client = ScreenerClient(api_key, secret_key)
    movers = client.get_market_movers(MarketMoversRequest(top=top))
    return {
        "gainers": [
            {"symbol": g.symbol, "price": float(g.price), "percent_change": float(g.percent_change)}
            for g in movers.gainers
        ],
        "losers": [
            {"symbol": l.symbol, "price": float(l.price), "percent_change": float(l.percent_change)}
            for l in movers.losers
        ],
    }


def filter_clean_movers(movers: list, clean_tickers: set, min_price: float = 3.0) -> list:
    """Keeps a mover only if its symbol is in `clean_tickers` (the same
    non-junk tradable universe used everywhere else in this project -- see
    momo.universe.fetch_tradable_universe, which already excludes
    warrants/rights/units/preferred by name) and its price clears
    min_price. Most raw movers fail this -- that's the filter doing its
    job, not a bug."""
    return [m for m in movers if m["symbol"] in clean_tickers and m["price"] >= min_price]


def fetch_intraday_trend(ticker: str) -> dict | None:
    """Minute-bar intraday technical read for one ticker -- same
    classify_trend logic as the /ticker/intraday-analysis endpoint,
    factored out here so the batch mover scan can call it once per
    candidate instead of only being available one ticker at a time.
    Returns the classify_trend fields plus session_range_pct: (today's
    session high - low) as a percent of the session open, computed from
    the same minute bars -- the volatility-capacity check for "has this
    stock actually shown it can move a couple percent today," which is a
    different question from "is RSI/MACD currently pointed up." A
    technically-confirmed stock that's barely moved all session is a
    poor 2-3% day-trade candidate no matter how clean its MACD looks.
    Returns None if there isn't enough of today's session yet (no
    minute-bar history, or fewer than 20 bars in so far)."""
    import pandas as pd
    from momo.data_layer import AlpacaDataLoader
    from momo.technicals import classify_trend

    today = pd.Timestamp.today().normalize()
    start = today - pd.Timedelta(days=5)
    try:
        loader = AlpacaDataLoader([ticker], start, pd.Timestamp.utcnow(), timeframe_amount=1, timeframe_unit="Minute")
        prices = loader.load_adj_close()
    except Exception:
        return None
    if ticker not in prices.columns:
        return None
    series = prices[ticker].dropna()
    if series.empty:
        return None
    last_session_date = series.index[-1].date()
    session = series[series.index.date == last_session_date]
    if len(session) < 20:
        return None
    trend = classify_trend(session, rsi_window=14, sma_window=min(20, len(session) - 1), stoch_window=14)
    session_open = float(session.iloc[0])
    session_range_pct = round((float(session.max()) - float(session.min())) / session_open * 100, 2) if session_open else 0.0
    # MACD is a lagging read on the WHOLE session's shape -- a ticker that
    # gapped up hard at the open and has been sliding ever since can still
    # show "bullish_accelerating" (histogram still positive off that early
    # spike) while every dollar since the open has gone the other way. This
    # is the actual "is it going up right now" number: last price vs. where
    # TODAY's session opened, independent of what MACD's shape says.
    session_change_pct = round((float(session.iloc[-1]) / session_open - 1) * 100, 2) if session_open else 0.0
    return {**trend, "session_range_pct": session_range_pct, "session_change_pct": session_change_pct}


def rank_intraday_leaders(candidates: list, min_session_range_pct: float = 2.0) -> list:
    """candidates: gainers (from filter_clean_movers) each merged with an
    "intraday_trend" key (fetch_intraday_trend output, or None). A raw
    gainer list is just "moved a lot already" (vs. YESTERDAY's close) --
    this applies three independent checks on top of that:

    (1) actually up right now -- session_change_pct > 0, i.e. the last
    price is above where TODAY's own session opened. This is the one that
    matters most and is checked first: MACD is a lagging read on the
    whole session's shape, so a ticker that gapped up hard at the open and
    has been sliding ever since can still show "bullish_accelerating"
    (histogram still positive off that early spike) while every dollar
    since the open has gone the other way. Without this check a stock
    that's actively reversing can pass as a "leader" on stale MACD alone --
    a real bug this project shipped once, caught by the person actually
    using it.

    (2) direction confirmation -- MACD bullish, RSI not yet overbought --
    still applied on top, since (1) alone doesn't rule out a choppy stock
    ticking fractionally positive with fading momentum.

    (3) capacity -- today's session has actually shown a high-low range of
    at least min_session_range_pct (default 2.0%), i.e. the stock has
    demonstrated it can move that much in a single session, not just that
    it's currently pointed the right way.

    Ranked by whether it's genuinely climbing right now, then by
    direction-confirmation strength, then by realized range. This is a
    technical/volatility-confirmation filter, not a prediction -- it says
    "is up right now and has the setup and the room to keep moving," not
    "will keep going up." """
    def qualifies(c):
        trend = c.get("intraday_trend")
        return (
            c.get("percent_change", 0) > 0
            and trend is not None
            and trend.get("session_change_pct", -1) > 0
            and trend["macd_state"].startswith("bullish")
            and trend["rsi_zone"] != "overbought"
            and trend.get("session_range_pct", 0) >= min_session_range_pct
        )

    def rank_key(c):
        trend = c["intraday_trend"]
        accelerating = trend["macd_state"] == "bullish_accelerating"
        return (trend.get("session_change_pct", 0), accelerating, trend.get("session_range_pct", 0))

    qualified = [c for c in candidates if qualifies(c)]
    qualified.sort(key=rank_key, reverse=True)
    return qualified
