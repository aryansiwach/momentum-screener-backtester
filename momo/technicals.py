"""Turns the same indicators the Screener already computes (momo/indicators.py)
into a per-ticker, human-readable trend read, plus a plain-English summary.
Descriptive only -- it states what the indicators currently show, using the
standard textbook zone conventions (RSI/stochastic 70-30 and 80-20), not a
buy/sell recommendation. Nothing here predicts what happens next."""

import pandas as pd

from momo.indicators import rsi, macd, sma, stochastic


def classify_trend(prices: pd.Series, rsi_window: int = 14, sma_window: int = 50, stoch_window: int = 14) -> dict:
    """prices: a single ticker's close-price Series, oldest to newest, at
    ANY bar frequency -- daily by default (rsi/stoch=14, sma=50 means "14
    days"/"50 days"), but the same math applies unchanged to minute bars
    with intraday-scaled windows (see build_intraday_analysis below, which
    passes sma_window=20 on minute bars to mean "20 minutes", not "20
    days"). The windows are the only thing that makes this a daily-momentum
    read versus an intraday one -- same functions, different granularity."""
    df = prices.to_frame("px")
    rsi_val = float(rsi(df, window=rsi_window)["px"].iloc[-1])
    macd_line, signal_line, hist = macd(df)
    hist_now = float(hist["px"].iloc[-1])
    hist_prev = float(hist["px"].iloc[-2]) if len(hist) > 1 else hist_now
    sma_val = float(sma(df, window=sma_window)["px"].iloc[-1])
    last_price = float(df["px"].iloc[-1])
    k, _ = stochastic(df, window=stoch_window)
    stoch_k = float(k["px"].iloc[-1])

    # A single missing/NaN price bar inside the lookback window (a halted
    # session, a reverse-split artifact from the data vendor -- happened
    # for real with BIAF's 2026-08-24 gap right at its ~13x split jump)
    # poisons a rolling min/max for the full window afterward: pandas'
    # rolling().min()/.max() need every point in the window non-NaN by
    # default, unlike the EWM-based indicators (rsi, macd) which recover
    # after one bad point. An unguarded NaN here isn't just wrong, it's a
    # JSON-serialization crash for the whole endpoint (NaN isn't valid
    # JSON) -- so stochastic_k needs the same "unavailable, not zero"
    # guard price_vs_sma50_pct already has below.
    stoch_k_valid = stoch_k == stoch_k  # NaN != NaN is the cheapest isnan check
    stoch_zone = (
        "overbought" if stoch_k_valid and stoch_k >= 80
        else "oversold" if stoch_k_valid and stoch_k <= 20
        else "neutral" if stoch_k_valid
        else None
    )

    rsi_zone = "overbought" if rsi_val >= 70 else "oversold" if rsi_val <= 30 else "neutral"

    if hist_now >= 0:
        macd_state = "bullish_accelerating" if hist_now >= hist_prev else "bullish_decelerating"
    else:
        macd_state = "bearish_accelerating" if hist_now <= hist_prev else "bearish_decelerating"

    price_vs_sma50_pct = ((last_price / sma_val) - 1) * 100 if sma_val and not pd.isna(sma_val) else None

    return {
        "last_price": round(last_price, 2),
        "rsi": round(rsi_val, 1),
        "rsi_zone": rsi_zone,
        "macd_histogram": round(hist_now, 4),
        "macd_state": macd_state,
        "price_vs_sma50_pct": round(price_vs_sma50_pct, 2) if price_vs_sma50_pct is not None else None,
        "stochastic_k": round(stoch_k, 1) if stoch_k_valid else None,
        "stochastic_zone": stoch_zone,
    }


_MACD_TEXT = {
    "bullish_accelerating": "MACD histogram positive and rising -- upward momentum is building",
    "bullish_decelerating": "MACD histogram positive but shrinking -- upward momentum is fading",
    "bearish_accelerating": "MACD histogram negative and falling -- downward pressure is building",
    "bearish_decelerating": "MACD histogram negative but shrinking -- downward pressure is fading",
}


def build_narrative(trend: dict, patterns: list, risk_flags: dict, ownership: dict = None) -> str:
    """Assembles the objective bullet-point read shown under the momentum
    score. Every clause traces to a specific number or flag above it --
    nothing here is inferred beyond what the data shows. `ownership` (short
    interest / insider trading, see momo.ownership) is optional -- callers
    without OpenBB data available just omit it."""
    lines = []

    if trend["price_vs_sma50_pct"] is not None:
        direction = "above" if trend["price_vs_sma50_pct"] >= 0 else "below"
        lines.append(f"Price is {abs(trend['price_vs_sma50_pct']):.1f}% {direction} its 50-day average.")

    rsi_desc = {
        "overbought": f"RSI is {trend['rsi']:.0f} -- overbought territory (>=70), often followed by a pause or pullback.",
        "oversold": f"RSI is {trend['rsi']:.0f} -- oversold territory (<=30), often followed by a bounce or continued weakness.",
        "neutral": f"RSI is {trend['rsi']:.0f} -- neutral, no overbought/oversold extreme.",
    }
    lines.append(rsi_desc[trend["rsi_zone"]])
    lines.append(_MACD_TEXT[trend["macd_state"]] + ".")

    if trend["stochastic_zone"] is None:
        lines.append("Stochastic %K is unavailable right now -- a gap in the recent price history is blocking that read.")
    elif trend["stochastic_zone"] != "neutral":
        lines.append(f"Stochastic %K at {trend['stochastic_k']:.0f} confirms {trend['stochastic_zone']} conditions.")

    if patterns:
        top = patterns[0]
        lines.append(
            f"Most recent candlestick pattern: {top['pattern'].replace('_', ' ')} ({top['bias']}), "
            f"a shape-recognition signal, not a tested predictor."
        )
    else:
        lines.append("No notable candlestick pattern in the recent sessions.")

    flag_bits = []
    if risk_flags.get("gap_risk"):
        flag_bits.append("a single-day gap/spike in its recent history")
    if risk_flags.get("illiquid"):
        flag_bits.append("thin trading liquidity")
    if risk_flags.get("high_short_interest"):
        flag_bits.append("heavy short interest")
    if risk_flags.get("news_flags"):
        flag_bits.append("flagged recent headlines (" + ", ".join(risk_flags["news_flags"]) + ")")
    if flag_bits:
        lines.append("Risk flags: " + "; ".join(flag_bits) + ".")
    else:
        lines.append("No gap, liquidity, short-interest, or headline risk flags detected.")

    if ownership:
        short_pct = ownership.get("short_interest_pct")
        if short_pct is not None:
            lines.append(f"Short interest is {short_pct * 100:.1f}% of float.")
        insiders = ownership.get("insider_summary")
        if insiders and (insiders["buys"] or insiders["sells"]):
            lines.append(
                f"Insider activity in the last {insiders['lookback_days']} days: "
                f"{insiders['buys']} buy transaction(s), {insiders['sells']} sell transaction(s) "
                f"-- reported as-is, not a buy/sell signal (insiders sell for many reasons, "
                f"including scheduled 10b5-1 plans and taxes)."
            )

    return " ".join(lines)


# Overextension threshold for "price has run far above its 50-day average"
# -- a standard technical-analysis rule of thumb, not statistically fit to
# this project's data (nothing here is optimized against a backtest).
_OVEREXTENDED_PCT = 15.0


def build_bull_bear_case(trend: dict, patterns: list, risk_flags: dict, ownership: dict = None) -> dict:
    """Structures the same computed signals build_narrative already uses
    into an explicit bull case / bear case, adapted from the bull-vs-bear
    researcher pattern in Tauric Research's TradingAgents (arXiv:2412.20138)
    -- their version debates via LLM agents; this is the same idea done as
    plain rules over data this project already computes, no LLM call and no
    added cost. Every point traces to a specific field; nothing is inferred
    beyond what's already in `trend`/`patterns`/`risk_flags`/`ownership`.
    This is a framing device for reading the same evidence from both sides,
    not two independent opinions -- and not a recommendation either way."""
    bull, bear = [], []

    if trend["macd_state"].startswith("bullish"):
        note = "and building" if trend["macd_state"] == "bullish_accelerating" else "but fading"
        bull.append(f"MACD histogram is positive {note} -- trend momentum favors the bulls.")
    else:
        note = "and building" if trend["macd_state"] == "bearish_accelerating" else "but fading"
        bear.append(f"MACD histogram is negative {note} -- trend momentum favors the bears.")

    if trend["rsi_zone"] == "oversold":
        bull.append(f"RSI at {trend['rsi']:.0f} is oversold -- a classic setup for a bounce, though weakness can persist.")
    elif trend["rsi_zone"] == "overbought":
        bear.append(f"RSI at {trend['rsi']:.0f} is overbought -- often precedes a pause or pullback.")

    if trend["stochastic_zone"] == "oversold":
        bull.append(f"Stochastic %K at {trend['stochastic_k']:.0f} is oversold, reinforcing the bounce case.")
    elif trend["stochastic_zone"] == "overbought":
        bear.append(f"Stochastic %K at {trend['stochastic_k']:.0f} is overbought, reinforcing the pullback case.")

    if trend["price_vs_sma50_pct"] is not None:
        pct = trend["price_vs_sma50_pct"]
        if pct >= _OVEREXTENDED_PCT:
            bear.append(f"Price is {pct:.1f}% above its 50-day average -- extended, with reversion risk.")
        elif pct > 0:
            bull.append(f"Price is {pct:.1f}% above its 50-day average -- an established uptrend.")
        elif pct <= -_OVEREXTENDED_PCT:
            bull.append(f"Price is {abs(pct):.1f}% below its 50-day average -- extended to the downside, with bounce potential.")
        else:
            bear.append(f"Price is {abs(pct):.1f}% below its 50-day average -- trend is below its recent trajectory.")

    if patterns:
        top = patterns[0]
        label = f"Most recent candlestick pattern is {top['pattern'].replace('_', ' ')}."
        if top["bias"] == "bullish":
            bull.append(label)
        elif top["bias"] == "bearish":
            bear.append(label)

    if risk_flags.get("gap_risk"):
        bear.append("A single-day gap/spike sits in its recent history -- elevated tail risk.")
    if risk_flags.get("illiquid"):
        bear.append("Thin trading liquidity -- wider real-world spreads and slippage than the quoted price implies.")
    if risk_flags.get("high_short_interest"):
        bear.append("Short interest is heavy -- the market has taken a meaningfully bearish position (this also means a squeeze is mechanically possible, which cuts both ways, not a reason to chase it).")
    for flag in risk_flags.get("news_flags", []):
        bear.append(f"Recent headlines are flagged: {flag}.")

    if ownership:
        insiders = ownership.get("insider_summary")
        if insiders and insiders["sells"] > insiders["buys"]:
            bear.append(
                f"Insiders sold more than they bought in the last {insiders['lookback_days']} days "
                f"({insiders['sells']} sells vs. {insiders['buys']} buys) -- context, not a signal on its own."
            )
        elif insiders and insiders["buys"] > insiders["sells"]:
            bull.append(
                f"Insiders bought more than they sold in the last {insiders['lookback_days']} days "
                f"({insiders['buys']} buys vs. {insiders['sells']} sells) -- context, not a signal on its own."
            )

    return {"bull_points": bull, "bear_points": bear}
