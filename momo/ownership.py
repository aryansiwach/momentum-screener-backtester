"""Short interest and insider trading context, pulled from OpenBB's SEC/
yfinance providers (no API key needed for either). Heavily shorted stocks
are exactly the profile prone to a violent one-day squeeze -- the BYND
lesson from earlier this session -- so short interest is a genuine risk
signal, not trivia. Insider activity is context, not a signal either way:
insiders sell for lots of reasons unrelated to the stock's outlook (taxes,
diversification, scheduled 10b5-1 plans), so this reports what happened,
not what it means."""


def short_interest_pct(shares_short, shares_float) -> float:
    """shares_short as a fraction of shares_float. None if float is
    missing/zero/NaN -- never divide by an unknown denominator and call it
    0, and never let a bare NaN leak into the response (standard JSON has
    no NaN literal; FastAPI's encoder raises on one rather than silently
    coercing it). OpenBB's DataFrame values arrive as numpy scalars (e.g.
    numpy.float64), which the JSON encoder also can't serialize directly --
    cast to native Python types explicitly rather than relying on the
    caller to remember to."""
    import math

    if shares_float is None or shares_short is None:
        return None
    shares_float = float(shares_float)
    shares_short = float(shares_short)
    if math.isnan(shares_float) or math.isnan(shares_short) or shares_float <= 0:
        return None
    return round(shares_short / shares_float, 4)


def high_short_interest(short_pct, threshold: float = 0.20) -> bool:
    """True if short interest is at or above `threshold` of float -- 20% is
    a commonly cited "heavily shorted" line (well above typical single-digit
    short interest for most large-caps)."""
    if short_pct is None:
        return False
    return bool(short_pct >= threshold)


def summarize_insider_trading(transactions: list, lookback_days: int = 90) -> dict:
    """transactions: list of {"transaction_date": str "YYYY-MM-DD",
    "acquisition_or_disposition": "Acquisition"|"Disposition"|"A"|"D",
    "securities_transacted": float, "transaction_price": float|None}
    (OpenBB's SEC-provider insider_trading schema spells these out in
    full -- "Acquisition"/"Disposition" -- but the raw SEC code is a
    single "A"/"D" letter, so both are accepted). Returns counts and
    dollar totals for buys vs. sells in the trailing window -- purely
    descriptive, no buy/sell signal implied."""
    import math
    import pandas as pd

    if not transactions:
        return {"buys": 0, "sells": 0, "buy_value": 0.0, "sell_value": 0.0, "lookback_days": lookback_days}

    def _safe_float(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.0
        return 0.0 if math.isnan(v) else v

    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=lookback_days)
    buys = sells = 0
    buy_value = sell_value = 0.0
    for t in transactions:
        date = pd.Timestamp(t.get("transaction_date"))
        if pd.isna(date) or date < cutoff:
            continue
        qty = _safe_float(t.get("securities_transacted"))
        price = _safe_float(t.get("transaction_price"))
        value = qty * price
        side = (t.get("acquisition_or_disposition") or "").strip().upper()[:1]
        if side == "A":
            buys += 1
            buy_value += value
        elif side == "D":
            sells += 1
            sell_value += value

    return {
        "buys": buys,
        "sells": sells,
        "buy_value": round(buy_value, 2),
        "sell_value": round(sell_value, 2),
        "lookback_days": lookback_days,
    }
