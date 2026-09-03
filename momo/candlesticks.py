"""Classic candlestick pattern detection over an OHLC DataFrame. These are
shape-recognition heuristics with a long history in technical analysis, not
statistically validated signals -- momo/significance.py's tools exist
precisely because "looks like a pattern" and "actually predicts the next
move" are different claims. Treat pattern names as descriptive labels for
what the candles look like, not as a tested edge."""

import pandas as pd


def _body(df):
    return (df["close"] - df["open"]).abs()


def _range(df):
    return (df["high"] - df["low"]).replace(0, float("nan"))


def _upper_shadow(df):
    return df["high"] - df[["open", "close"]].max(axis=1)


def _lower_shadow(df):
    return df[["open", "close"]].min(axis=1) - df["low"]


def detect_patterns(ohlc: pd.DataFrame, lookback: int = 10) -> list:
    """Scan the trailing `lookback` candles and return every pattern found,
    most recent first, as {"date", "pattern", "bias"} -- bias is "bullish",
    "bearish", or "neutral" (doji)."""
    df = ohlc.tail(lookback + 3).copy()  # a little extra context for 3-candle patterns
    if len(df) < 3:
        return []

    body = _body(df)
    rng = _range(df)
    upper = _upper_shadow(df)
    lower = _lower_shadow(df)
    bullish = df["close"] > df["open"]
    bearish = df["close"] < df["open"]

    found = []
    idx = df.index
    for i in range(2, len(df)):
        date = idx[i]
        row_body, row_rng = body.iloc[i], rng.iloc[i]
        if pd.isna(row_rng) or row_rng == 0:
            continue

        # Doji: body is a sliver of the day's range.
        if row_body / row_rng < 0.1:
            found.append({"date": date, "pattern": "doji", "bias": "neutral"})

        # Hammer: small body in the upper part of the range, long lower
        # shadow, little/no upper shadow.
        if row_body / row_rng < 0.35 and lower.iloc[i] >= 2 * row_body and upper.iloc[i] <= row_body:
            found.append({"date": date, "pattern": "hammer", "bias": "bullish"})

        # Shooting star: mirror of hammer, long upper shadow.
        if row_body / row_rng < 0.35 and upper.iloc[i] >= 2 * row_body and lower.iloc[i] <= row_body:
            found.append({"date": date, "pattern": "shooting_star", "bias": "bearish"})

        # Bullish engulfing: prior red candle's body fully engulfed by a
        # green candle.
        if (
            bearish.iloc[i - 1] and bullish.iloc[i]
            and df["open"].iloc[i] <= df["close"].iloc[i - 1]
            and df["close"].iloc[i] >= df["open"].iloc[i - 1]
        ):
            found.append({"date": date, "pattern": "bullish_engulfing", "bias": "bullish"})

        # Bearish engulfing: mirror.
        if (
            bullish.iloc[i - 1] and bearish.iloc[i]
            and df["open"].iloc[i] >= df["close"].iloc[i - 1]
            and df["close"].iloc[i] <= df["open"].iloc[i - 1]
        ):
            found.append({"date": date, "pattern": "bearish_engulfing", "bias": "bearish"})

        # Morning star: long red, small-bodied middle candle, long green
        # closing back into the first candle's body -- classic 3-candle
        # bottoming pattern.
        if (
            bearish.iloc[i - 2] and body.iloc[i - 2] / rng.iloc[i - 2] > 0.5
            and body.iloc[i - 1] / rng.iloc[i - 1] < 0.35
            and bullish.iloc[i] and body.iloc[i] / rng.iloc[i] > 0.5
            and df["close"].iloc[i] > (df["open"].iloc[i - 2] + df["close"].iloc[i - 2]) / 2
        ):
            found.append({"date": date, "pattern": "morning_star", "bias": "bullish"})

        # Evening star: mirror topping pattern.
        if (
            bullish.iloc[i - 2] and body.iloc[i - 2] / rng.iloc[i - 2] > 0.5
            and body.iloc[i - 1] / rng.iloc[i - 1] < 0.35
            and bearish.iloc[i] and body.iloc[i] / rng.iloc[i] > 0.5
            and df["close"].iloc[i] < (df["open"].iloc[i - 2] + df["close"].iloc[i - 2]) / 2
        ):
            found.append({"date": date, "pattern": "evening_star", "bias": "bearish"})

        # Three white soldiers: three consecutive strong green candles, each
        # closing higher than the last.
        if (
            all(bullish.iloc[i - k] for k in range(3))
            and all(body.iloc[i - k] / rng.iloc[i - k] > 0.5 for k in range(3))
            and df["close"].iloc[i] > df["close"].iloc[i - 1] > df["close"].iloc[i - 2]
        ):
            found.append({"date": date, "pattern": "three_white_soldiers", "bias": "bullish"})

        # Three black crows: mirror.
        if (
            all(bearish.iloc[i - k] for k in range(3))
            and all(body.iloc[i - k] / rng.iloc[i - k] > 0.5 for k in range(3))
            and df["close"].iloc[i] < df["close"].iloc[i - 1] < df["close"].iloc[i - 2]
        ):
            found.append({"date": date, "pattern": "three_black_crows", "bias": "bearish"})

    found.sort(key=lambda p: p["date"], reverse=True)
    return found
