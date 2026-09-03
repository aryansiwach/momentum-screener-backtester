import sys
import os

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momo.candlesticks import detect_patterns


def _ohlc(rows):
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="D")
    df = pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close"])
    return df


def _pattern_names(found):
    return {p["pattern"] for p in found}


def test_doji_detected():
    rows = [
        [100, 105, 95, 101],
        [101, 106, 96, 102],
        [102, 108, 92, 102.05],  # open ~= close, wide range
    ]
    found = detect_patterns(_ohlc(rows), lookback=10)
    assert "doji" in _pattern_names(found)


def test_hammer_detected():
    rows = [
        [110, 111, 100, 101],
        [101, 102, 92, 93],
        [100, 103, 90, 102],  # small body near top, long lower shadow
    ]
    found = detect_patterns(_ohlc(rows), lookback=10)
    matches = [p for p in found if p["pattern"] == "hammer"]
    assert matches and matches[0]["bias"] == "bullish"


def test_shooting_star_detected():
    rows = [
        [90, 100, 89, 99],
        [99, 108, 98, 107],
        [100, 110, 97, 98],  # small body near bottom, long upper shadow
    ]
    found = detect_patterns(_ohlc(rows), lookback=10)
    matches = [p for p in found if p["pattern"] == "shooting_star"]
    assert matches and matches[0]["bias"] == "bearish"


def test_bullish_engulfing_detected():
    rows = [
        [105, 106, 94, 96],
        [100, 101, 89, 95],  # bearish, body 100->95
        [94, 102, 93, 101],  # bullish, opens below prior close, closes above prior open
    ]
    found = detect_patterns(_ohlc(rows), lookback=10)
    matches = [p for p in found if p["pattern"] == "bullish_engulfing"]
    assert matches and matches[0]["bias"] == "bullish"


def test_bearish_engulfing_detected():
    rows = [
        [95, 106, 94, 105],
        [95, 101, 89, 100],  # bullish, body 95->100
        [101, 102, 92, 94],  # bearish, opens above prior close, closes below prior open
    ]
    found = detect_patterns(_ohlc(rows), lookback=10)
    matches = [p for p in found if p["pattern"] == "bearish_engulfing"]
    assert matches and matches[0]["bias"] == "bearish"


def test_morning_star_detected():
    rows = [
        [100, 101, 89, 90],   # long red, body 10 of range 12
        [89, 90, 87, 88],     # small body
        [89, 99, 88, 98],     # long green closing above midpoint of first candle
    ]
    found = detect_patterns(_ohlc(rows), lookback=10)
    matches = [p for p in found if p["pattern"] == "morning_star"]
    assert matches and matches[0]["bias"] == "bullish"


def test_evening_star_detected():
    rows = [
        [90, 101, 89, 100],   # long green
        [101, 102, 99, 100.5],  # small body
        [101, 102, 91, 92],   # long red closing below midpoint of first candle
    ]
    found = detect_patterns(_ohlc(rows), lookback=10)
    matches = [p for p in found if p["pattern"] == "evening_star"]
    assert matches and matches[0]["bias"] == "bearish"


def test_three_white_soldiers_detected():
    rows = [
        [90, 96, 89, 95],
        [95, 101, 94, 100],
        [100, 106, 99, 105],
    ]
    found = detect_patterns(_ohlc(rows), lookback=10)
    matches = [p for p in found if p["pattern"] == "three_white_soldiers"]
    assert matches and matches[0]["bias"] == "bullish"


def test_three_black_crows_detected():
    rows = [
        [105, 106, 99, 100],
        [100, 101, 94, 95],
        [95, 96, 89, 90],
    ]
    found = detect_patterns(_ohlc(rows), lookback=10)
    matches = [p for p in found if p["pattern"] == "three_black_crows"]
    assert matches and matches[0]["bias"] == "bearish"


def test_too_little_history_returns_empty():
    rows = [[100, 101, 99, 100.5]]
    found = detect_patterns(_ohlc(rows), lookback=10)
    assert found == []


def test_boring_flat_candles_produce_no_false_positives():
    rows = [[100, 100.5, 99.5, 100.2] for _ in range(10)]
    found = detect_patterns(_ohlc(rows), lookback=10)
    # small consistent bodies with tight ranges shouldn't spuriously fire
    # multi-candle reversal patterns
    assert "morning_star" not in _pattern_names(found)
    assert "evening_star" not in _pattern_names(found)
