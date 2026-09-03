import sys
import os

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momo.session import is_market_open, minutes_to_close, should_flatten, can_open_new_position


def test_market_open_during_regular_session():
    # 2024-01-03 was a normal Wednesday trading day; 15:00 UTC = 10:00 ET
    assert is_market_open("2024-01-03 15:00:00")


def test_market_closed_on_weekend():
    assert not is_market_open("2024-01-06 15:00:00")  # Saturday


def test_market_closed_on_holiday():
    # New Year's Day -- a real market holiday, not just "not a weekday"
    assert not is_market_open("2024-01-01 15:00:00")


def test_market_closed_before_open():
    # 13:00 UTC = 8:00 ET, before the 9:30 ET open
    assert not is_market_open("2024-01-03 13:00:00")


def test_should_flatten_near_close():
    # market_close on 2024-01-03 is 21:00 UTC (4:00 PM ET); 20:57 is inside
    # a 5-minute buffer
    assert should_flatten("2024-01-03 20:57:00", buffer_minutes=5)


def test_should_not_flatten_mid_session():
    assert not should_flatten("2024-01-03 15:00:00", buffer_minutes=5)


def test_should_flatten_outside_market_hours():
    # nothing should be holding a day-trade position after the close
    assert should_flatten("2024-01-03 22:00:00")


def test_cannot_open_new_position_too_close_to_close():
    assert not can_open_new_position("2024-01-03 20:50:00", cutoff_minutes_before_close=15)


def test_can_open_new_position_mid_session():
    assert can_open_new_position("2024-01-03 15:00:00", cutoff_minutes_before_close=15)


def test_minutes_to_close_is_none_when_market_closed():
    assert minutes_to_close("2024-01-06 15:00:00") is None
