"""NYSE trading-session awareness for day trading -- built on a real trading
calendar (pandas_market_calendars), not a naive weekday check, because a
naive check is wrong on every market holiday and every early-close day."""

import pandas as pd
import pandas_market_calendars as mcal

_NYSE = mcal.get_calendar("NYSE")


def session_window(date) -> dict:
    """Today's (or `date`'s) market open/close in UTC, or None if it's not
    a trading day at all (weekend or holiday)."""
    date = pd.Timestamp(date)
    sched = _NYSE.schedule(start_date=date.normalize(), end_date=date.normalize())
    if sched.empty:
        return None
    return {"open": sched.iloc[0]["market_open"], "close": sched.iloc[0]["market_close"]}


def is_market_open(now) -> bool:
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    window = session_window(now)
    if window is None:
        return False
    return window["open"] <= now <= window["close"]


def minutes_to_close(now) -> float | None:
    """None if the market isn't open right now."""
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    window = session_window(now)
    if window is None or not (window["open"] <= now <= window["close"]):
        return None
    return (window["close"] - now).total_seconds() / 60.0


def should_flatten(now, buffer_minutes=5) -> bool:
    """True once a day-trade position must be closed -- inside the
    close-buffer window, or the market isn't even open (nothing should be
    holding a day-trade position outside session hours in the first
    place)."""
    minutes_left = minutes_to_close(now)
    if minutes_left is None:
        return True
    return minutes_left <= buffer_minutes


def can_open_new_position(now, cutoff_minutes_before_close=15) -> bool:
    """Don't open a fresh day trade too close to the bell -- not enough
    session left to reach a target or manage an exit."""
    minutes_left = minutes_to_close(now)
    if minutes_left is None:
        return False
    return minutes_left > cutoff_minutes_before_close
