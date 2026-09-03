import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momo.monitor import check_exits


def test_hold_within_bounds():
    positions = {"AAPL": {"entry_price": 100.0, "current_price": 102.0}}
    results = check_exits(positions, highs={}, persist=False)
    assert results == [{"ticker": "AAPL", "action": "HOLD", "reason": None, "pnl_pct": 0.02}]


def test_stop_loss_triggers_sell():
    positions = {"AAPL": {"entry_price": 100.0, "current_price": 90.0}}
    results = check_exits(positions, stop_loss_pct=0.08, highs={}, persist=False)
    assert results[0]["action"] == "SELL"
    assert results[0]["reason"] == "stop_loss"


def test_take_profit_triggers_sell():
    positions = {"AAPL": {"entry_price": 100.0, "current_price": 125.0}}
    results = check_exits(positions, take_profit_pct=0.20, highs={}, persist=False)
    assert results[0]["action"] == "SELL"
    assert results[0]["reason"] == "take_profit"


def test_trailing_stop_uses_high_carried_from_prior_check():
    # Position ran up to $130 on a prior check (recorded in `highs`), then
    # pulled back to $115 -- a 11.5% drawdown from the $130 high should
    # trigger a 10% trailing stop even though price is still above entry.
    positions = {"AAPL": {"entry_price": 100.0, "current_price": 115.0}}
    results = check_exits(positions, trailing_stop_pct=0.10, highs={"AAPL": 130.0}, persist=False)
    assert results[0]["action"] == "SELL"
    assert results[0]["reason"] == "trailing_stop"


def test_high_water_mark_carries_forward_across_calls():
    # First call establishes a new high (below the take-profit threshold,
    # so it doesn't sell for that reason); a second call at a lower price
    # should still trail from that first-call high, not reset to the
    # second call's own (lower) price.
    positions_up = {"AAPL": {"entry_price": 100.0, "current_price": 115.0}}
    first = check_exits(positions_up, take_profit_pct=0.20, highs={}, persist=False)
    assert first[0]["action"] == "HOLD"

    positions_down = {"AAPL": {"entry_price": 100.0, "current_price": 102.0}}
    second = check_exits(
        positions_down, take_profit_pct=0.20, trailing_stop_pct=0.10, highs={"AAPL": 115.0}, persist=False
    )
    assert second[0]["action"] == "SELL"
    assert second[0]["reason"] == "trailing_stop"


def test_closed_position_is_dropped_from_tracked_highs():
    # A ticker present in `highs` but no longer in `positions` (already
    # sold) shouldn't linger and shouldn't appear in the results.
    results = check_exits({}, highs={"OLD": 50.0}, persist=False)
    assert results == []


def test_multiple_positions_evaluated_independently():
    positions = {
        "WINNER": {"entry_price": 100.0, "current_price": 130.0},
        "LOSER": {"entry_price": 100.0, "current_price": 85.0},
    }
    results = check_exits(positions, take_profit_pct=0.20, stop_loss_pct=0.08, highs={}, persist=False)
    by_ticker = {r["ticker"]: r for r in results}
    assert by_ticker["WINNER"]["action"] == "SELL"
    assert by_ticker["WINNER"]["reason"] == "take_profit"
    assert by_ticker["LOSER"]["action"] == "SELL"
    assert by_ticker["LOSER"]["reason"] == "stop_loss"
