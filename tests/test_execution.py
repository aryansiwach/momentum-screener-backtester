import sys
import os

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momo.execution import compute_target_orders, enforce_risk_gates
from momo.execution import RiskGateBlocked


def test_new_position_buys_correct_share_count():
    orders = compute_target_orders(
        target_weights={"AAPL": 0.5}, current_positions={}, prices={"AAPL": 100.0}, equity=10000.0
    )
    assert orders == {"AAPL": 50}


def test_zero_target_weight_sells_full_existing_position():
    orders = compute_target_orders(
        target_weights={}, current_positions={"AAPL": 20.0}, prices={"AAPL": 100.0}, equity=10000.0
    )
    assert orders == {"AAPL": -20}


def test_no_change_needed_produces_no_order():
    orders = compute_target_orders(
        target_weights={"AAPL": 0.5}, current_positions={"AAPL": 50.0}, prices={"AAPL": 100.0}, equity=10000.0
    )
    assert orders == {}


def test_short_target_weight_produces_negative_qty_without_overshooting():
    orders = compute_target_orders(
        target_weights={"AAPL": -0.25}, current_positions={}, prices={"AAPL": 100.0}, equity=10000.0
    )
    # trunc toward zero: -25.0 shares exactly, must not become -26 via floor
    assert orders == {"AAPL": -25}


def test_trade_below_min_notional_is_skipped():
    orders = compute_target_orders(
        target_weights={"AAPL": 0.001}, current_positions={}, prices={"AAPL": 100.0},
        equity=10000.0, min_notional=5.0
    )
    assert orders == {}


def test_missing_price_is_skipped_not_erroring():
    orders = compute_target_orders(
        target_weights={"GHOST": 0.5}, current_positions={}, prices={}, equity=10000.0
    )
    assert orders == {}


def test_fractionable_buys_partial_share_instead_of_rounding_to_zero():
    # $50 of a $309 stock is 0 whole shares but a real, tradeable fractional
    # position on an account with fractional trading enabled -- this is the
    # exact case that silently produced an empty order dict before fractional
    # support existed.
    orders = compute_target_orders(
        target_weights={"AAPL": 0.05}, current_positions={}, prices={"AAPL": 309.27},
        equity=1000.0, min_notional=1.0, fractionable=True,
    )
    expected = int((0.05 * 1000.0 / 309.27) * 1_000_000) / 1_000_000
    assert orders == {"AAPL": pytest.approx(expected, abs=1e-9)}
    assert 0 < orders["AAPL"] < 1  # a real fractional position, not zero and not a whole share


def test_non_fractionable_still_rounds_down_to_zero_shares():
    orders = compute_target_orders(
        target_weights={"AAPL": 0.05}, current_positions={}, prices={"AAPL": 309.27},
        equity=1000.0, fractionable=False,
    )
    assert orders == {}


def _empty_log():
    return pd.DataFrame(columns=["timestamp", "equity", "positions"])


def test_risk_gates_pass_with_empty_log_and_no_sectors():
    # No history to breach a circuit breaker against, and no sector map
    # supplied -- should not raise.
    enforce_risk_gates(_empty_log(), equity=10000.0, target_weights={"AAPL": 0.5})


def test_risk_gates_block_on_circuit_breaker_daily_loss():
    now = pd.Timestamp.now(tz="UTC")
    log = pd.DataFrame({"timestamp": [now], "equity": [10000.0], "positions": [{}]})
    with pytest.raises(RiskGateBlocked, match="Circuit breaker"):
        enforce_risk_gates(log, equity=9600.0, target_weights={"AAPL": 0.5})  # -4% today, breaches 3% default


def test_risk_gates_block_on_sector_concentration():
    sectors = {"AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology"}
    weights = {"AAPL": 0.34, "MSFT": 0.33, "GOOGL": 0.33}  # 100% Technology
    with pytest.raises(RiskGateBlocked, match="Sector concentration"):
        enforce_risk_gates(_empty_log(), equity=10000.0, target_weights=weights, ticker_sectors=sectors)


def test_risk_gates_pass_when_sectors_diversified():
    sectors = {"AAPL": "Technology", "JPM": "Financials", "XOM": "Energy"}
    weights = {"AAPL": 0.34, "JPM": 0.33, "XOM": 0.33}
    enforce_risk_gates(_empty_log(), equity=10000.0, target_weights=weights, ticker_sectors=sectors)
