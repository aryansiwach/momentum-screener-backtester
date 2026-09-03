import sys
import os

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momo.risk import (
    evaluate_exit, estimate_price_range, position_preview, check_pdt_compliance,
    check_drawdown_circuit_breaker, check_sector_concentration,
)


def test_hold_when_pnl_within_bounds():
    result = evaluate_exit(entry_price=100, current_price=105)
    assert result["action"] == "HOLD"


def test_sell_on_stop_loss():
    result = evaluate_exit(entry_price=100, current_price=91, stop_loss_pct=0.08)
    assert result == {"action": "SELL", "reason": "stop_loss", "pnl_pct": -0.09}


def test_sell_on_take_profit():
    result = evaluate_exit(entry_price=100, current_price=121, take_profit_pct=0.20)
    assert result["action"] == "SELL"
    assert result["reason"] == "take_profit"


def test_sell_on_trailing_stop_after_running_up():
    # ran up to 130 (high_since_entry), pulled back to 115: 11.5% off the high
    result = evaluate_exit(entry_price=100, current_price=115, high_since_entry=130,
                            take_profit_pct=0.50, trailing_stop_pct=0.10)
    assert result["action"] == "SELL"
    assert result["reason"] == "trailing_stop"


def test_no_trailing_stop_trigger_before_ever_being_profitable():
    # never exceeded entry price, so trailing-stop logic must not fire
    result = evaluate_exit(entry_price=100, current_price=97, high_since_entry=100,
                            stop_loss_pct=0.08, trailing_stop_pct=0.02)
    assert result["action"] == "HOLD"


def test_estimate_price_range_scales_with_sqrt_horizon():
    # fixed daily return series -> known std, so the horizon scaling
    # (std * sqrt(days)) is checkable to a tolerance rather than just "some number"
    returns = pd.Series([0.01, -0.01] * 30)
    r10 = estimate_price_range(returns, horizon_days=10)
    r40 = estimate_price_range(returns, horizon_days=40)
    # 40 days is 4x the variance of 10 days -> sqrt(4) = 2x the vol
    assert abs(r40["horizon_vol_pct"] - r10["horizon_vol_pct"] * 2) < 0.02


def test_pdt_never_blocks_since_finra_eliminated_the_rule():
    # FINRA eliminated the PDT designation and $25k threshold June 4, 2026
    # (Regulatory Notice 26-10) -- no equity/day-trade-count combination
    # should block a trade anymore.
    result = check_pdt_compliance(equity=10000, day_trades_in_last_5_sessions=10)
    assert result["allowed"] is True


def test_pdt_no_warning_above_low_equity_threshold():
    result = check_pdt_compliance(equity=30000, day_trades_in_last_5_sessions=10)
    assert result["warning"] is None


def test_pdt_no_warning_under_three_day_trades():
    result = check_pdt_compliance(equity=10000, day_trades_in_last_5_sessions=2)
    assert result["warning"] is None


def test_pdt_warns_on_frequent_day_trading_with_low_equity():
    result = check_pdt_compliance(equity=10000, day_trades_in_last_5_sessions=3)
    assert result["allowed"] is True
    assert "10,000" in result["warning"]


def test_position_preview_computes_shares_and_exit_prices():
    returns = pd.Series([0.005, -0.004, 0.006, -0.003] * 20)
    preview = position_preview(
        entry_price=100.0, dollar_amount=1000.0, daily_returns=returns,
        stop_loss_pct=0.08, take_profit_pct=0.20, trailing_stop_pct=0.10,
    )
    assert preview["shares"] == 10.0
    assert preview["stop_loss_price"] == 92.0
    assert preview["take_profit_price"] == 120.0
    assert preview["historical_range"]["estimated_high_price"] > 100.0
    assert preview["historical_range"]["estimated_low_price"] < 100.0


def _equity_log(rows):
    return pd.DataFrame(rows, columns=["timestamp", "equity", "positions"])


def test_circuit_breaker_allows_trading_with_empty_log():
    result = check_drawdown_circuit_breaker(_equity_log([]), current_equity=1000.0)
    assert result["halted"] is False


def test_circuit_breaker_halts_on_daily_loss_breach():
    now = pd.Timestamp.now(tz="UTC")
    log = _equity_log([[now.replace(hour=9, minute=30), 1000.0, {}]])
    # down 5% today, over the 3% default daily limit
    result = check_drawdown_circuit_breaker(log, current_equity=950.0, max_daily_loss_pct=0.03)
    assert result["halted"] is True
    assert "Daily loss" in result["reason"]


def test_circuit_breaker_allows_trading_within_daily_limit():
    now = pd.Timestamp.now(tz="UTC")
    log = _equity_log([[now.replace(hour=9, minute=30), 1000.0, {}]])
    # down 1%, under the 3% default limit
    result = check_drawdown_circuit_breaker(log, current_equity=990.0, max_daily_loss_pct=0.03)
    assert result["halted"] is False


def test_circuit_breaker_halts_on_weekly_loss_breach():
    week_ago = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=3)
    log = _equity_log([[week_ago, 1000.0, {}]])
    # down 8% over the week, over the 6% default weekly limit; no row from today
    # so the daily check is skipped and only the weekly check applies
    result = check_drawdown_circuit_breaker(log, current_equity=920.0, max_weekly_loss_pct=0.06)
    assert result["halted"] is True
    assert "Weekly loss" in result["reason"]


def test_sector_concentration_flags_breach():
    sectors = {"AAPL": "Technology", "MSFT": "Technology", "JPM": "Financials"}
    weights = {"AAPL": 0.3, "MSFT": 0.3, "JPM": 0.2}
    result = check_sector_concentration(sectors, weights, max_sector_weight=0.4)
    assert result["within_limits"] is False
    assert "Technology" in result["breaches"]
    assert result["breaches"]["Technology"] == 0.6


def test_sector_concentration_within_limits_when_diversified():
    sectors = {"AAPL": "Technology", "JPM": "Financials", "XOM": "Energy"}
    weights = {"AAPL": 0.3, "JPM": 0.3, "XOM": 0.3}
    result = check_sector_concentration(sectors, weights, max_sector_weight=0.4)
    assert result["within_limits"] is True
    assert result["breaches"] == {}


def test_sector_concentration_unknown_sector_for_missing_ticker():
    result = check_sector_concentration({}, {"XYZ": 0.5}, max_sector_weight=0.4)
    assert result["sector_weights"]["Unknown"] == 0.5
    assert "Unknown" in result["breaches"]
