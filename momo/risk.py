"""Systematic exit rules. No claim to predict the exact minute to sell --
these are price-triggered guardrails: a stop-loss, a take-profit, and a
trailing-stop that locks in gains as a winning position runs."""

import math

import pandas as pd

# Day-trade exits are on a completely different scale than swing/position
# exits (the 8/20/10% defaults elsewhere in this file) -- a day trade is
# meant to be flat by the close, not held for weeks, so both the target and
# the stop need to be small enough to actually hit within a single session.
DAY_TRADE_TAKE_PROFIT_PCT = 0.03
DAY_TRADE_STOP_LOSS_PCT = 0.015


def estimate_price_range(daily_returns, horizon_days=30, z=1.0, use_garch=False):
    """A historical-volatility range, not a forecast. Default: realized
    daily return std scaled by sqrt(time) under a random-walk assumption --
    the standard textbook approximation, flat regardless of whether the
    market's been calm or turbulent lately. use_garch=True swaps in a
    GARCH(1,1) forecast (Bollerslev 1986) that responds to recent
    volatility clustering instead; falls back to the flat estimate if
    GARCH doesn't converge (e.g. too little history)."""
    vol_model = "historical_std"
    daily_vol = float(daily_returns.std())
    if use_garch:
        from momo.volatility import forecast_volatility
        garch_result = forecast_volatility(daily_returns, horizon_days=1)
        if garch_result.get("converged"):
            daily_vol = garch_result["forecast_daily_vol_pct"] / 100
            vol_model = "garch(1,1)"

    horizon_vol = daily_vol * math.sqrt(horizon_days)
    return {
        "daily_vol_pct": round(daily_vol * 100, 3),
        "horizon_days": horizon_days,
        "horizon_vol_pct": round(horizon_vol * 100, 2),
        "expected_up_pct": round(z * horizon_vol * 100, 2),
        "expected_down_pct": round(-z * horizon_vol * 100, 2),
        "vol_model": vol_model,
    }


def position_preview(entry_price, dollar_amount, daily_returns, horizon_days=30,
                      stop_loss_pct=0.08, take_profit_pct=0.20, trailing_stop_pct=0.10,
                      use_garch=False):
    shares = dollar_amount / entry_price
    price_range = estimate_price_range(daily_returns, horizon_days=horizon_days, use_garch=use_garch)
    return {
        "entry_price": round(entry_price, 2),
        "shares": round(shares, 4),
        "dollar_amount": dollar_amount,
        "stop_loss_price": round(entry_price * (1 - stop_loss_pct), 2),
        "stop_loss_pct": stop_loss_pct * 100,
        "take_profit_price": round(entry_price * (1 + take_profit_pct), 2),
        "take_profit_pct": take_profit_pct * 100,
        "trailing_stop_pct": trailing_stop_pct * 100,
        "historical_range": {
            **price_range,
            "estimated_high_price": round(entry_price * (1 + price_range["expected_up_pct"] / 100), 2),
            "estimated_low_price": round(entry_price * (1 + price_range["expected_down_pct"] / 100), 2),
        },
    }


def check_pdt_compliance(equity, day_trades_in_last_5_sessions, low_equity_warning=25000.0):
    """FINRA eliminated the Pattern Day Trader designation and its $25,000
    minimum-equity requirement effective June 4, 2026 (Regulatory Notice
    26-10) -- a margin account is no longer restricted purely for racking up
    a 4th day trade in 5 sessions under that threshold. It was replaced with
    a real-time "intraday margin deficit" standard computed per-transaction
    against actual account exposure, which is the broker's margin engine's
    job, not something derivable from equity and a day-trade count alone --
    this function does not attempt to reimplement it and never blocks a
    trade on that basis. It still surfaces a plain risk warning (not a
    compliance gate) when day-trading is frequent on a low-equity account,
    since that's genuinely risky independent of what FINRA currently
    requires."""
    allowed = True  # no equity-threshold trading restriction exists under the current rule
    warning = None
    if equity < low_equity_warning and day_trades_in_last_5_sessions >= 3:
        warning = (
            f"{day_trades_in_last_5_sessions} day trades in the last 5 sessions on "
            f"${equity:,.0f} equity -- frequent day trading on a small account carries "
            f"outsized risk regardless of margin rules. Not a compliance restriction "
            f"(FINRA's PDT rule and $25,000 threshold were eliminated June 4, 2026)."
        )
    return {"allowed": allowed, "reason": None, "warning": warning}


def check_drawdown_circuit_breaker(equity_log: pd.DataFrame, current_equity: float,
                                    max_daily_loss_pct: float = 0.03,
                                    max_weekly_loss_pct: float = 0.06) -> dict:
    """Halts new entries once the account has already lost more than the
    configured threshold today or this week. The per-trade stop-loss in
    evaluate_exit() only bounds a single position -- this is what stops
    several bad positions at once from compounding into an outsized loss."""
    if equity_log.empty:
        return {"halted": False, "reason": None}

    log = equity_log.copy()
    if log["timestamp"].dt.tz is None:
        log["timestamp"] = log["timestamp"].dt.tz_localize("UTC")
    now = pd.Timestamp.now(tz="UTC")

    today_rows = log[log["timestamp"].dt.date == now.date()]
    if not today_rows.empty:
        day_start_equity = float(today_rows["equity"].iloc[0])
        daily_loss_pct = (day_start_equity - current_equity) / day_start_equity if day_start_equity else 0.0
        if daily_loss_pct >= max_daily_loss_pct:
            return {
                "halted": True,
                "reason": (
                    f"Daily loss {daily_loss_pct:.1%} >= {max_daily_loss_pct:.1%} limit "
                    f"(${day_start_equity:,.2f} -> ${current_equity:,.2f})"
                ),
            }

    week_rows = log[log["timestamp"] >= now - pd.Timedelta(days=7)]
    if not week_rows.empty:
        week_start_equity = float(week_rows["equity"].iloc[0])
        weekly_loss_pct = (week_start_equity - current_equity) / week_start_equity if week_start_equity else 0.0
        if weekly_loss_pct >= max_weekly_loss_pct:
            return {
                "halted": True,
                "reason": (
                    f"Weekly loss {weekly_loss_pct:.1%} >= {max_weekly_loss_pct:.1%} limit "
                    f"(${week_start_equity:,.2f} -> ${current_equity:,.2f})"
                ),
            }

    return {"halted": False, "reason": None}


def check_sector_concentration(ticker_sectors: dict, weights: dict, max_sector_weight: float = 0.4) -> dict:
    """Flags when target weights concentrate too much in one sector.
    Momentum can pick several correlated names riding the same theme, which
    quietly turns 'diversified top-N picks' into one concentrated bet."""
    sector_totals: dict = {}
    for ticker, weight in weights.items():
        sector = ticker_sectors.get(ticker, "Unknown")
        sector_totals[sector] = sector_totals.get(sector, 0.0) + weight

    breaches = {s: round(w, 4) for s, w in sector_totals.items() if w > max_sector_weight}
    return {
        "sector_weights": {s: round(w, 4) for s, w in sector_totals.items()},
        "breaches": breaches,
        "within_limits": len(breaches) == 0,
    }


def evaluate_exit(entry_price, current_price, high_since_entry=None,
                   stop_loss_pct=0.08, take_profit_pct=0.20, trailing_stop_pct=0.10):
    high_since_entry = high_since_entry if high_since_entry is not None else max(entry_price, current_price)
    pnl_pct = (current_price - entry_price) / entry_price
    drawdown_from_high = (high_since_entry - current_price) / high_since_entry

    if pnl_pct <= -stop_loss_pct:
        return {"action": "SELL", "reason": "stop_loss", "pnl_pct": round(pnl_pct, 4)}
    if pnl_pct >= take_profit_pct:
        return {"action": "SELL", "reason": "take_profit", "pnl_pct": round(pnl_pct, 4)}
    if high_since_entry > entry_price and drawdown_from_high >= trailing_stop_pct:
        return {"action": "SELL", "reason": "trailing_stop", "pnl_pct": round(pnl_pct, 4)}
    return {"action": "HOLD", "reason": None, "pnl_pct": round(pnl_pct, 4)}
