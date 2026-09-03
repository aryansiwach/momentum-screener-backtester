"""Turns target portfolio weights into broker orders. compute_target_orders
is pure and network-free so it's directly testable; AlpacaBroker is a thin
wrapper that fetches account state and submits the resulting orders."""

import os
import math


class RiskGateBlocked(Exception):
    """Raised by AlpacaBroker.rebalance_to when a real (non-dry-run) order
    submission is refused because a risk gate is breached -- the circuit
    breaker or sector-concentration check existed and was tested
    (momo/risk.py) but nothing previously called them before submitting an
    order. This is that wiring: a breach here stops the submission, it
    doesn't just log a warning."""


def enforce_risk_gates(equity_log, equity: float, target_weights: dict, ticker_sectors: dict = None):
    """Pure, network-free risk gate (see AlpacaBroker.rebalance_to, which
    loads equity_log from disk and calls this before submitting real
    orders). Raises RiskGateBlocked if the drawdown circuit breaker or
    sector-concentration limit is breached; does nothing otherwise."""
    from momo.risk import check_drawdown_circuit_breaker, check_sector_concentration

    breaker = check_drawdown_circuit_breaker(equity_log, equity)
    if breaker["halted"]:
        raise RiskGateBlocked(f"Circuit breaker: {breaker['reason']}")

    if ticker_sectors:
        concentration = check_sector_concentration(ticker_sectors, target_weights)
        if not concentration["within_limits"]:
            breaches = ", ".join(f"{s} {w * 100:.0f}%" for s, w in concentration["breaches"].items())
            raise RiskGateBlocked(f"Sector concentration: {breaches} exceeds the limit")


def compute_target_orders(target_weights: dict, current_positions: dict, prices: dict,
                           equity: float, min_notional: float = 1.0, fractionable: bool = False) -> dict:
    orders = {}
    for ticker in set(target_weights) | set(current_positions):
        price = prices.get(ticker)
        if not price or price <= 0:
            continue
        raw_qty = (target_weights.get(ticker, 0.0) * equity) / price
        # Whole-share accounts truncate to an int; fractional-share accounts
        # (the default on a real Alpaca account) truncate to a fine but
        # finite precision instead of rounding down to zero on anything
        # under one full share -- which matters a lot on a small account
        # trying to hold an expensive stock.
        target_qty = math.trunc(raw_qty * 1_000_000) / 1_000_000 if fractionable else math.trunc(raw_qty)
        current_qty = current_positions.get(ticker, 0.0)
        delta = target_qty - current_qty
        if delta != 0 and abs(delta) * price >= min_notional:
            orders[ticker] = delta
    return orders


class AlpacaBroker:
    def __init__(self, api_key=None, secret_key=None, paper=True, fractionable=True):
        from alpaca.trading.client import TradingClient
        self.paper = paper
        self.fractionable = fractionable
        api_key = api_key or os.environ["ALPACA_API_KEY"]
        secret_key = secret_key or os.environ["ALPACA_SECRET_KEY"]
        self.client = TradingClient(api_key, secret_key, paper=paper)

    def get_equity(self) -> float:
        return float(self.client.get_account().equity)

    def get_positions(self) -> dict:
        return {p.symbol: float(p.qty) for p in self.client.get_all_positions()}

    def get_positions_detailed(self) -> dict:
        """ticker -> {qty, entry_price, current_price} -- richer than
        get_positions(), needed to actually evaluate a stop-loss/take-profit/
        trailing-stop exit (see momo.monitor.check_exits), not just know
        what's held."""
        return {
            p.symbol: {
                "qty": float(p.qty),
                "entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
            }
            for p in self.client.get_all_positions()
        }

    def check_exits(self, stop_loss_pct: float = 0.08, take_profit_pct: float = 0.20,
                     trailing_stop_pct: float = 0.10) -> list:
        """Evaluates every open position against its exit levels (see
        momo.monitor.check_exits) -- read-only, submits nothing."""
        from momo.monitor import check_exits as _check_exits
        positions = self.get_positions_detailed()
        return _check_exits(positions, stop_loss_pct, take_profit_pct, trailing_stop_pct)

    def monitor_and_exit(self, stop_loss_pct: float = 0.08, take_profit_pct: float = 0.20,
                          trailing_stop_pct: float = 0.10, dry_run: bool = True) -> list:
        """Same as check_exits, but actually submits a sell for anything
        that says SELL when dry_run=False. Defaults to dry_run=True on
        purpose -- this is the function that would auto-liquidate a real
        position with real money, so it never does that without the caller
        explicitly opting in."""
        positions = self.get_positions_detailed()
        from momo.monitor import check_exits as _check_exits
        verdicts = _check_exits(positions, stop_loss_pct, take_profit_pct, trailing_stop_pct)
        if not dry_run:
            for v in verdicts:
                if v["action"] == "SELL":
                    self._submit(v["ticker"], -positions[v["ticker"]]["qty"])
        return verdicts

    def rebalance_to(self, target_weights: dict, prices: dict, dry_run: bool = True,
                      check_risk: bool = True, ticker_sectors: dict = None) -> dict:
        equity = self.get_equity()
        current = self.get_positions()
        orders = compute_target_orders(target_weights, current, prices, equity, fractionable=self.fractionable)

        if not dry_run and check_risk:
            from momo.reporting import load_progress_log
            enforce_risk_gates(load_progress_log(), equity, target_weights, ticker_sectors)

        if not dry_run:
            for ticker, qty in orders.items():
                self._submit(ticker, qty)
        return orders

    def _submit(self, ticker: str, qty: float):
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        side = OrderSide.BUY if qty > 0 else OrderSide.SELL
        request = MarketOrderRequest(
            symbol=ticker, qty=abs(qty), side=side, time_in_force=TimeInForce.DAY
        )
        return self.client.submit_order(request)
