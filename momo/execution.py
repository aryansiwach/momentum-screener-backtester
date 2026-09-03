"""Turns target portfolio weights into broker orders. compute_target_orders
is pure and network-free so it's directly testable; AlpacaBroker is a thin
wrapper that fetches account state and submits the resulting orders."""

import os
import math
import json
import datetime
from pathlib import Path

# Append-only audit trail of every REAL (non-dry-run) rebalance_to call --
# what was computed, whether the risk gates passed or blocked it (and
# why), and the outcome of each order submission attempt. Same
# append-only-JSON-lines pattern as api.py's full-market scan log and
# momo.reporting's equity log. Not size-capped like that scan log: this
# only writes on real order-submission events (a handful a day at most
# for a personal account), not a continuous background loop, so unbounded
# growth isn't a practical concern the way it is there.
_ORDER_LOG_PATH = Path(__file__).resolve().parent.parent / "progress" / "order_log.jsonl"


def _append_order_log(entry: dict):
    try:
        _ORDER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_ORDER_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # audit logging is a nice-to-have -- a write failure must never block or mask a real risk decision


class RiskGateBlocked(Exception):
    """Raised by AlpacaBroker.rebalance_to when a real (non-dry-run) order
    submission is refused because a risk gate is breached -- the circuit
    breaker or sector-concentration check existed and was tested
    (momo/risk.py) but nothing previously called them before submitting an
    order. This is that wiring: a breach here stops the submission, it
    doesn't just log a warning."""


def enforce_risk_gates(equity_log, equity: float, target_weights: dict, ticker_sectors: dict = None,
                        max_position_weight: float = 0.4, max_gross_exposure: float = 1.05):
    """Pure, network-free risk gate (see AlpacaBroker.rebalance_to, which
    loads equity_log from disk and calls this before submitting real
    orders). Raises RiskGateBlocked if the drawdown circuit breaker,
    sector concentration, single-position concentration, or gross
    exposure sanity check is breached; does nothing otherwise. Checked in
    this order deliberately -- circuit breaker first, since a halted
    account shouldn't get a more specific reason for why its trade was
    ALSO oversized."""
    from momo.risk import (
        check_drawdown_circuit_breaker, check_sector_concentration,
        check_position_concentration, check_gross_exposure,
    )

    breaker = check_drawdown_circuit_breaker(equity_log, equity)
    if breaker["halted"]:
        raise RiskGateBlocked(f"Circuit breaker: {breaker['reason']}")

    if ticker_sectors:
        concentration = check_sector_concentration(ticker_sectors, target_weights)
        if not concentration["within_limits"]:
            breaches = ", ".join(f"{s} {w * 100:.0f}%" for s, w in concentration["breaches"].items())
            raise RiskGateBlocked(f"Sector concentration: {breaches} exceeds the limit")

    position_check = check_position_concentration(target_weights, max_position_weight)
    if not position_check["within_limits"]:
        breaches = ", ".join(f"{t} {w * 100:.0f}%" for t, w in position_check["breaches"].items())
        raise RiskGateBlocked(f"Position concentration: {breaches} exceeds the {max_position_weight * 100:.0f}% single-position limit")

    exposure_check = check_gross_exposure(target_weights, max_gross_exposure)
    if not exposure_check["within_limits"]:
        raise RiskGateBlocked(
            f"Gross exposure {exposure_check['gross_exposure'] * 100:.0f}% exceeds the "
            f"{max_gross_exposure * 100:.0f}% sanity limit -- target_weights likely has a construction bug"
        )


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

        if dry_run:
            return orders  # a preview -- nothing decided, nothing to audit

        log_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "target_weights": target_weights,
            "orders": orders,
            "risk_gate": "skipped (check_risk=False)",
            "submissions": [],
        }
        if check_risk:
            from momo.reporting import load_progress_log
            try:
                enforce_risk_gates(load_progress_log(), equity, target_weights, ticker_sectors)
                log_entry["risk_gate"] = "passed"
            except RiskGateBlocked as exc:
                log_entry["risk_gate"] = f"blocked: {exc}"
                _append_order_log(log_entry)
                raise

        # Each order is submitted independently -- one ticker's rejection
        # (e.g. insufficient buying power, an untradeable symbol) must not
        # silently prevent the other, unrelated orders in this rebalance
        # from going in. But a swallowed failure is a real-money footgun,
        # so every attempt is still logged and any failures are raised
        # together at the end, after every order has had its turn.
        failures = []
        for ticker, qty in orders.items():
            try:
                self._submit(ticker, qty)
                log_entry["submissions"].append({"ticker": ticker, "qty": qty, "status": "submitted"})
            except Exception as exc:
                log_entry["submissions"].append({"ticker": ticker, "qty": qty, "status": "error", "detail": str(exc)})
                failures.append(f"{ticker}: {exc}")
        _append_order_log(log_entry)
        if failures:
            raise RuntimeError(f"{len(failures)} of {len(orders)} order(s) failed to submit: {'; '.join(failures)}")
        return orders

    def _submit(self, ticker: str, qty: float):
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        side = OrderSide.BUY if qty > 0 else OrderSide.SELL
        request = MarketOrderRequest(
            symbol=ticker, qty=abs(qty), side=side, time_in_force=TimeInForce.DAY
        )
        return self.client.submit_order(request)
