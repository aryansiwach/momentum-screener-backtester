"""Watches open positions against their stop-loss / take-profit / trailing-
stop levels and decides when to exit. momo.risk.evaluate_exit already has
this logic but was never called anywhere in the codebase -- the numbers
shown in the dashboard's position sizer were preview-only, and nothing
actually watched a real position once it was opened. This module is that
missing piece.

A trailing stop needs the running high price since entry, which Alpaca's
position object doesn't track across days by itself -- persisted to disk
(same pattern as reporting.py's equity_log.csv) so it survives restarts."""

import json
from pathlib import Path

from momo.risk import evaluate_exit

_HIGHS_PATH = Path(__file__).resolve().parent.parent / "progress" / "position_highs.json"


def _load_highs() -> dict:
    if not _HIGHS_PATH.exists():
        return {}
    try:
        return json.loads(_HIGHS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_highs(highs: dict):
    _HIGHS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _HIGHS_PATH.write_text(json.dumps(highs), encoding="utf-8")


def check_exits(positions: dict, stop_loss_pct: float = 0.08, take_profit_pct: float = 0.20,
                 trailing_stop_pct: float = 0.10, highs: dict = None, persist: bool = True) -> list:
    """positions: {ticker: {"entry_price": float, "current_price": float}}.
    Returns [{"ticker", "action", "reason", "pnl_pct"}, ...] for every
    position -- "action" is "SELL" or "HOLD" per evaluate_exit. Pass
    `highs` explicitly (and persist=False) for a pure, disk-free unit test;
    otherwise the running high-since-entry is loaded from and saved back to
    disk automatically so a trailing stop survives a process restart."""
    if highs is None:
        highs = _load_highs() if persist else {}
    else:
        highs = dict(highs)

    results = []
    still_open = set()
    for ticker, info in positions.items():
        still_open.add(ticker)
        entry = info["entry_price"]
        current = info["current_price"]
        high = max(highs.get(ticker, entry), current, entry)
        highs[ticker] = high
        verdict = evaluate_exit(
            entry, current, high_since_entry=high,
            stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct, trailing_stop_pct=trailing_stop_pct,
        )
        results.append({"ticker": ticker, **verdict})

    # Positions that closed since the last check no longer need a tracked
    # high -- otherwise a re-entered position would start from its old high
    # instead of a fresh entry.
    for ticker in list(highs.keys()):
        if ticker not in still_open:
            del highs[ticker]

    if persist:
        _save_highs(highs)
    return results
