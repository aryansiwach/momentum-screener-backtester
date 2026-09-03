"""API-layer tests for api.py, via FastAPI's TestClient. Previously this
file (~1000 lines) had zero automated coverage -- every endpoint was only
ever verified by hand, with curl or the browser, during development.

Deliberately does NOT use `with TestClient(app) as client:` -- that form
triggers the app's @app.on_event("startup") handler, which would start
the three real background threads (full-market scanner, exit monitor,
pre-market scan) and have them hit live Alpaca/Yahoo network calls during
a routine test run. Plain `TestClient(app)` calls never fire startup
handlers, so the background workers never start here regardless of what
keys happen to be in the environment -- keeping this file fast,
deterministic, and network-independent, consistent with the rest of the
test suite.

Coverage here focuses on what's actually testable without live network
credentials: the 503 Alpaca-key gate (every endpoint that's supposed to
enforce it, checked over real HTTP, not just as a unit-tested function),
the 400 invalid-range validation, the 503 ElevenLabs gate, and the
_fetch_guard context manager's error-wrapping behavior (the exact code
added in this session's simplification pass) via a mocked external call.
Endpoints whose success path requires a real Yahoo/Alpaca network call
are exercised instead through momo/*'s own unit tests (e.g.
test_data_layer.py), not duplicated here."""

import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

import api

client = TestClient(api.app)


@pytest.fixture(autouse=True)
def no_external_keys(monkeypatch):
    """Every test in this file runs as if no external service is
    configured, regardless of what's in the developer's real .env --
    guarantees the 503-gate tests are testing the gate, not today's
    actual credential state, and that no test can accidentally reach a
    real external API."""
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)


def test_health_check():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "service": "Momentum Trading API"}


def test_session_status_works_without_any_keys():
    # Pure computation from momo.session -- no external dependency at all.
    res = client.get("/session/status")
    assert res.status_code == 200
    body = res.json()
    assert set(body) == {"market_open", "minutes_to_close", "should_flatten", "can_open_new_position"}
    assert isinstance(body["market_open"], bool)


@pytest.mark.parametrize("path", [
    "/positions/exits",
    "/premarket/watch",
    "/momentum/full-market",
    "/momentum/full-market/cached",
    "/account",
    "/intraday/scan",
    "/ticker/intraday-analysis?ticker=AAPL",
])
def test_alpaca_gated_endpoints_503_without_keys(path):
    res = client.get(path)
    assert res.status_code == 503
    assert "ALPACA_API_KEY" in res.json()["detail"]


def test_momentum_history_intraday_range_503s_without_keys():
    # Only the minute-granularity branch (1D/1W) needs Alpaca -- daily
    # ranges use free Yahoo data and must not be gated (checked below).
    res = client.get("/momentum/history?ticker=AAPL&range=1D")
    assert res.status_code == 503
    assert "ALPACA_API_KEY" in res.json()["detail"]


def test_momentum_history_invalid_range_400s_before_touching_any_data_source():
    res = client.get("/momentum/history?ticker=AAPL&range=5Y")
    assert res.status_code == 400
    assert "Unknown range" in res.json()["detail"]
    assert "5Y" in res.json()["detail"]


def test_briefing_audio_503s_without_elevenlabs_key():
    res = client.get("/briefing/audio")
    assert res.status_code == 503
    assert "ELEVENLABS_API_KEY" in res.json()["detail"]


def test_full_market_history_works_without_any_keys():
    # Reads straight off disk (see _read_scan_log) -- deliberately not
    # gated behind Alpaca keys, since it's just replaying whatever the
    # last real scan already found.
    res = client.get("/momentum/full-market/history?limit=5")
    assert res.status_code == 200
    assert "scans" in res.json()


def test_fetch_guard_wraps_an_external_failure_as_502_not_a_500():
    # Exercises the exact _fetch_guard context manager added in this
    # session's simplification pass, over real HTTP: a data-source
    # exception must come back as a typed 502 with the same message this
    # endpoint always used, not an unhandled 500.
    with patch("yfinance.Tickers", side_effect=RuntimeError("simulated network failure")):
        res = client.get("/quotes?tickers=AAPL")
    assert res.status_code == 502
    assert "Quote fetch failed" in res.json()["detail"]
    assert "simulated network failure" in res.json()["detail"]


def test_fetch_guard_lets_http_exceptions_pass_through_unwrapped():
    # _fetch_guard's explicit `except HTTPException: raise` -- pairs_scan's
    # `with _fetch_guard(...)` block calls _fetch_pair_prices, which raises
    # its OWN HTTPException(404) when a ticker's data is missing. That 404
    # must reach the client as-is, not get rewrapped into a generic 502 --
    # this is the one case in api.py where an HTTPException is raised from
    # INSIDE a guarded block, so it's the only endpoint that actually
    # exercises this branch of _fetch_guard.
    import pandas as pd

    class _EmptyLoader:
        def __init__(self, *a, **k):
            pass

        def load_adj_close(self):
            return pd.DataFrame()  # neither requested ticker present -> _fetch_pair_prices's own 404

    with patch("api.YFDataLoader", _EmptyLoader):
        res = client.get("/pairs/scan?ticker_a=AAPL&ticker_b=MSFT")
    assert res.status_code == 404
    assert "No usable price history" in res.json()["detail"]
