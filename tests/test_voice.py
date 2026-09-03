import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momo.voice import build_briefing_text, synthesize_speech


def test_empty_picks_produces_fallback_text():
    text = build_briefing_text([], source="alpaca_full_market")
    assert "No momentum picks" in text


def test_briefing_mentions_top_three_picks_in_order():
    picks = [
        {"ticker": "HAE", "momentum_score": 0.97, "suggested_dollars": 100.0},
        {"ticker": "TXG", "momentum_score": 0.96, "suggested_dollars": 100.0},
        {"ticker": "PAYC", "momentum_score": 0.96, "suggested_dollars": 100.0},
        {"ticker": "CRL", "momentum_score": 0.95, "suggested_dollars": 100.0},
    ]
    text = build_briefing_text(picks, source="alpaca_full_market")
    assert "HAE" in text
    assert "TXG" in text
    assert "PAYC" in text
    assert "CRL" not in text  # only top 3 spoken, not the full list
    assert text.index("HAE") < text.index("TXG") < text.index("PAYC")


def test_briefing_includes_disclaimer():
    picks = [{"ticker": "AAPL", "momentum_score": 0.9}]
    text = build_briefing_text(picks, source="yahoo_finance")
    assert "not investment advice" in text


def test_briefing_omits_dollar_line_when_no_equity_context():
    picks = [{"ticker": "AAPL", "momentum_score": 0.9}]
    text = build_briefing_text(picks, source="yahoo_finance")
    assert "dollars" not in text


def test_synthesize_speech_raises_clearly_without_api_key(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
        synthesize_speech("hello", api_key=None)
