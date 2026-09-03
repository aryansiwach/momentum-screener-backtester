import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momo.technicals import classify_trend, build_narrative, build_bull_bear_case


def _series(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx)


def test_steady_uptrend_classifies_bullish_and_above_sma():
    prices = _series(100 * (1.01 ** np.arange(80)))
    trend = classify_trend(prices)
    assert trend["price_vs_sma50_pct"] > 0
    assert trend["rsi"] > 50
    assert trend["macd_histogram"] >= 0 or trend["macd_state"].startswith("bullish")


def test_steady_downtrend_classifies_bearish_and_below_sma():
    prices = _series(100 * (0.99 ** np.arange(80)))
    trend = classify_trend(prices)
    assert trend["price_vs_sma50_pct"] < 0
    assert trend["rsi"] < 50


def test_custom_windows_apply_regardless_of_bar_frequency():
    # Same math, deliberately fed a short intraday-scale window (e.g. 20
    # minute bars instead of 50 daily bars) -- classify_trend doesn't care
    # what the index frequency represents, only how many windows fit.
    prices = _series(100 * (1.02 ** np.arange(40)))
    trend = classify_trend(prices, rsi_window=14, sma_window=20, stoch_window=14)
    assert trend["price_vs_sma50_pct"] is not None
    assert trend["price_vs_sma50_pct"] > 0


def test_declining_intraday_session_reads_bearish_even_with_a_strong_daily_score():
    # The exact scenario this endpoint exists for: a ticker sliding all
    # session should classify as bearish/below-sma on intraday windows,
    # independent of whatever its 63-day daily score says.
    session = _series(100 * (0.998 ** np.arange(180)))  # ~180 one-minute bars, drifting down
    trend = classify_trend(session, rsi_window=14, sma_window=20, stoch_window=14)
    assert trend["price_vs_sma50_pct"] < 0
    assert trend["rsi"] < 50


def test_extreme_uptrend_hits_overbought_zone():
    prices = _series(100 * (1.08 ** np.arange(30)))
    trend = classify_trend(prices)
    assert trend["rsi_zone"] == "overbought"
    assert trend["stochastic_zone"] == "overbought"


def test_nan_gap_in_price_history_does_not_crash_and_reports_stochastic_unavailable():
    # Reproduces the real BIAF bug: a single missing/NaN bar (a halted
    # session, a reverse-split artifact from the data vendor) inside the
    # stochastic lookback window poisons pandas' rolling().min()/.max() for
    # the rest of that window, since -- unlike the EWM-based RSI/MACD --
    # rolling requires every point in the window to be non-NaN by default.
    # An unguarded NaN here used to reach the JSON response as a raw float
    # and crash serialization for the whole /ticker/analysis endpoint.
    values = list(100 * (1.01 ** np.arange(60)))
    values[-5] = float("nan")  # a gap inside the last stoch_window=14 bars
    prices = _series(values)
    trend = classify_trend(prices)
    assert trend["stochastic_k"] is None
    assert trend["stochastic_zone"] is None
    # RSI/MACD, being EWM-based, should still recover a real number despite
    # the same gap -- confirms this is specifically a rolling-window issue.
    assert trend["rsi"] == trend["rsi"]  # not NaN
    import json
    json.dumps(trend)  # must not raise "Out of range float values are not JSON compliant"


def test_narrative_handles_unavailable_stochastic_without_crashing():
    values = list(100 * (1.01 ** np.arange(60)))
    values[-5] = float("nan")
    trend = classify_trend(_series(values))
    narrative = build_narrative(trend, [], {"gap_risk": False, "illiquid": False, "high_short_interest": False, "news_flags": []})
    assert "unavailable" in narrative.lower()


def test_narrative_mentions_no_flags_when_clean():
    trend = classify_trend(_series(100 * (1.01 ** np.arange(80))))
    narrative = build_narrative(trend, patterns=[], risk_flags={})
    assert "No gap, liquidity, short-interest, or headline risk flags detected." in narrative
    assert "No notable candlestick pattern" in narrative


def test_narrative_surfaces_risk_flags():
    trend = classify_trend(_series(100 * (1.01 ** np.arange(80))))
    flags = {"gap_risk": True, "illiquid": True, "news_flags": ["fraud/investigation"]}
    narrative = build_narrative(trend, patterns=[], risk_flags=flags)
    assert "single-day gap" in narrative
    assert "thin trading liquidity" in narrative
    assert "fraud/investigation" in narrative


def test_narrative_surfaces_high_short_interest_flag():
    trend = classify_trend(_series(100 * (1.01 ** np.arange(80))))
    narrative = build_narrative(trend, patterns=[], risk_flags={"high_short_interest": True})
    assert "heavy short interest" in narrative


def test_narrative_reports_short_interest_percentage():
    trend = classify_trend(_series(100 * (1.01 ** np.arange(80))))
    ownership = {"short_interest_pct": 0.223}
    narrative = build_narrative(trend, patterns=[], risk_flags={}, ownership=ownership)
    assert "22.3% of float" in narrative


def test_narrative_reports_insider_activity_without_implying_a_signal():
    trend = classify_trend(_series(100 * (1.01 ** np.arange(80))))
    ownership = {"insider_summary": {"buys": 1, "sells": 3, "lookback_days": 90}}
    narrative = build_narrative(trend, patterns=[], risk_flags={}, ownership=ownership)
    assert "1 buy transaction(s), 3 sell transaction(s)" in narrative
    assert "not a buy/sell signal" in narrative


def test_narrative_omits_insider_line_when_no_transactions():
    trend = classify_trend(_series(100 * (1.01 ** np.arange(80))))
    ownership = {"insider_summary": {"buys": 0, "sells": 0, "lookback_days": 90}}
    narrative = build_narrative(trend, patterns=[], risk_flags={}, ownership=ownership)
    assert "Insider activity" not in narrative


def test_narrative_surfaces_top_pattern():
    trend = classify_trend(_series(100 * (1.01 ** np.arange(80))))
    patterns = [{"date": pd.Timestamp("2024-03-01"), "pattern": "bullish_engulfing", "bias": "bullish"}]
    narrative = build_narrative(trend, patterns=patterns, risk_flags={})
    assert "bullish engulfing" in narrative


def test_bull_bear_case_steady_uptrend_favors_bulls():
    # Mild enough that price stays under the "extended" threshold, so the
    # plain "established uptrend" bull branch fires instead of the bearish
    # "extended, reversion risk" branch.
    trend = classify_trend(_series(100 * (1.002 ** np.arange(80))))
    case = build_bull_bear_case(trend, patterns=[], risk_flags={})
    assert any("MACD" in p for p in case["bull_points"])
    assert any("uptrend" in p for p in case["bull_points"])


def test_bull_bear_case_downtrend_favors_bears():
    # A pure monotonic geometric decline doesn't reliably produce a
    # negative MACD histogram (an artifact of how fast/slow EMAs settle on
    # a smooth decay) -- a rise followed by a sharp recent drop does, and
    # is the more realistic bearish-crossover shape anyway.
    vals = np.concatenate([100 * (1.01 ** np.arange(60)), 100 * (1.01 ** 59) * (0.98 ** np.arange(20))])
    trend = classify_trend(_series(vals))
    case = build_bull_bear_case(trend, patterns=[], risk_flags={})
    assert any("MACD" in p for p in case["bear_points"])


def test_bull_bear_case_overbought_is_a_bear_point_even_in_an_uptrend():
    trend = classify_trend(_series(100 * (1.03 ** np.arange(60))))
    case = build_bull_bear_case(trend, patterns=[], risk_flags={})
    assert any("overbought" in p for p in case["bear_points"])


def test_bull_bear_case_extended_price_is_a_bear_point_despite_uptrend():
    # Needs >= 50 days for a real 50-day SMA (a 30-day series leaves
    # price_vs_sma50_pct as None -- no signal, not a false negative).
    trend = classify_trend(_series(100 * (1.03 ** np.arange(60))))
    case = build_bull_bear_case(trend, patterns=[], risk_flags={})
    assert any("extended" in p for p in case["bear_points"])


def test_bull_bear_case_risk_flags_all_land_in_bear_points():
    trend = classify_trend(_series(100 * (1.002 ** np.arange(80))))
    flags = {"gap_risk": True, "illiquid": True, "high_short_interest": True, "news_flags": ["fraud/investigation"]}
    case = build_bull_bear_case(trend, patterns=[], risk_flags=flags)
    assert len(case["bear_points"]) >= 4
    joined = " ".join(case["bear_points"]).lower()
    assert "gap" in joined and "liquidity" in joined and "short interest" in joined and "fraud/investigation" in joined


def test_bull_bear_case_bullish_pattern_lands_in_bull_points():
    trend = classify_trend(_series(100 * (1.01 ** np.arange(80))))
    patterns = [{"date": pd.Timestamp("2024-03-01"), "pattern": "hammer", "bias": "bullish"}]
    case = build_bull_bear_case(trend, patterns=patterns, risk_flags={})
    assert any("hammer" in p for p in case["bull_points"])


def test_bull_bear_case_insider_buying_favors_bulls_selling_favors_bears():
    trend = classify_trend(_series(100 * (1.01 ** np.arange(80))))
    buy_heavy = {"insider_summary": {"buys": 5, "sells": 1, "lookback_days": 90}}
    sell_heavy = {"insider_summary": {"buys": 1, "sells": 5, "lookback_days": 90}}

    bull_case = build_bull_bear_case(trend, patterns=[], risk_flags={}, ownership=buy_heavy)
    bear_case = build_bull_bear_case(trend, patterns=[], risk_flags={}, ownership=sell_heavy)

    assert any("Insiders bought more" in p for p in bull_case["bull_points"])
    assert any("Insiders sold more" in p for p in bear_case["bear_points"])
