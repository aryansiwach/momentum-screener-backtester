import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momo.premarket import filter_clean_movers, rank_intraday_leaders


def test_warrant_style_ticker_excluded_not_in_clean_universe():
    movers = [{"symbol": "GFAIW", "price": 0.0144, "percent_change": 476.0}]
    result = filter_clean_movers(movers, clean_tickers={"AAPL", "MSFT"}, min_price=3.0)
    assert result == []


def test_clean_ticker_below_min_price_excluded():
    movers = [{"symbol": "AAPL", "price": 1.50, "percent_change": 20.0}]
    result = filter_clean_movers(movers, clean_tickers={"AAPL"}, min_price=3.0)
    assert result == []


def test_clean_liquid_mover_passes():
    movers = [{"symbol": "AAPL", "price": 230.0, "percent_change": 5.2}]
    result = filter_clean_movers(movers, clean_tickers={"AAPL"}, min_price=3.0)
    assert result == movers


def test_mix_of_clean_and_junk_keeps_only_clean():
    movers = [
        {"symbol": "AAPL", "price": 230.0, "percent_change": 5.2},
        {"symbol": "RCKTW", "price": 0.0001, "percent_change": -75.0},
        {"symbol": "PENNY", "price": 0.50, "percent_change": 300.0},
    ]
    result = filter_clean_movers(movers, clean_tickers={"AAPL", "PENNY"}, min_price=3.0)
    assert result == [{"symbol": "AAPL", "price": 230.0, "percent_change": 5.2}]


def test_empty_movers_list():
    assert filter_clean_movers([], clean_tickers={"AAPL"}, min_price=3.0) == []


def _trend(macd_state="bullish_accelerating", rsi_zone="neutral", session_range_pct=3.0, session_change_pct=5.0):
    return {"macd_state": macd_state, "rsi_zone": rsi_zone, "rsi": 55.0, "macd_histogram": 0.1,
            "last_price": 10.0, "price_vs_sma50_pct": 1.0, "stochastic_k": 50.0, "stochastic_zone": "neutral",
            "session_range_pct": session_range_pct, "session_change_pct": session_change_pct}


def test_leader_excluded_when_rsi_already_overbought():
    candidates = [{"symbol": "AAA", "percent_change": 12.0, "intraday_trend": _trend(rsi_zone="overbought")}]
    assert rank_intraday_leaders(candidates) == []


def test_leader_excluded_when_macd_bearish():
    candidates = [{"symbol": "AAA", "percent_change": 12.0, "intraday_trend": _trend(macd_state="bearish_accelerating")}]
    assert rank_intraday_leaders(candidates) == []


def test_leader_excluded_when_no_intraday_trend_available():
    candidates = [{"symbol": "AAA", "percent_change": 12.0, "intraday_trend": None}]
    assert rank_intraday_leaders(candidates) == []


def test_leader_excluded_when_not_actually_a_gainer():
    candidates = [{"symbol": "AAA", "percent_change": -2.0, "intraday_trend": _trend()}]
    assert rank_intraday_leaders(candidates) == []


def test_leader_excluded_when_up_vs_prior_close_but_falling_within_todays_session():
    # The real bug this was built to catch: up big vs. YESTERDAY's close
    # (gapped up at the open), MACD still reads "bullish_accelerating" off
    # the shape of the whole session, but the stock has been sliding all
    # session and sits below where it opened today. This must not qualify
    # as a "leader" no matter how positive percent_change or MACD look.
    candidates = [{
        "symbol": "AAA", "percent_change": 19.2,
        "intraday_trend": _trend(macd_state="bullish_accelerating", session_change_pct=-15.6),
    }]
    assert rank_intraday_leaders(candidates) == []


def test_leader_excluded_when_session_range_too_quiet_for_target():
    # Technically confirmed bullish, but today's whole session has only
    # spanned 0.8% high-to-low -- not enough demonstrated room to plausibly
    # clear a 2-3% target, whatever the RSI/MACD say.
    candidates = [{"symbol": "AAA", "percent_change": 5.0, "intraday_trend": _trend(session_range_pct=0.8)}]
    assert rank_intraday_leaders(candidates) == []


def test_leader_included_when_session_range_exactly_at_threshold():
    candidates = [{"symbol": "AAA", "percent_change": 5.0, "intraday_trend": _trend(session_range_pct=2.0)}]
    result = rank_intraday_leaders(candidates)
    assert len(result) == 1 and result[0]["symbol"] == "AAA"


def test_custom_min_session_range_pct_is_respected():
    candidates = [{"symbol": "AAA", "percent_change": 5.0, "intraday_trend": _trend(session_range_pct=2.5)}]
    assert rank_intraday_leaders(candidates, min_session_range_pct=3.0) == []
    assert len(rank_intraday_leaders(candidates, min_session_range_pct=2.0)) == 1


def test_qualifying_gainer_passes_through():
    candidates = [{"symbol": "AAA", "percent_change": 8.0, "intraday_trend": _trend()}]
    result = rank_intraday_leaders(candidates)
    assert len(result) == 1 and result[0]["symbol"] == "AAA"


def test_ranked_by_todays_session_gain_first_regardless_of_prior_close_percent():
    # BIG is up more vs. YESTERDAY's close, but SMALL is actually up more
    # within TODAY's own session -- SMALL should rank first, since "is it
    # going up right now" is what this list is for.
    big_stale_gain = {"symbol": "BIG", "percent_change": 40.0, "intraday_trend": _trend(session_change_pct=1.0)}
    small_live_gain = {"symbol": "SMALL", "percent_change": 5.0, "intraday_trend": _trend(session_change_pct=6.0)}
    result = rank_intraday_leaders([big_stale_gain, small_live_gain])
    assert [c["symbol"] for c in result] == ["SMALL", "BIG"]


def test_within_same_session_gain_accelerating_macd_ranks_above_decelerating():
    decel = {"symbol": "DECEL", "intraday_trend": _trend(macd_state="bullish_decelerating", session_change_pct=3.0)}
    accel = {"symbol": "ACCEL", "intraday_trend": _trend(macd_state="bullish_accelerating", session_change_pct=3.0)}
    decel["percent_change"] = accel["percent_change"] = 5.0
    result = rank_intraday_leaders([decel, accel])
    assert [c["symbol"] for c in result] == ["ACCEL", "DECEL"]


def test_within_same_session_gain_and_macd_ranked_by_session_range_descending():
    lower_range = {"symbol": "LOWRANGE", "percent_change": 5.0, "intraday_trend": _trend(session_range_pct=2.1, session_change_pct=3.0)}
    higher_range = {"symbol": "HIRANGE", "percent_change": 5.0, "intraday_trend": _trend(session_range_pct=5.0, session_change_pct=3.0)}
    result = rank_intraday_leaders([lower_range, higher_range])
    assert [c["symbol"] for c in result] == ["HIRANGE", "LOWRANGE"]
