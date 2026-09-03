import sys
import os
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momo.ownership import short_interest_pct, high_short_interest, summarize_insider_trading


def test_short_interest_pct_computes_fraction():
    assert short_interest_pct(shares_short=20_000_000, shares_float=100_000_000) == 0.2


def test_short_interest_pct_none_when_float_missing():
    assert short_interest_pct(shares_short=20_000_000, shares_float=None) is None
    assert short_interest_pct(shares_short=20_000_000, shares_float=0) is None


def test_high_short_interest_flags_at_threshold():
    assert high_short_interest(0.20, threshold=0.20) is True
    assert high_short_interest(0.19, threshold=0.20) is False
    assert high_short_interest(None, threshold=0.20) is False


def _recent(days_ago):
    d = datetime.date.today() - datetime.timedelta(days=days_ago)
    return d.strftime("%Y-%m-%d")


def test_summarize_insider_trading_counts_buys_and_sells():
    transactions = [
        {"transaction_date": _recent(5), "acquisition_or_disposition": "A", "securities_transacted": 100, "transaction_price": 10.0},
        {"transaction_date": _recent(10), "acquisition_or_disposition": "D", "securities_transacted": 50, "transaction_price": 20.0},
        {"transaction_date": _recent(10), "acquisition_or_disposition": "D", "securities_transacted": 25, "transaction_price": 20.0},
    ]
    summary = summarize_insider_trading(transactions, lookback_days=90)
    assert summary["buys"] == 1
    assert summary["sells"] == 2
    assert summary["buy_value"] == 1000.0
    assert summary["sell_value"] == 1500.0


def test_summarize_insider_trading_excludes_transactions_outside_lookback():
    transactions = [
        {"transaction_date": _recent(5), "acquisition_or_disposition": "D", "securities_transacted": 10, "transaction_price": 5.0},
        {"transaction_date": _recent(200), "acquisition_or_disposition": "D", "securities_transacted": 999, "transaction_price": 5.0},
    ]
    summary = summarize_insider_trading(transactions, lookback_days=90)
    assert summary["sells"] == 1
    assert summary["sell_value"] == 50.0


def test_summarize_insider_trading_accepts_full_word_spelling():
    # OpenBB's SEC provider spells this out as "Acquisition"/"Disposition",
    # not the raw SEC "A"/"D" code -- both must work.
    transactions = [
        {"transaction_date": _recent(5), "acquisition_or_disposition": "Disposition", "securities_transacted": 10, "transaction_price": 5.0},
        {"transaction_date": _recent(5), "acquisition_or_disposition": "Acquisition", "securities_transacted": 20, "transaction_price": 5.0},
    ]
    summary = summarize_insider_trading(transactions, lookback_days=90)
    assert summary["buys"] == 1
    assert summary["sells"] == 1


def test_summarize_insider_trading_handles_empty_list():
    summary = summarize_insider_trading([], lookback_days=90)
    assert summary == {"buys": 0, "sells": 0, "buy_value": 0.0, "sell_value": 0.0, "lookback_days": 90}
