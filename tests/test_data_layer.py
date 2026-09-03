import sys
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momo.data_layer import YFDataLoader
from momo.reporting import morning_momentum_report


def _fake_yf_download(tickers, start, end, interval, auto_adjust, progress):
    idx = pd.bdate_range(start, end)
    cols = pd.MultiIndex.from_product([["Adj Close"], tickers.split()])
    data = np.tile(np.linspace(100, 110, len(idx)).reshape(-1, 1), (1, len(cols)))
    return pd.DataFrame(data, index=idx, columns=cols)


def test_short_window_below_min_non_na_raises_clear_error():
    # A window shorter than min_non_na trading days must not silently return
    # an empty/partial frame -- this is the exact bug that made every ticker
    # vanish from the live report with a 180-day default lookback.
    loader = YFDataLoader(["AAPL"], "2024-06-01", "2024-08-01", min_non_na=200)
    with patch("yfinance.download", side_effect=_fake_yf_download):
        with pytest.raises(ValueError, match="sufficient data"):
            loader.load_adj_close()


def test_window_covering_min_non_na_succeeds():
    loader = YFDataLoader(["AAPL"], "2024-01-01", "2024-12-31", min_non_na=200)
    with patch("yfinance.download", side_effect=_fake_yf_download):
        prices = loader.load_adj_close()
    assert "AAPL" in prices.columns
    assert prices["AAPL"].notna().sum() >= 200


def test_reporting_default_lookback_clears_the_min_non_na_floor():
    # Regression guard: reporting.py's default lookback must stay comfortably
    # above YFDataLoader's default min_non_na (200 trading days), or every
    # ticker gets filtered out by default with no obvious cause.
    with patch("yfinance.download", side_effect=_fake_yf_download):
        report = morning_momentum_report(["AAPL", "MSFT"], top_n=2)
    assert not report.empty
