import sys
import os

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momo.data_layer import ChunkedLoader


class _FakeLoader:
    """Returns one column of constant prices per ticker; raises for 'BAD'."""

    def __init__(self, tickers, start, end):
        self.tickers = tickers
        self.start = start
        self.end = end

    def load_adj_close(self):
        if "BAD" in self.tickers:
            raise ValueError("simulated delisted ticker")
        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        return pd.DataFrame({t: [100.0, 101.0, 102.0] for t in self.tickers}, index=idx)


def test_chunks_and_concatenates_all_tickers():
    tickers = [f"T{i}" for i in range(5)]
    loader = ChunkedLoader(tickers, "2024-01-01", "2024-01-05",
                            inner_loader_cls=_FakeLoader, chunk_size=2)
    prices = loader.load_adj_close()
    assert sorted(prices.columns) == sorted(tickers)
    assert len(prices) == 3


def test_skips_chunk_that_errors_without_failing_whole_scan():
    tickers = ["GOOD1", "GOOD2", "BAD", "GOOD3"]
    loader = ChunkedLoader(tickers, "2024-01-01", "2024-01-05",
                            inner_loader_cls=_FakeLoader, chunk_size=1)
    prices = loader.load_adj_close()
    assert "BAD" not in prices.columns
    assert set(prices.columns) == {"GOOD1", "GOOD2", "GOOD3"}


def test_raises_if_every_chunk_fails():
    loader = ChunkedLoader(["BAD"], "2024-01-01", "2024-01-05",
                            inner_loader_cls=_FakeLoader, chunk_size=1)
    with pytest.raises(ValueError):
        loader.load_adj_close()
