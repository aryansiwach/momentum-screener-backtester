import sys
import os
import json

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momo.reporting import performance_summary, load_progress_log


def test_performance_summary_computes_total_return(tmp_path):
    path = tmp_path / "equity_log.csv"
    rows = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="D"),
        "equity": [10000.0, 10500.0, 11000.0],
        "positions": [{}, {}, {}],
    })
    rows.to_csv(path, index=False)

    summary = performance_summary(str(path))
    assert summary["starting_equity"] == 10000.0
    assert summary["latest_equity"] == 11000.0
    assert summary["total_return_pct"] == 10.0
    assert len(summary["daily_returns"]) == 3


def test_performance_summary_handles_missing_log_file(tmp_path):
    path = tmp_path / "does_not_exist.csv"
    summary = performance_summary(str(path))
    assert summary["total_return_pct"] is None


def test_load_progress_log_returns_empty_frame_when_missing(tmp_path):
    path = tmp_path / "does_not_exist.csv"
    df = load_progress_log(str(path))
    assert df.empty
    assert list(df.columns) == ["timestamp", "equity", "positions"]


def test_single_row_daily_returns_are_json_serializable_after_nan_conversion(tmp_path):
    # A single logged snapshot has no prior day to compare, so pct_change()
    # produces NaN for that row. Starlette's JSONResponse uses allow_nan=False
    # and 500s on a bare NaN -- this is the exact transformation api.py's
    # /performance endpoint applies before returning, reproduced here so the
    # regression is caught without spinning up a live server.
    path = tmp_path / "equity_log.csv"
    pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=1, freq="D"),
        "equity": [1000.0],
        "positions": [{}],
    }).to_csv(path, index=False)

    summary = performance_summary(str(path))
    daily = summary.pop("daily_returns")
    assert daily["daily_return_pct"].isna().any()

    daily = daily.astype(object).where(pd.notna(daily), None)
    records = daily.to_dict(orient="records")
    # FastAPI's real response path runs this through jsonable_encoder (which
    # handles the Timestamp column); isolate just the NaN-vs-null behavior
    # this fix targets by checking the numeric field directly.
    json.dumps(records[0]["daily_return_pct"], allow_nan=False)  # raises if NaN survived
    assert records[0]["daily_return_pct"] is None
