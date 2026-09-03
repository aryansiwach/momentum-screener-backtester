"""Ticker -> sector lookup, shared by the dashboard's display-only banner
(api.py) and the actual execution path (momo/intraday.py, momo/live.py) so
the sector-concentration check enforced before a real order goes out uses
the same data as what the dashboard shows."""


def fetch_sectors(tickers) -> dict:
    """Best-effort ticker -> sector via yfinance. A missing/failed lookup
    counts as "Unknown" rather than silently dropping the ticker -- an
    unknown sector should never let a concentration breach go undetected
    just because the lookup failed."""
    import yfinance as yf

    sectors = {}
    for t in tickers:
        try:
            sectors[t] = yf.Ticker(t).info.get("sector") or "Unknown"
        except Exception:
            sectors[t] = "Unknown"
    return sectors
