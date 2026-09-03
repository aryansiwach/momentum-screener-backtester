"""Fetches the tradable US equity universe from Alpaca -- every active,
exchange-listed common stock you can actually trade, not a hardcoded
watchlist."""

import os
import re

# A positive "name must contain 'Common Stock'" filter is fragile: real,
# large, liquid companies (Mastercard's Alpaca name is literally "Mastercard
# Incorporated", Visa's is "VISA Inc." -- neither says "Common Stock") were
# being silently excluded from the universe by that filter. A negative
# filter on the instrument types that actually aren't a single company's
# common equity is more robust.
_JUNK_PATTERN = re.compile(
    r"\b(Warrant|Warrants|Right|Rights|Unit|Units|Preferred|Depositary|ETFs?|ETNs?|"
    r"Ordinary Shares?|Trust Unit|Notes?|Exchange.Traded Fund|Bond Fund|"
    r"Allocation Fund|Income Fund|Dividend Fund|Growth Fund|Municipal Fund)\b",
    re.IGNORECASE,
)


def _fetch_assets(api_key, secret_key, exchanges, common_stock_only):
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass, AssetStatus

    api_key = api_key or os.environ["ALPACA_API_KEY"]
    secret_key = secret_key or os.environ["ALPACA_SECRET_KEY"]
    client = TradingClient(api_key, secret_key, paper=True)

    request = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
    assets = client.get_all_assets(request)

    return [
        a for a in assets
        if a.tradable and a.exchange in exchanges and "/" not in a.symbol
        and (not common_stock_only or not (a.name and _JUNK_PATTERN.search(a.name)))
    ]


def fetch_tradable_universe(api_key=None, secret_key=None, exchanges=("NYSE", "NASDAQ", "ARCA", "BATS"),
                             common_stock_only=True):
    # Alpaca's asset name reliably distinguishes real common stock from
    # SPAC warrants/units/rights and preferred shares ("Apple Inc. Common
    # Stock" vs "Armada Acquisition Corp. III Warrant") -- ticker-pattern
    # heuristics can't do this reliably, the name field can.
    assets = _fetch_assets(api_key, secret_key, exchanges, common_stock_only)
    return sorted({a.symbol for a in assets})


def fetch_universe_issuer_names(api_key=None, secret_key=None, exchanges=("NYSE", "NASDAQ", "ARCA", "BATS"),
                                 common_stock_only=True) -> dict:
    """ticker -> normalized issuer name, stripping share-class suffixes
    ("Class A Common Stock", "Class B Common Stock", ...) so two tickers
    from the *same* company (e.g. dual-class tracking stocks) can be
    detected and excluded from pairs screening -- they trivially correlate
    because they're the same economic interest, not an independent bet."""
    import re

    assets = _fetch_assets(api_key, secret_key, exchanges, common_stock_only)
    names = {}
    for a in assets:
        if not a.name:
            continue
        normalized = re.sub(
            r"\s*(Class|Series)\s+[A-Z]\b.*$", "", a.name, flags=re.IGNORECASE
        ).strip()
        normalized = re.sub(r"\s*Common Stock\s*$", "", normalized, flags=re.IGNORECASE).strip()
        names[a.symbol] = normalized
    return names
