"""News headlines + lightweight sentiment scoring, run over a shortlist
(the day's momentum picks), not the full market -- pulling and scoring news
for thousands of tickers daily isn't a free-tier operation, and it's the
shortlist that actually needs it before you size a position."""

import os
import re
from datetime import timedelta

import pandas as pd

# Keyword-based risk screen over recent headlines -- deliberately simple and
# auditable (every flag traces back to a literal word in a real headline)
# rather than a black-box "risk score". This catches known-bad-news events
# already public (fraud, delisting, halts, restatements, going-concern
# doubt); it does NOT predict news that hasn't happened yet -- no filter
# can. Grouped so a hit also says *why* a ticker got excluded.
_RISK_KEYWORDS = {
    "fraud/investigation": [
        "fraud", "sec investigation", "sec probe", "doj investigation",
        "securities investigation", "class action", "subpoena", "indicted",
    ],
    "financial distress": [
        "bankruptcy", "chapter 11", "chapter 7", "insolvency", "insolvent",
        "going concern", "default on debt", "debt default", "restructuring debt",
    ],
    "delisting/halt": [
        "delisting", "delisted", "trading halt", "halted trading",
        "nasdaq compliance", "nyse compliance", "reverse split",
    ],
    "accounting/governance": [
        "restatement", "restate financial", "resigns amid", "ceo resigns",
        "cfo resigns", "accounting irregularit", "internal control weakness",
    ],
    "regulatory/legal action": [
        "recall", "lawsuit", "sues", "sued", "ftc action", "cease and desist",
        "guidance withdrawn", "withdraws guidance",
    ],
}


def detect_risk_flags(headlines_by_ticker: dict) -> dict:
    """ticker -> sorted list of risk categories whose keywords appear in its
    recent headlines. Empty list means no flagged headline was found, not
    "confirmed safe" -- absence of bad news isn't the same as absence of
    risk, and headline coverage itself is incomplete for thin-news tickers."""
    flags = {}
    for ticker, headlines in headlines_by_ticker.items():
        text = " ".join(headlines).lower()
        hit_categories = [
            category
            for category, keywords in _RISK_KEYWORDS.items()
            if any(re.search(re.escape(kw), text) for kw in keywords)
        ]
        flags[ticker] = sorted(hit_categories)
    return flags


def fetch_recent_news(tickers, lookback_days=3, api_key=None, secret_key=None, limit=10):
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest

    api_key = api_key or os.environ["ALPACA_API_KEY"]
    secret_key = secret_key or os.environ["ALPACA_SECRET_KEY"]
    client = NewsClient(api_key, secret_key)

    end = pd.Timestamp.today()
    start = end - timedelta(days=lookback_days)
    # NewsRequest.symbols is a single comma-separated string, not a list --
    # passing a list raises a pydantic ValidationError.
    request = NewsRequest(symbols=",".join(tickers), start=start, end=end, limit=limit)
    news = client.get_news(request)

    headlines_by_ticker = {t: [] for t in tickers}
    for item in news.data.get("news", []):
        for symbol in item.symbols:
            if symbol in headlines_by_ticker:
                headlines_by_ticker[symbol].append(item.headline)
    return headlines_by_ticker


def score_sentiment(headlines_by_ticker: dict) -> dict:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    scores = {}
    for ticker, headlines in headlines_by_ticker.items():
        if not headlines:
            scores[ticker] = 0.0
            continue
        compounds = [analyzer.polarity_scores(h)["compound"] for h in headlines]
        scores[ticker] = round(sum(compounds) / len(compounds), 4)
    return scores
