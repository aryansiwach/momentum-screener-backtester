import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momo.news import detect_risk_flags


def test_clean_headlines_produce_no_flags():
    headlines = {"AAPL": ["Apple beats earnings estimates", "New iPhone launch draws crowds"]}
    flags = detect_risk_flags(headlines)
    assert flags["AAPL"] == []


def test_fraud_investigation_headline_is_flagged():
    headlines = {"XYZ": ["XYZ Corp shares plunge after SEC investigation announced"]}
    flags = detect_risk_flags(headlines)
    assert "fraud/investigation" in flags["XYZ"]


def test_bankruptcy_headline_is_flagged():
    headlines = {"XYZ": ["XYZ Corp files for Chapter 11 bankruptcy protection"]}
    flags = detect_risk_flags(headlines)
    assert "financial distress" in flags["XYZ"]


def test_delisting_and_halt_headline_is_flagged():
    headlines = {"XYZ": ["Trading halted in XYZ shares pending news"]}
    flags = detect_risk_flags(headlines)
    assert "delisting/halt" in flags["XYZ"]


def test_multiple_categories_can_flag_simultaneously():
    headlines = {
        "XYZ": [
            "XYZ CEO resigns amid accounting irregularities probe",
            "Trading halted in XYZ shares after restatement of financials",
        ]
    }
    flags = detect_risk_flags(headlines)
    assert set(flags["XYZ"]) >= {"accounting/governance", "delisting/halt"}


def test_empty_headlines_produce_no_flags():
    headlines = {"AAPL": []}
    flags = detect_risk_flags(headlines)
    assert flags["AAPL"] == []
