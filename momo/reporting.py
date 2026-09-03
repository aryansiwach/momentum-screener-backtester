"""Morning momentum report and equity/position progress tracking. Both are
plain functions over the existing pipeline/broker objects -- no new state
of their own beyond the CSV log file."""

from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from momo.orchestrator import MomentumPipeline
from momo.data_layer import YFDataLoader, ChunkedLoader

# A relative "progress/equity_log.csv" resolves against whatever directory
# the *process* happens to be launched from -- different for a manually-run
# script (cwd = project root) versus the API server started via an absolute
# --app-dir (cwd = wherever the launcher's cwd was). That silently splits
# the log into two different files depending on who's asking. Anchor to the
# project root (one level up from this file's package) instead.
_DEFAULT_LOG_PATH = str(Path(__file__).resolve().parent.parent / "progress" / "equity_log.csv")


def morning_momentum_report(tickers, lookback_days=400, top_n=5,
                             screener_kwargs=None, equity=None,
                             data_loader_cls=YFDataLoader, exclude_gap_movers=True,
                             gap_lookback_days=63, max_daily_return=0.25):
    # 400 calendar days clears YFDataLoader's 200-trading-day min_non_na
    # floor with room for holidays/weekends -- below that every ticker gets
    # silently dropped as "insufficient data".
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=lookback_days)
    pipeline = MomentumPipeline(
        tickers, start, end, top_n=top_n,
        screener_kwargs=screener_kwargs or {},
        data_loader_cls=data_loader_cls,
    )
    results = pipeline.run()
    prices = results["prices"]
    latest_scores = results["scores"].iloc[-1].copy()

    if exclude_gap_movers:
        from momo.quality import filter_gap_movers
        smooth_mask = filter_gap_movers(prices, lookback_days=gap_lookback_days, max_daily_return=max_daily_return)
        latest_scores = latest_scores.where(
            smooth_mask.reindex(latest_scores.index, fill_value=False), other=float("-inf")
        )

    ranked = latest_scores.sort_values(ascending=False)
    ranked = ranked[ranked > float("-inf")]
    picks_tickers = ranked.head(top_n).index
    weight = 1.0 / len(picks_tickers) if len(picks_tickers) > 0 else 0.0

    rows = []
    for ticker in picks_tickers:
        row = {
            "ticker": ticker,
            "momentum_score": round(float(latest_scores[ticker]), 4),
            "target_weight": round(weight, 4),
        }
        if equity is not None:
            row["suggested_dollars"] = round(weight * equity, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def full_market_scan(lookback_days=400, top_n=10, min_price=5.0, screener_kwargs=None,
                      inner_loader_cls=YFDataLoader, universe=None, equity=None,
                      with_news=False, exclude_gap_movers=True, gap_lookback_days=63,
                      max_daily_return=0.25, exclude_news_risk=False, news_candidate_pool=None,
                      exclude_illiquid=False, liquidity_lookback_days=20, min_dollar_volume=5_000_000):
    """Funnel scan: cheap broad pass across the whole tradable universe,
    narrowed to the top momentum names -- news/sentiment (expensive per
    ticker) only runs on that shortlist, not the full market.

    exclude_gap_movers drops names whose recent momentum is really a single
    halt/reverse-split/news-gap day rather than a sustained trend (see
    momo.quality.filter_gap_movers) before ranking -- otherwise a stock up
    2000% in one session outranks everything with genuine multi-week
    momentum, every time.

    exclude_news_risk and exclude_illiquid both screen a wider candidate
    pool (news_candidate_pool, default top_n*3) after ranking and drop any
    ticker with a flagged headline (momo.news.detect_risk_flags) or below
    min_dollar_volume average daily liquidity (momo.quality.filter_illiquid)
    -- thin names are the ones most prone to violent single-day gaps and to
    slippage on the suggested allocation. Remaining candidates backfill in
    rank order so top_n picks are still returned where the pool allows.
    exclude_illiquid needs volume data, only available via AlpacaDataLoader
    -- it's a no-op without Alpaca configured. Both filters reduce exposure
    to already-visible risk; neither predicts risk that hasn't shown up yet,
    and neither guarantees a pick can't still lose money."""
    if universe is None:
        from momo.universe import fetch_tradable_universe
        universe = fetch_tradable_universe()

    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=lookback_days)

    pipeline = MomentumPipeline(
        universe, start, end, top_n=top_n,
        screener_kwargs=screener_kwargs or {},
        data_loader_cls=ChunkedLoader,
        data_loader_kwargs={"inner_loader_cls": inner_loader_cls},
    )
    results = pipeline.run()
    prices = results["prices"]
    latest_price = prices.iloc[-1]
    scores = results["scores"].iloc[-1].copy()

    if exclude_gap_movers:
        from momo.quality import filter_gap_movers
        smooth_mask = filter_gap_movers(prices, lookback_days=gap_lookback_days, max_daily_return=max_daily_return)
        scores = scores.where(smooth_mask.reindex(scores.index, fill_value=False), other=float("-inf"))

    ranked = scores[latest_price.reindex(scores.index) >= min_price]
    ranked = ranked.sort_values(ascending=False)
    ranked = ranked[ranked > float("-inf")]

    if (exclude_news_risk or exclude_illiquid) and len(ranked) > 0:
        pool_size = news_candidate_pool or top_n * 3
        candidates = ranked.head(pool_size).index.tolist()
        disqualified = set()

        if exclude_news_risk:
            from momo.news import fetch_recent_news, detect_risk_flags
            headlines = fetch_recent_news(candidates)
            news_flags_by_ticker = detect_risk_flags(headlines)
            disqualified |= {t for t in candidates if news_flags_by_ticker.get(t)}

        if exclude_illiquid:
            from momo.data_layer import AlpacaDataLoader
            from momo.quality import filter_illiquid
            vol_start = end - pd.Timedelta(days=liquidity_lookback_days * 3)
            ohlcv = AlpacaDataLoader(candidates, vol_start, end, timeframe_unit="Day").load_ohlcv()
            liquid_by_ticker = filter_illiquid(
                ohlcv, lookback_days=liquidity_lookback_days, min_dollar_volume=min_dollar_volume
            )
            disqualified |= {t for t in candidates if not liquid_by_ticker.get(t, False)}

        clean = [t for t in candidates if t not in disqualified]
        picks_tickers = pd.Index(clean[:top_n])
    else:
        picks_tickers = ranked.head(top_n).index

    weight = 1.0 / len(picks_tickers) if len(picks_tickers) > 0 else 0.0

    rows = []
    for ticker in picks_tickers:
        row = {
            "ticker": ticker,
            "price": round(float(latest_price[ticker]), 2),
            "momentum_score": round(float(scores[ticker]), 4),
            "target_weight": round(weight, 4),
        }
        if equity is not None:
            row["suggested_dollars"] = round(weight * equity, 2)
        rows.append(row)
    report = pd.DataFrame(rows)

    if with_news and not report.empty:
        from momo.news import fetch_recent_news, score_sentiment
        headlines = fetch_recent_news(report["ticker"].tolist())
        sentiment = score_sentiment(headlines)
        report["news_sentiment"] = report["ticker"].map(sentiment)

    return report


def log_progress_snapshot(broker, path=_DEFAULT_LOG_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    equity = broker.get_equity()
    positions = broker.get_positions()
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "equity": equity,
        "positions": positions,
    }
    header = not Path(path).exists()
    pd.DataFrame([row]).to_csv(path, mode="a", header=header, index=False)
    return row


def load_progress_log(path=_DEFAULT_LOG_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=["timestamp", "equity", "positions"])
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp")


def performance_summary(path=_DEFAULT_LOG_PATH) -> dict:
    df = load_progress_log(path)
    if df.empty:
        return {"starting_equity": None, "latest_equity": None, "total_return_pct": None, "daily_returns": df}

    starting = float(df["equity"].iloc[0])
    latest = float(df["equity"].iloc[-1])
    total_return_pct = round((latest - starting) / starting * 100, 2) if starting else None

    df = df.copy()
    df["daily_return_pct"] = df["equity"].pct_change().mul(100).round(2)

    return {
        "starting_equity": starting,
        "latest_equity": latest,
        "total_return_pct": total_return_pct,
        "daily_returns": df[["timestamp", "equity", "daily_return_pct"]],
    }
