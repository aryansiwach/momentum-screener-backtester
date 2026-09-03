"""Market-regime filter: a well-established technique for reducing
momentum's worst-regime risk -- trade momentum at full size only when the
broad market itself is in an uptrend (price above its 200-day average),
scale down or sit in cash otherwise. Not a timing model and it won't dodge
every drawdown; it reduces exposure to the regime momentum is
historically worst in, on average, not on every single day."""

import pandas as pd


def compute_trend_state(benchmark_prices: pd.Series, sma_window: int = 200) -> pd.Series:
    sma = benchmark_prices.rolling(sma_window).mean()
    return benchmark_prices > sma  # True = risk-on (price above trend)


def current_regime(benchmark_prices: pd.Series, sma_window: int = 200) -> dict:
    sma = benchmark_prices.rolling(sma_window).mean()
    state = benchmark_prices > sma
    if state.empty or pd.isna(state.iloc[-1]) or pd.isna(sma.iloc[-1]):
        return {"regime": "unknown", "risk_on": None}

    latest_price = float(benchmark_prices.iloc[-1])
    latest_sma = float(sma.iloc[-1])
    risk_on = bool(state.iloc[-1])
    return {
        "regime": "risk_on" if risk_on else "risk_off",
        "risk_on": risk_on,
        "benchmark_price": round(latest_price, 2),
        "benchmark_sma": round(latest_sma, 2),
        "pct_above_sma": round((latest_price / latest_sma - 1) * 100, 2),
    }


def apply_regime_scaling(weights: pd.DataFrame, trend_state: pd.Series, risk_off_scale: float = 0.0) -> pd.DataFrame:
    """Scales each day's target weights by risk_off_scale on days the
    trend filter says risk-off (0.0 = fully flat, 0.3 = run at 30% size).
    trend_state must be aligned to weights' index -- reindex/ffill the
    benchmark trend onto the strategy's calendar before calling this."""
    # Reindexing a bool Series introduces NaN (extended index), which
    # upcasts it to object dtype and makes ffill()/fillna() warn about a
    # future downcast -- convert to float first so it's numeric throughout.
    numeric_state = trend_state.astype(float)
    aligned = numeric_state.reindex(weights.index).ffill()
    scale = aligned.map({1.0: 1.0, 0.0: risk_off_scale})
    scale = scale.fillna(risk_off_scale)
    return weights.mul(scale, axis=0)
