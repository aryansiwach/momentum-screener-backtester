"""Volatility-targeted position scaling: instead of a binary regime
on/off switch (which whipsaws around its threshold -- see regime.py's
tested, honest finding that it hurts Sharpe more than it helps), scale
exposure continuously and inversely to forecasted volatility. Calm
markets run closer to full size; turbulent markets run smaller,
smoothly, without flipping state. Standard technique, not novel; the
point here is using the GARCH forecast already validated in
momo/volatility.py to drive it instead of a lagging trend line."""

import pandas as pd

from momo.volatility import forecast_volatility


def compute_vol_scale(returns: pd.Series, target_daily_vol_pct: float = 1.5,
                       min_scale: float = 0.25, max_scale: float = 1.5) -> dict:
    """scale = target_vol / forecast_vol, clipped to [min_scale, max_scale]
    so a very calm market doesn't lever up past a sane bound and a very
    turbulent one doesn't go to zero (that's the regime filter's job if
    you want a hard stop; this only resizes)."""
    garch = forecast_volatility(returns, horizon_days=1)
    if not garch.get("converged"):
        return {"scale": 1.0, "vol_model": "none", "reason": garch.get("reason")}

    forecast_vol = garch["forecast_daily_vol_pct"]
    if forecast_vol <= 0:
        return {"scale": 1.0, "vol_model": "garch(1,1)", "reason": "zero forecast volatility"}

    raw_scale = target_daily_vol_pct / forecast_vol
    scale = max(min_scale, min(max_scale, raw_scale))
    return {
        "scale": round(scale, 3),
        "vol_model": "garch(1,1)",
        "forecast_daily_vol_pct": forecast_vol,
        "target_daily_vol_pct": target_daily_vol_pct,
        "reason": None,
    }


def apply_vol_targeting(weights: pd.DataFrame, prices: pd.DataFrame, target_daily_vol_pct: float = 1.5,
                         min_scale: float = 0.25, max_scale: float = 1.5, lookback: int = 250,
                         refit_every: int = 5) -> pd.DataFrame:
    """Scales each day's portfolio weights by a GARCH-vol-targeted factor
    computed from the equal-weighted basket's own return history up to
    that day (no look-ahead: only data available at time t is used).
    Refits every `refit_every` days and holds the scale constant between
    refits -- both for tractability (refitting GARCH daily over a
    multi-year backtest is needlessly slow) and because it's how a real
    system would actually run this, not on every single tick."""
    basket_returns = prices.pct_change(fill_method=None).mean(axis=1).fillna(0)
    scales = pd.Series(1.0, index=weights.index)

    current_scale = 1.0
    for i in range(lookback, len(basket_returns)):
        if (i - lookback) % refit_every == 0:
            window = basket_returns.iloc[i - lookback:i]
            current_scale = compute_vol_scale(window, target_daily_vol_pct, min_scale, max_scale)["scale"]
        scales.iloc[i] = current_scale

    return weights.mul(scales, axis=0)
