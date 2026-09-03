"""GARCH(1,1) conditional volatility (Engle 1982's ARCH, generalized by
Bollerslev 1986) -- a volatility estimate that responds to recent
clustering (calm markets stay calm, turbulent markets stay turbulent)
instead of a flat trailing standard deviation. This forecasts the SIZE of
future moves, not their direction -- it is not a price or return forecast."""

import numpy as np
import pandas as pd


def fit_garch(returns: pd.Series, p: int = 1, q: int = 1):
    """Fits GARCH(p,q) on daily returns. arch's optimizer is numerically
    happier with returns scaled to roughly unit variance (percent, not
    decimal) -- rescale on the way in; forecast_volatility() rescales the
    output back to decimal."""
    from arch import arch_model

    clean_returns = returns.dropna() * 100
    model = arch_model(clean_returns, vol="Garch", p=p, q=q, dist="normal", mean="Constant")
    return model.fit(disp="off")


def forecast_volatility(returns: pd.Series, horizon_days: int = 1, p: int = 1, q: int = 1) -> dict:
    """Forecasted daily/annualized volatility at the given horizon, plus
    the fitted persistence (alpha+beta): near 1 means volatility shocks
    decay slowly (long memory, real clustering); near 0 means they fade
    fast and a flat historical std dev would have been almost as good."""
    r = returns.dropna()
    if len(r) < 50:
        return {"converged": False, "reason": "fewer than 50 observations -- GARCH needs real history to fit"}

    try:
        result = fit_garch(r, p=p, q=q)
    except Exception as exc:
        return {"converged": False, "reason": f"GARCH fit failed: {exc}"}

    forecast = result.forecast(horizon=horizon_days, reindex=False)
    variance_pct2 = float(forecast.variance.values[-1, -1])
    daily_vol_decimal = np.sqrt(variance_pct2) / 100

    alpha = float(result.params.get("alpha[1]", np.nan))
    beta = float(result.params.get("beta[1]", np.nan))
    persistence = alpha + beta if not (np.isnan(alpha) or np.isnan(beta)) else None

    return {
        "horizon_days": horizon_days,
        "forecast_daily_vol_pct": round(daily_vol_decimal * 100, 3),
        "forecast_annualized_vol_pct": round(daily_vol_decimal * np.sqrt(252) * 100, 2),
        "alpha": round(alpha, 4) if not np.isnan(alpha) else None,
        "beta": round(beta, 4) if not np.isnan(beta) else None,
        "persistence": round(persistence, 4) if persistence is not None else None,
        "converged": True,
    }
