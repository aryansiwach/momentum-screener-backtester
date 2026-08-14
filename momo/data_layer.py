"""Price data loaders: a synthetic GBM generator for demos, and a Yahoo
Finance loader that normalizes yfinance's inconsistent column layouts into
a clean tickers-as-columns DataFrame of adjusted close prices."""

import pandas as pd
import numpy as np

# -------- Synthetic loader (unchanged) --------
class DataLoader:
    def __init__(self, tickers, start, end, freq="B", seed=7):
        self.tickers = list(dict.fromkeys(tickers))
        self.start   = pd.Timestamp(start)
        self.end     = pd.Timestamp(end)
        self.freq    = freq
        self.seed    = seed

    def load_adj_close(self) -> pd.DataFrame:
        idx = pd.date_range(self.start, self.end, freq=self.freq)
        n = len(idx)
        rng = np.random.default_rng(self.seed)
        drift_range = np.linspace(0.03, 0.25, num=len(self.tickers))
        vol = 0.20; dt = 1/252
        prices = {}
        for i, t in enumerate(self.tickers):
            mu = drift_range[i]
            shocks = rng.normal((mu - 0.5*vol**2)*dt, vol*np.sqrt(dt), size=n).cumsum()
            s0 = 100*(1 + 0.1*rng.random())
            prices[t] = s0*np.exp(shocks)
        return pd.DataFrame(prices, index=idx)

# -------- Yahoo Finance loader (robust to column layouts) --------
class YFDataLoader:
    def __init__(self, tickers, start, end, freq="1d", pad_lookback=260, min_non_na=200, auto_ffill=True):
        self.tickers = list(dict.fromkeys(tickers))
        self.start   = pd.Timestamp(start)
        self.end     = pd.Timestamp(end)
        self.freq    = freq
        self.pad_lb  = int(pad_lookback)
        self.min_na  = int(min_non_na)
        self.auto_ff = auto_ffill

    def _extract_adj_close(self, df: pd.DataFrame) -> pd.DataFrame:
        # Handles both MultiIndex and single-index returns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            levels = [df.columns.get_level_values(i).unique().tolist()
                      for i in range(df.columns.nlevels)]
            # case A: level 0 holds fields ('Adj Close', 'Close', ...)
            if any(x in ("Adj Close", "Close") for x in levels[0]):
                field = "Adj Close" if "Adj Close" in levels[0] else "Close"
                out = df[field]
            # case B: level 1 holds fields
            elif len(levels) > 1 and any(x in ("Adj Close", "Close") for x in levels[1]):
                field = "Adj Close" if "Adj Close" in levels[1] else "Close"
                out = df.xs(field, axis=1, level=1)
            else:
                raise ValueError(f"Could not find Adj Close/Close in columns: {df.columns[:5]}")
            # Ensure columns are tickers (strings)
            out.columns = [str(c) for c in out.columns]
            return out

        # Single-index: one ticker or yfinance returned flat columns
        cols = [c for c in df.columns if isinstance(c, str)]
        # Try exact names first
        if "Adj Close" in df.columns:
            out = df[["Adj Close"]].copy()
        elif "Close" in df.columns:
            out = df[["Close"]].copy()
        else:
            # Try suffix pattern like 'AAPL Adj Close'
            adj_like = [c for c in cols if c.endswith("Adj Close")]
            close_like = [c for c in cols if c.endswith("Close")]
            chosen = adj_like or close_like
            if not chosen:
                raise ValueError(f"No Adj Close/Close-like columns in {cols[:5]}")
            out = df[chosen].copy()
            # rename 'AAPL Adj Close' -> 'AAPL'
            out.columns = [c.replace(" Adj Close","").replace(" Close","") for c in out.columns]

        # If it was truly single ticker, set column name to the ticker we asked for
        if out.shape[1] == 1 and len(self.tickers) == 1:
            out.columns = [self.tickers[0]]
        return out

    def load_adj_close(self) -> pd.DataFrame:
        import yfinance as yf
        padded_start = self.start - pd.tseries.offsets.BDay(self.pad_lb)

        df = yf.download(
            tickers=" ".join(self.tickers),
            start=padded_start.strftime("%Y-%m-%d"),
            end=(self.end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval=self.freq,
            auto_adjust=False,
            progress=False,
        )

        prices = self._extract_adj_close(df)
        prices.index = pd.to_datetime(prices.index, utc=False)
        prices = prices.sort_index()

        # reindex to business days for consistency
        bidx = pd.date_range(prices.index.min(), prices.index.max(), freq="B")
        prices = prices.reindex(bidx)
        if self.auto_ff:
            prices = prices.ffill()

        # trim to requested window
        prices = prices.loc[(prices.index >= self.start) & (prices.index <= self.end)]

        # drop tickers with too little data
        keep = [c for c in prices.columns if prices[c].notna().sum() >= self.min_na]
        prices = prices[keep]
        if prices.shape[1] == 0:
            raise ValueError("No tickers have sufficient data after cleaning.")
        return prices
