"""Price data loaders: a synthetic GBM generator for demos, a Yahoo Finance
loader that normalizes yfinance's inconsistent column layouts, and an
Alpaca loader for real(ish)-time bars usable in live trading. All expose
the same load_adj_close() -> tickers-as-columns DataFrame interface."""

import os
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


# -------- Alpaca loader (for live/paper trading) --------
class AlpacaDataLoader:
    def __init__(self, tickers, start, end, api_key=None, secret_key=None, feed="iex",
                 timeframe_amount=1, timeframe_unit="Day"):
        self.tickers    = list(dict.fromkeys(tickers))
        self.start      = pd.Timestamp(start)
        self.end        = pd.Timestamp(end)
        self.api_key    = api_key or os.environ["ALPACA_API_KEY"]
        self.secret_key = secret_key or os.environ["ALPACA_SECRET_KEY"]
        self.feed       = feed
        self.timeframe_amount = timeframe_amount
        self.timeframe_unit   = timeframe_unit

    def _bars(self):
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        client = StockHistoricalDataClient(self.api_key, self.secret_key)
        # TimeFrameUnit's *values* are irregular ("Min" for Minute, "Day" for
        # Day) -- look up by member name instead of constructing from a
        # string value, or "Minute" raises ValueError.
        timeframe = TimeFrame(self.timeframe_amount, getattr(TimeFrameUnit, self.timeframe_unit))
        req = StockBarsRequest(
            symbol_or_symbols=self.tickers,
            timeframe=timeframe,
            start=self.start,
            end=self.end,
            feed=self.feed,
        )
        bars = client.get_stock_bars(req).df
        if bars.empty:
            raise ValueError(f"Alpaca returned no bars for {self.tickers} between {self.start} and {self.end}")
        return bars

    def load_adj_close(self) -> pd.DataFrame:
        bars = self._bars()
        prices = bars["close"].unstack(level=0)
        prices.index = pd.to_datetime(prices.index)
        return prices.sort_index()

    def load_ohlcv(self) -> dict:
        """Per-ticker OHLCV DataFrames -- volume is needed for intraday
        signals (VWAP, volume-confirmed breakouts) that close-only data
        can't support."""
        bars = self._bars()
        out = {}
        for ticker in self.tickers:
            if ticker not in bars.index.get_level_values(0):
                continue
            df = bars.xs(ticker, level=0)[["open", "high", "low", "close", "volume"]].copy()
            df.index = pd.to_datetime(df.index)
            out[ticker] = df.sort_index()
        return out


# -------- WRDS loader (for deep historical backtesting, not live trading) --------
class WRDSDataLoader:
    """Adjusted-close prices from the WRDS CRSP daily stock file -- built for
    backtest depth/rigor, not live execution (CRSP data lags by days).
    Requires a WRDS account with CRSP access; connects via the official
    `wrds` package (prompts for a password unless ~/.pgpass or
    WRDS_USERNAME/WRDS_PASSWORD env vars are set)."""

    def __init__(self, tickers, start, end, wrds_username=None):
        self.tickers = list(dict.fromkeys(tickers))
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.wrds_username = wrds_username or os.environ.get("WRDS_USERNAME")

    def load_adj_close(self) -> pd.DataFrame:
        import wrds
        db = wrds.Connection(wrds_username=self.wrds_username)
        try:
            tickers_sql = ", ".join(f"'{t}'" for t in self.tickers)
            query = f"""
                select a.date, b.ticker, a.prc, a.cfacpr
                from crsp.dsf as a
                join crsp.dsenames as b
                  on a.permno = b.permno
                 and a.date between b.namedt and coalesce(b.nameendt, a.date)
                where b.ticker in ({tickers_sql})
                  and a.date between '{self.start.date()}' and '{self.end.date()}'
            """
            df = db.raw_sql(query, date_cols=["date"])
        finally:
            db.close()

        if df.empty:
            raise ValueError(
                f"WRDS returned no CRSP data for {self.tickers} between "
                f"{self.start.date()} and {self.end.date()}"
            )

        # CRSP prc is negative when it's a bid/ask midpoint rather than a
        # trade price; cfacpr is the cumulative split/dividend adjustment
        # factor -- dividing by it reproduces CRSP's own adjusted price.
        df["adj_close"] = df["prc"].abs() / df["cfacpr"]
        prices = df.pivot(index="date", columns="ticker", values="adj_close")
        return prices.sort_index()


# -------- Chunked wrapper (for scanning large ticker universes) --------
class ChunkedLoader:
    """Wraps another loader class to fetch a large ticker list in batches --
    most data APIs cap symbols per request -- concatenating the results.
    Skips any chunk that errors (e.g. a delisted ticker) instead of failing
    the whole scan."""

    def __init__(self, tickers, start, end, inner_loader_cls=None, chunk_size=200, inner_kwargs=None):
        self.tickers = list(dict.fromkeys(tickers))
        self.start = start
        self.end = end
        self.inner_loader_cls = inner_loader_cls or YFDataLoader
        self.chunk_size = chunk_size
        self.inner_kwargs = inner_kwargs or {}

    def load_adj_close(self) -> pd.DataFrame:
        frames = []
        for i in range(0, len(self.tickers), self.chunk_size):
            chunk = self.tickers[i:i + self.chunk_size]
            try:
                loader = self.inner_loader_cls(chunk, self.start, self.end, **self.inner_kwargs)
                frames.append(loader.load_adj_close())
            except Exception:
                continue
        if not frames:
            raise ValueError("No price data returned for any ticker in the universe.")
        return pd.concat(frames, axis=1)
