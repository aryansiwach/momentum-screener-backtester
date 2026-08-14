# app.py
import os
import sys
import time
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------- Project path so we can import the local momo package ----------
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from momo.orchestrator import MomentumPipeline
from momo.data_layer import DataLoader, YFDataLoader
from momo.screener import Screener


# ----------------------- Helpers -----------------------
def _to_df(x, name="value"):
    if isinstance(x, pd.DataFrame):
        return x
    if isinstance(x, pd.Series):
        return x.to_frame(name)
    if x is None:
        return pd.DataFrame()
    return pd.DataFrame(x)


def _csv_bytes(df: pd.DataFrame) -> bytes:
    df = _to_df(df)
    df = df.dropna(how="all")
    return df.to_csv(index=True).encode("utf-8")


def build_excel(scores, weights, equity, stats):
    scores = _to_df(scores).dropna(how="all")
    weights = _to_df(weights).dropna(how="all")
    equity_df = _to_df(equity, "Equity").dropna(how="all")
    stats_df = _to_df(stats, "Value").dropna(how="all")

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        if not scores.empty:
            scores.to_excel(writer, sheet_name="Scores")
        if not weights.empty:
            weights.to_excel(writer, sheet_name="Weights")
        if not equity_df.empty:
            equity_df.to_excel(writer, sheet_name="Equity")
        if not stats_df.empty:
            stats_df.to_excel(writer, sheet_name="Stats")
    buf.seek(0)
    return buf


@st.cache_data(show_spinner=False, ttl=60 * 60)
def cached_prices(tickers_tuple, start, end):
    dl = YFDataLoader(
        tickers=list(tickers_tuple),
        start=str(start),
        end=str(end),
        freq="1d",
        pad_lookback=300,
        min_non_na=60,
        auto_ffill=True,
    )
    return dl.load_adj_close()


# ----------------------- Page -----------------------
st.set_page_config(page_title="Momentum Screener", page_icon="📈", layout="wide")

st.markdown(
    """
<style>
/* layout */
.block-container { padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1180px; }
h1,h2,h3 { letter-spacing: -0.02em; }

/* subtle subtitle */
.subtle { opacity: .78; font-size: .95rem; margin-top: -.35rem; }

/* center stage */
.stage {
  margin: 0.9rem auto 1.35rem auto;
  padding: 1.25rem 1.25rem;
  border-radius: 22px;
  border: 1px solid rgba(255,255,255,0.10);
  background: radial-gradient(circle at 12% 0%, rgba(120,119,255,.20), rgba(0,0,0,0));
}

/* "google-like" search */
.searchbar input {
  border-radius: 999px !important;
  padding: 0.78rem 1.2rem !important;
  font-size: 1.05rem !important;
}

/* card container */
.card {
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 18px;
  padding: 1rem 1rem;
  background: rgba(255,255,255,0.03);
}

/* primary button */
.stButton>button {
  border-radius: 999px;
  padding: .72rem 1.3rem;
  font-weight: 700;
  border: 1px solid rgba(124,92,255,.85);
  box-shadow: 0 10px 30px rgba(124,92,255,.22);
  font-size: 1.05rem;
  height: 3rem;
}
.stButton>button:hover { transform: translateY(-1px); }

/* download buttons */
.stDownloadButton>button {
  border-radius: 14px;
  padding: .58rem .95rem;
  font-weight: 650;
}

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 10px; }
.stTabs [data-baseweb="tab"] {
  border-radius: 12px 12px 0 0;
  background: rgba(255,255,255,0.04);
  padding-top: 10px; padding-bottom: 10px;
}

/* small section titles */
.section-title { font-weight: 700; margin: 0.2rem 0 0.6rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("# Momentum Screener & Backtester")
st.markdown(
    '<div class="subtle">Composite momentum ranking + backtest (RSI, MACD, SMA trend, Stochastic, returns). Works with one ticker or many.</div>',
    unsafe_allow_html=True,
)

# ----------------------- Centered "Google-like" configuration -----------------------
st.markdown('<div class="stage">', unsafe_allow_html=True)
_, mid, _ = st.columns([1, 2.3, 1])

with mid:
    # Search bar
    st.markdown('<div class="searchbar">', unsafe_allow_html=True)
    tickers_text = st.text_input(
        "Tickers",
        value="AAPL MSFT NVDA META GOOGL AMZN TSLA BRK-B JPM XOM SPY QQQ",
        placeholder="Type one ticker (AAPL) or many (AAPL MSFT TSLA)...",
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Dates + source (centered)
    d1, d2, d3 = st.columns([1, 1, 1])
    with d1:
        start = st.date_input("Start", pd.to_datetime("2019-01-01"))
    with d2:
        end = st.date_input("End", pd.to_datetime("2024-12-31"))
    with d3:
        datasource = st.selectbox("Data", ["Yahoo Finance (real)", "Synthetic (demo)"], index=0)

    # Parse tickers (safe)
    tickers = [t.strip().upper() for t in tickers_text.replace(",", " ").split() if t.strip()]
    if len(tickers) == 0:
        tickers = ["AAPL"]
    n = len(tickers)

    # Collapsible advanced settings (in the center)
    with st.expander("Advanced settings", expanded=False):
        colA, colB = st.columns(2)

        with colA:
            if n <= 1:
                top_n = 1
                st.write("Top N (long): **1** (single ticker)")
            else:
                top_n = st.slider("Top N (long)", 1, n, min(4, n), 1)

        with colB:
            if n <= 1:
                short_n = 0
                st.write("Bottom N (short): **0** (needs ≥ 2 tickers)")
            else:
                short_n = st.slider("Bottom N (short)", 0, n - 1, min(2, n - 1), 1)

        rebalance_label = st.selectbox("Rebalance", ["Monthly", "Weekly"], index=0)
        rebalance = {"Monthly": "ME", "Weekly": "W"}[rebalance_label]
        costs_bps = st.slider("Transaction cost (bps)", 0, 50, 5, 1)

        st.markdown("**Screener lookbacks**")
        lb1, lb2, lb3, lb4 = st.columns(4)
        with lb1:
            mom_lb = st.number_input("Momentum (days)", 10, 260, 63, 1)
        with lb2:
            sma_lb = st.number_input("SMA window", 5, 260, 50, 1)
        with lb3:
            rsi_lb = st.number_input("RSI window", 3, 50, 14, 1)
        with lb4:
            stoch_lb = st.number_input("Stochastic", 5, 50, 14, 1)

        st.markdown("**Screener weights**")
        w1, w2, w3, w4, w5 = st.columns(5)
        with w1:
            w_mom = st.slider("Momentum", 0.0, 1.0, 0.35, 0.05)
        with w2:
            w_rsi = st.slider("RSI", 0.0, 1.0, 0.15, 0.05)
        with w3:
            w_macd = st.slider("MACD", 0.0, 1.0, 0.25, 0.05)
        with w4:
            w_sma = st.slider("SMA trend", 0.0, 1.0, 0.15, 0.05)
        with w5:
            w_stoch = st.slider("Stochastic", 0.0, 1.0, 0.10, 0.05)

    # Defaults if expander never opened (Streamlit still defines vars when created,
    # but just in case you comment the expander later)
    if "top_n" not in locals():
        top_n = 1 if n <= 1 else min(4, n)
    if "short_n" not in locals():
        short_n = 0 if n <= 1 else min(2, n - 1)
    if "rebalance" not in locals():
        rebalance = "ME"
    if "costs_bps" not in locals():
        costs_bps = 5
    if "mom_lb" not in locals():
        mom_lb, sma_lb, rsi_lb, stoch_lb = 63, 50, 14, 14
        w_mom, w_rsi, w_macd, w_sma, w_stoch = 0.35, 0.15, 0.25, 0.15, 0.10

    run_clicked = st.button("Run analysis", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)  # close stage

st.caption("Tip: BRK-B is the correct Yahoo ticker for Berkshire Hathaway Class B.")

if not run_clicked:
    st.stop()

# ----------------------- Progress + Timing -----------------------
t0 = time.time()
progress = st.progress(0, text="Starting…")

screener_kwargs = {
    "lookbacks": {"mom": int(mom_lb), "sma": int(sma_lb), "rsi": int(rsi_lb), "stoch": int(stoch_lb)},
    "weights": {"mom": w_mom, "rsi": w_rsi, "macd": w_macd, "sma": w_sma, "stoch": w_stoch},
}

try:
    progress.progress(10, text="Preparing data loader…")

    if datasource.startswith("Yahoo"):
        progress.progress(20, text="Downloading prices (cached after first run)…")
        _ = cached_prices(tuple(tickers), str(start), str(end))

        data_loader_cls = YFDataLoader
        data_loader_kwargs = dict(
            freq="1d",
            pad_lookback=max(120, int(mom_lb) + 30),
            min_non_na=60,
            auto_ffill=True,
        )
    else:
        data_loader_cls = DataLoader
        data_loader_kwargs = {}

    progress.progress(35, text="Building pipeline…")

    pipeline = MomentumPipeline(
        tickers=tickers,
        start=str(start),
        end=str(end),
        top_n=int(top_n),
        short_n=int(short_n),
        rebalance=rebalance,
        costs_bps=int(costs_bps),
        screener_kwargs=screener_kwargs,
        data_loader_cls=data_loader_cls,
        data_loader_kwargs=data_loader_kwargs,
    )

    progress.progress(55, text="Running screener + backtest…")
    results = pipeline.run()

    progress.progress(80, text="Computing picks + formatting outputs…")

except Exception as e:
    progress.empty()
    st.error(f"Run failed: {e}")
    st.stop()

elapsed = time.time() - t0
progress.progress(100, text=f"Done in {elapsed:.2f}s")
time.sleep(0.25)
progress.empty()
st.success(f"Done in {elapsed:.2f} seconds")

# ----------------------- Unpack results -----------------------
prices = results.get("prices")
scores = _to_df(results.get("scores"))
weights = _to_df(results.get("weights"))
returns = results.get("returns")
equity = results.get("equity")
stats = results.get("stats")

if prices is None or getattr(prices, "empty", True):
    st.error("No price data returned. Check tickers/date range.")
    st.stop()

# ----------------------- Latest picks -----------------------
scr = Screener(**screener_kwargs)
cscores = _to_df(scr.composite_scores(prices))
rebal_dates = cscores.resample(rebalance).last().index
last_dt = rebal_dates[-1]
last_row = cscores.loc[:last_dt].iloc[-1]
top_picks = last_row.nlargest(int(top_n))
bottom_picks = last_row.nsmallest(int(short_n)) if int(short_n) > 0 else pd.Series(dtype=float)

# ----------------------- Quick metrics -----------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("CAGR", f"{stats['CAGR']*100:.2f}%")
m2.metric("Sharpe", f"{stats['Sharpe']:.2f}")
m3.metric("Sortino", f"{stats['Sortino']:.2f}")
m4.metric("Max Drawdown", f"{stats['Max Drawdown']*100:.1f}%")

# ----------------------- Tabs -----------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Charts", "📋 Picks", "📄 Tables", "⬇️ Downloads"])

with tab1:
    cA, cB = st.columns([2, 1])
    with cA:
        fig1 = plt.figure(figsize=(10, 4))
        _to_df(equity, "Equity").plot(ax=plt.gca())
        plt.title("Equity Curve")
        plt.tight_layout()
        st.pyplot(fig1)

        ppy = 252
        window = 63
        rs = (returns.rolling(window).mean() * ppy) / (returns.rolling(window).std() * np.sqrt(ppy))
        fig2 = plt.figure(figsize=(10, 3))
        rs.plot()
        plt.title(f"Rolling Sharpe (window={window})")
        plt.tight_layout()
        st.pyplot(fig2)

        fig3 = plt.figure(figsize=(10, 4))
        subset = scores.dropna(how="all").tail(150)
        if subset.empty:
            st.warning("Scores are empty (not enough data). Try a wider date range.")
        else:
            plt.imshow(subset.T, aspect="auto", cmap="coolwarm", interpolation="none")
            plt.colorbar(label="Composite Score")
            plt.yticks(range(len(subset.columns)), subset.columns)
            plt.title("Composite Momentum Scores (recent)")
            plt.tight_layout()
            st.pyplot(fig3)

    with cB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Latest rebalance</div>', unsafe_allow_html=True)
        st.caption(f"{last_dt.date()}")
        st.markdown("**Top longs**")
        st.dataframe(top_picks.round(4).to_frame("Score"), use_container_width=True)
        st.markdown("**Bottom shorts**")
        if int(short_n) > 0:
            st.dataframe(bottom_picks.round(4).to_frame("Score"), use_container_width=True)
        else:
            st.info("No shorts selected.")
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    left, right = st.columns(2)
    with left:
        st.markdown("### Long picks")
        st.dataframe(top_picks.round(4).to_frame("Score"), use_container_width=True)
    with right:
        st.markdown("### Short picks")
        if int(short_n) > 0:
            st.dataframe(bottom_picks.round(4).to_frame("Score"), use_container_width=True)
        else:
            st.info("No shorts selected.")

with tab3:
    st.markdown("### Scores (tail)")
    st.dataframe(scores.dropna(how="all").tail(60).round(4), use_container_width=True)

    st.markdown("### Weights (tail)")
    st.dataframe(weights.dropna(how="all").tail(60).round(4), use_container_width=True)

    st.markdown("### Stats")
    st.dataframe(pd.DataFrame(stats, columns=["Value"]).round(4), use_container_width=True)

with tab4:
    scores_clean = scores.dropna(how="all")
    weights_clean = weights.dropna(how="all")
    excel_file = build_excel(scores_clean, weights_clean, equity, stats)

    c1, c2, c3, c4 = st.columns(4)
    c1.download_button(
        "Excel (all sheets)",
        data=excel_file,
        file_name="momentum_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    c2.download_button(
        "Scores CSV",
        data=_csv_bytes(scores_clean),
        file_name="scores.csv",
        mime="text/csv",
        use_container_width=True,
    )
    c3.download_button(
        "Weights CSV",
        data=_csv_bytes(weights_clean),
        file_name="weights.csv",
        mime="text/csv",
        use_container_width=True,
    )
    c4.download_button(
        "Equity CSV",
        data=_csv_bytes(_to_df(equity, "Equity")),
        file_name="equity.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.download_button(
        "Stats CSV",
        data=_csv_bytes(pd.DataFrame(stats, columns=["Value"])),
        file_name="stats.csv",
        mime="text/csv",
        use_container_width=True,
    )
