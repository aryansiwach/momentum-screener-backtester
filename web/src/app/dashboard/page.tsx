"use client";

import { useCallback, useEffect, useRef, useState, type MouseEvent, type ReactNode } from "react";
import { motion } from "motion/react";
import {
  TrendUp,
  TrendDown,
  ChartLineUp,
  CurrencyDollar,
  CircleNotch,
  MagnifyingGlass,
  SpeakerHigh,
  Microphone,
} from "@phosphor-icons/react";
import { firaCode, firaSans } from "./fonts";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Design tokens (scoped to this page, not the global theme) -- Aurora UI
// treatment over a gold/purple fintech palette (ui-ux-pro-max design pass).
// Gold + purple carry the brand/decorative layer (badges, glows, gradients);
// green/red stay reserved strictly for actual price direction -- color
// can't be the only signal for financial gain/loss, so it never doubles as
// a decorative choice.
const c = {
  bg: "#0A0714",
  card: "#181229",
  cardAlt: "#130E22",
  border: "#332A4D",
  borderStrong: "#4C3F73",
  fg: "#F8FAFC",
  muted: "#A79FC2",
  // Lightened from #6E6390 -- the original failed WCAG AA (3.3:1-3.7:1
  // against the three dark backgrounds this token is used on; AA needs
  // 4.5:1 for normal-size text). This token is used everywhere for
  // small captions, timestamps, and disclaimers, not large/decorative
  // text, so AA-large's 3:1 exemption doesn't apply. #8479AC clears
  // 4.5:1 against every background it's actually painted on (verified:
  // 5.04 vs bg, 4.59 vs card, 4.77 vs cardAlt) while staying the
  // dimmest/most secondary text tone on the page.
  mutedDim: "#8479AC",
  accent: "#22C55E",
  accentDim: "#16A34A",
  destructive: "#EF4444",
  gold: "#F59E0B",
  goldBright: "#FBBF24",
  purple: "#8B5CF6",
  purpleBright: "#A78BFA",
};

type Pick = {
  ticker: string;
  momentum_score: number;
  target_weight: number;
  suggested_dollars?: number;
};

type SectorConcentration = {
  sector_weights: Record<string, number>;
  breaches: Record<string, number>;
  within_limits: boolean;
  ticker_sectors: Record<string, string>;
} | null;

type Mover = {
  symbol: string;
  price: number;
  percent_change: number;
  headline: string | null;
  news_flags: string[];
  news_sentiment: number | null;
};
type IntradayLeader = Mover & {
  intraday_trend: TrendRead & { session_range_pct: number; session_change_pct: number };
};
type PremarketWatch = {
  gainers: Mover[];
  losers: Mover[];
  intraday_leaders: IntradayLeader[];
  updated_at: string | null;
  scanning: boolean;
  error: string | null;
};

type WatchlistResponse = {
  source: string;
  picks: Pick[];
  universe_size?: number | null;
  updated_at?: string | null;
  scanning?: boolean;
  sector_concentration?: SectorConcentration;
};

type Performance = {
  starting_equity: number | null;
  latest_equity: number | null;
  total_return_pct: number | null;
};

type SessionStatus = {
  market_open: boolean;
  minutes_to_close: number | null;
  should_flatten: boolean;
  can_open_new_position: boolean;
};

type ChartRange = "1D" | "1W" | "3M" | "YTD" | "1Y" | "ALL";
const CHART_RANGES: ChartRange[] = ["1D", "1W", "3M", "YTD", "1Y", "ALL"];

type HistoryPoint = { date: string; close: number };
type HistoryResponse = { ticker: string; range: ChartRange; granularity: "minute" | "daily"; points: HistoryPoint[] };
type CompanyInfo = { ticker: string; name: string; sector: string | null; industry: string | null };

type TrendRead = {
  last_price: number;
  rsi: number;
  rsi_zone: "overbought" | "oversold" | "neutral";
  macd_histogram: number;
  macd_state: "bullish_accelerating" | "bullish_decelerating" | "bearish_accelerating" | "bearish_decelerating";
  price_vs_sma50_pct: number | null;
  stochastic_k: number | null;
  stochastic_zone: "overbought" | "oversold" | "neutral" | null;
};
type ProjectedRange = {
  horizon_days: number;
  expected_up_pct: number;
  expected_down_pct: number;
  vol_model: string;
};
type CandlePattern = { date: string; pattern: string; bias: "bullish" | "bearish" | "neutral" };
type RiskFlags = { gap_risk: boolean; illiquid: boolean; high_short_interest: boolean; news_flags: string[] };
type InsiderSummary = { buys: number; sells: number; buy_value: number; sell_value: number; lookback_days: number };
type Ownership = { short_interest_pct: number | null; insider_summary: InsiderSummary | null } | null;
type BullBearCase = { bull_points: string[]; bear_points: string[] };
type TickerAnalysis = {
  ticker: string;
  trend: TrendRead;
  projected_range: ProjectedRange[];
  patterns: CandlePattern[];
  risk_flags: RiskFlags;
  ownership: Ownership;
  news_sentiment: number | null;
  bull_bear_case: BullBearCase;
  summary: string;
};

type IntradayAnalysis =
  | { ticker: string; available: false; reason: string }
  | {
      ticker: string;
      available: true;
      session_date: string;
      bars: number;
      session_change_pct: number;
      trend: TrendRead;
      summary: string;
    };

type PositionPreview = {
  ticker: string;
  entry_price: number;
  shares: number;
  dollar_amount: number;
  stop_loss_price: number;
  stop_loss_pct: number;
  take_profit_price: number;
  take_profit_pct: number;
  trailing_stop_pct: number;
  historical_range: {
    horizon_days: number;
    expected_up_pct: number;
    expected_down_pct: number;
    estimated_high_price: number;
    estimated_low_price: number;
  };
  disclaimer: string;
};

function useInterval(callback: () => void, ms: number) {
  useEffect(() => {
    const id = setInterval(callback, ms);
    return () => clearInterval(id);
  }, [callback, ms]);
}

export default function Dashboard() {
  const [data, setData] = useState<WatchlistResponse | null>(null);
  const [performance, setPerformance] = useState<Performance | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [quotes, setQuotes] = useState<Record<string, number | null>>({});
  const [sessionStatus, setSessionStatus] = useState<SessionStatus | null>(null);

  useEffect(() => {
    let cancelled = false;
    const fetchStatus = async () => {
      try {
        const res = await fetch(`${API_BASE}/session/status`);
        if (res.ok && !cancelled) setSessionStatus(await res.json());
      } catch {
        // session status is supplementary -- the dashboard still works without it
      }
    };
    fetchStatus();
    const id = setInterval(fetchStatus, 60_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const [premarket, setPremarket] = useState<PremarketWatch | null>(null);

  // "What's actually moving right now, and why" -- a different question
  // from the 63-day daily momentum score above. Polled the same 60s cadence
  // as the background scanner that produces it.
  useEffect(() => {
    let cancelled = false;
    const fetchPremarket = async () => {
      try {
        const res = await fetch(`${API_BASE}/premarket/watch`);
        if (res.ok && !cancelled) setPremarket(await res.json());
      } catch {
        // pre-market watch is supplementary -- the dashboard still works without it
      }
    };
    fetchPremarket();
    const id = setInterval(fetchPremarket, 60_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const loadQuotes = useCallback(async (tickers: string[]) => {
    if (tickers.length === 0) return;
    try {
      const res = await fetch(`${API_BASE}/quotes?tickers=${tickers.map(encodeURIComponent).join(",")}`);
      if (res.ok) {
        const body: { quotes: Record<string, number | null> } = await res.json();
        setQuotes(body.quotes);
      }
    } catch {
      // live price is supplementary -- the momentum table still works without it
    }
  }, []);

  const [previewTicker, setPreviewTicker] = useState("AAPL");
  const [previewAmount, setPreviewAmount] = useState("1000");
  const [preview, setPreview] = useState<PositionPreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);

  // The selected ticker's chart is a persistent, always-visible panel (not
  // a modal) so it can auto-refresh in real time alongside the rest of the
  // dashboard -- switching tickers just swaps what the panel is watching.
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null);
  const [chartRange, setChartRange] = useState<ChartRange>("3M");
  const [chartData, setChartData] = useState<HistoryResponse | null>(null);
  const [chartError, setChartError] = useState<string | null>(null);
  const [chartLoading, setChartLoading] = useState(false);
  const [companyInfo, setCompanyInfo] = useState<CompanyInfo | null>(null);
  const [searchedScore, setSearchedScore] = useState<number | null>(null);

  // The watchlist table already carries a momentum score for its own 30
  // names. A ticker typed into search won't be in that list, so it needs
  // its own fetch -- scored against the same reference watchlist so the
  // number stays comparable, not a self-rank of 1.0 for whatever's typed.
  useEffect(() => {
    if (!selectedTicker) return;
    if (data?.picks.some((p) => p.ticker === selectedTicker)) {
      setSearchedScore(null);
      return;
    }
    let cancelled = false;
    fetch(`${API_BASE}/momentum/score?ticker=${encodeURIComponent(selectedTicker)}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((body) => {
        if (!cancelled) setSearchedScore(body?.momentum_score ?? null);
      })
      .catch(() => {
        if (!cancelled) setSearchedScore(null);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedTicker, data]);

  const [analysis, setAnalysis] = useState<TickerAnalysis | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  // Trend/pattern/volatility read for whichever ticker is selected -- a
  // slower-moving read than the price chart, so it refreshes on ticker
  // switch rather than every 30s poll.
  useEffect(() => {
    if (!selectedTicker) return;
    let cancelled = false;
    setAnalysisLoading(true);
    setAnalysisError(null);
    fetch(`${API_BASE}/ticker/analysis?ticker=${encodeURIComponent(selectedTicker)}`)
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body.detail || `Request failed (${res.status})`);
        }
        return res.json();
      })
      .then((body) => {
        if (!cancelled) setAnalysis(body);
      })
      .catch((err) => {
        if (!cancelled) {
          setAnalysis(null);
          setAnalysisError(err instanceof Error ? err.message : "Failed to load analysis");
        }
      })
      .finally(() => {
        if (!cancelled) setAnalysisLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedTicker]);

  const [intraday, setIntraday] = useState<IntradayAnalysis | null>(null);
  const [intradayLoading, setIntradayLoading] = useState(false);

  // The Momentum Score above is a 63-day DAILY signal -- it can read
  // "strong" while a ticker is actively declining in today's session, with
  // nothing on screen saying so. Only worth fetching when the user is
  // actually looking at an intraday range, where that gap matters.
  useEffect(() => {
    if (!selectedTicker || (chartRange !== "1D" && chartRange !== "1W")) {
      setIntraday(null);
      return;
    }
    let cancelled = false;
    setIntradayLoading(true);
    fetch(`${API_BASE}/ticker/intraday-analysis?ticker=${encodeURIComponent(selectedTicker)}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((body) => {
        if (!cancelled) setIntraday(body);
      })
      .catch(() => {
        if (!cancelled) setIntraday(null);
      })
      .finally(() => {
        if (!cancelled) setIntradayLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedTicker, chartRange]);

  const loadChart = useCallback(async (ticker: string, range: ChartRange, opts: { withCompany?: boolean } = {}) => {
    setChartError(null);
    setChartLoading(true);

    if (opts.withCompany !== false) {
      setCompanyInfo(null);
      fetch(`${API_BASE}/company/info?ticker=${encodeURIComponent(ticker)}`)
        .then((res) => (res.ok ? res.json() : null))
        .then((info) => setCompanyInfo(info))
        .catch(() => {
          // company name is a nice-to-have -- chart still works without it
        });
    }

    try {
      const res = await fetch(`${API_BASE}/momentum/history?ticker=${encodeURIComponent(ticker)}&range=${range}`);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }
      setChartData(await res.json());
    } catch (err) {
      setChartError(err instanceof Error ? err.message : "Failed to load chart");
    } finally {
      setChartLoading(false);
    }
  }, []);

  const selectTicker = useCallback(
    (ticker: string) => {
      setSelectedTicker(ticker);
      loadChart(ticker, chartRange);
      // Search and the position sizer can trigger this from anywhere on the
      // page -- bring the chart into view instead of updating off-screen.
      document.getElementById("ticker-chart-panel")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    },
    [loadChart, chartRange]
  );

  const changeRange = useCallback(
    (range: ChartRange) => {
      setChartRange(range);
      if (selectedTicker) loadChart(selectedTicker, range);
    },
    [loadChart, selectedTicker]
  );

  const runPreview = useCallback(async () => {
    setPreviewLoading(true);
    setPreviewError(null);
    try {
      const res = await fetch(
        `${API_BASE}/position/preview?ticker=${encodeURIComponent(previewTicker)}&amount=${encodeURIComponent(previewAmount)}`
      );
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }
      setPreview(await res.json());
      // Any ticker the user prices up should also show its chart, not just
      // the ones already in the top-10 momentum watchlist.
      selectTicker(previewTicker);
    } catch (err) {
      setPreview(null);
      setPreviewError(err instanceof Error ? err.message : "Failed to load preview");
    } finally {
      setPreviewLoading(false);
    }
  }, [previewTicker, previewAmount, selectTicker]);

  // Read via ref inside `load` so picking a default ticker doesn't change
  // `load`'s identity every time the selection changes -- that would reset
  // the 30s polling interval on every ticker switch instead of keeping a
  // steady cadence.
  const selectedTickerRef = useRef<string | null>(null);
  useEffect(() => {
    selectedTickerRef.current = selectedTicker;
  }, [selectedTicker]);

  const load = useCallback(async () => {
    // Size suggestions against the real account balance, not a placeholder --
    // a $1,000 account and a $10,000 assumption produce very different
    // "suggested allocation" numbers, and only one of them is real.
    let equity = 10000;
    try {
      const acctRes = await fetch(`${API_BASE}/account`);
      if (acctRes.ok) {
        const acct: { equity: number } = await acctRes.json();
        equity = acct.equity;
      }
    } catch {
      // fall back to the placeholder if Alpaca isn't configured
    }

    try {
      // The full-market scanner (see api.py's background thread) ranks the
      // whole tradable universe, not just the 30-name fallback list -- use
      // it whenever it has actually produced a completed pass yet.
      let json: WatchlistResponse | null = null;
      try {
        const fullRes = await fetch(`${API_BASE}/momentum/full-market/cached?equity=${equity}`);
        if (fullRes.ok) {
          const fullJson: WatchlistResponse = await fullRes.json();
          if (fullJson.picks.length > 0) json = fullJson;
        }
      } catch {
        // full-market cache is a nice-to-have -- fall through to the fixed watchlist
      }

      if (!json) {
        const res = await fetch(`${API_BASE}/momentum/watchlist?top_n=10&equity=${equity}`);
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body.detail || `Request failed (${res.status})`);
        }
        json = await res.json();
      }
      if (!json) throw new Error("No momentum data available");
      setData(json);
      setError(null);
      setLastUpdated(new Date());
      loadQuotes(json.picks.map((p) => p.ticker));

      if (!selectedTickerRef.current && json.picks.length > 0) {
        const top = [...json.picks].sort((a, b) => b.momentum_score - a.momentum_score)[0];
        selectTicker(top.ticker);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load momentum data");
    } finally {
      setLoading(false);
    }

    try {
      const perfRes = await fetch(`${API_BASE}/performance`);
      if (perfRes.ok) setPerformance(await perfRes.json());
    } catch {
      // performance log is optional -- no live-trading history yet is not an error
    }
  }, [loadQuotes, selectTicker]);

  useEffect(() => {
    load();
  }, [load]);
  useInterval(load, 30_000);

  // Real-time chart: re-pull the selected ticker's price history on the
  // same 30s cadence as everything else. Company info doesn't change, so
  // only the price series refreshes.
  useInterval(() => {
    if (selectedTicker) loadChart(selectedTicker, chartRange, { withCompany: false });
  }, 30_000);

  useInterval(() => {
    if (preview) runPreview();
  }, 30_000);

  const top3 = data ? [...data.picks].sort((a, b) => b.momentum_score - a.momentum_score).slice(0, 3) : [];

  return (
    <div
      className={`${firaSans.variable} ${firaCode.variable} relative min-h-screen overflow-hidden px-6 py-10 sm:px-10 lg:px-16`}
      style={{ backgroundColor: c.bg, color: c.fg, fontFamily: "var(--font-fira-sans)" }}
    >
      <AuroraBackground />
      <div className="relative mx-auto max-w-[1400px]">
        <header
          className="mb-8 flex flex-wrap items-end justify-between gap-4 border-b pb-6"
          style={{ borderColor: c.border }}
        >
          <div>
            <h1
              className="bg-clip-text text-3xl font-bold tracking-tight text-transparent"
              style={{
                fontFamily: "var(--font-fira-code)",
                backgroundImage: `linear-gradient(90deg, ${c.goldBright}, ${c.purpleBright})`,
              }}
            >
              Momentum Dashboard
            </h1>
            <p className="mt-1.5 text-sm" style={{ color: c.muted }}>
              Composite momentum ranking. Updates automatically — no reload needed.
            </p>
          </div>
          <div className="flex flex-col items-end gap-1.5 text-xs" style={{ color: c.muted }}>
            {data && (
              <span
                className="rounded-full border px-2.5 py-1 uppercase tracking-wide"
                style={{ borderColor: c.borderStrong }}
              >
                Source: {data.source.replace(/_/g, " ")}
              </span>
            )}
            <span className="flex items-center gap-1.5" role="status" aria-live="polite">
              <LivePulse />
              {lastUpdated ? `Live — updated ${lastUpdated.toLocaleTimeString()}` : "Connecting…"}
            </span>
          </div>
        </header>

        {premarket && (premarket.gainers.length > 0 || premarket.losers.length > 0) && (
          <PremarketWatchSection data={premarket} onSelect={selectTicker} />
        )}

        <div className="grid grid-cols-1 gap-8 xl:grid-cols-[1fr_300px]">
        <div className="min-w-0">

        <TickerSearchBar onSearch={selectTicker} />

        {selectedTicker && (
          <SelectedTickerPanel
            ticker={selectedTicker}
            company={companyInfo}
            data={chartData}
            error={chartError}
            loading={chartLoading}
            momentumScore={data?.picks.find((p) => p.ticker === selectedTicker)?.momentum_score ?? searchedScore}
            tickers={data?.picks.map((p) => p.ticker) ?? []}
            onSelect={selectTicker}
            range={chartRange}
            onRangeChange={changeRange}
            intraday={intraday}
            intradayLoading={intradayLoading}
          />
        )}

        {selectedTicker && (
          <TickerAnalysisPanel data={analysis} loading={analysisLoading} error={analysisError} />
        )}

        {data?.sector_concentration && <SectorConcentrationBanner data={data.sector_concentration} />}

        {top3.length > 0 && (
          <section className="mb-8">
            <div className="mb-3 flex items-center justify-between gap-2">
              <h2 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide" style={{ color: c.muted }}>
                <ChartLineUp size={14} weight="bold" aria-hidden="true" />
                Top 3 today
              </h2>
              <BriefingButton />
            </div>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
              {top3.map((pick, i) => {
                const brand = i === 0 ? c.gold : i === 1 ? c.purple : c.purpleBright;
                return (
                  <motion.button
                    key={pick.ticker}
                    initial={{ opacity: 0, y: 12, scale: 0.97 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    transition={{ duration: 0.35, delay: i * 0.06, ease: [0.16, 1, 0.3, 1] }}
                    whileHover={{ y: -3, boxShadow: `0 8px 30px -8px ${brand}55` }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => selectTicker(pick.ticker)}
                    className="relative cursor-pointer overflow-hidden rounded-xl border p-4 text-left"
                    style={{
                      borderColor: `${brand}55`,
                      background: `linear-gradient(150deg, ${brand}22, ${c.card} 60%)`,
                    }}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-lg font-semibold" style={{ fontFamily: "var(--font-fira-code)" }}>
                        {pick.ticker}
                      </span>
                      <span
                        className="rounded-full px-2 py-0.5 text-[11px] font-bold text-black"
                        style={{ background: `linear-gradient(90deg, ${brand}, ${c.purpleBright})` }}
                      >
                        #{i + 1}
                      </span>
                    </div>
                    <div className="mt-2 flex items-center gap-1 text-sm font-medium" style={{ color: c.accent }}>
                      <TrendUp size={14} weight="bold" aria-hidden="true" />
                      Score {pick.momentum_score.toFixed(2)}
                    </div>
                    {pick.suggested_dollars != null && (
                      <div className="mt-1 flex items-center gap-1 text-xs" style={{ color: c.muted }}>
                        <CurrencyDollar size={12} aria-hidden="true" />
                        Suggested {fmtMoney(pick.suggested_dollars)}
                      </div>
                    )}
                  </motion.button>
                );
              })}
            </div>
          </section>
        )}

        {performance && (
          <section className="mb-8 grid grid-cols-1 gap-3 sm:grid-cols-3">
            <StatCard label="Starting equity" value={fmtMoney(performance.starting_equity)} />
            <StatCard label="Latest equity" value={fmtMoney(performance.latest_equity)} />
            <StatCard
              label="Total return"
              value={performance.total_return_pct != null ? `${performance.total_return_pct}%` : "—"}
              tone={
                performance.total_return_pct == null
                  ? "neutral"
                  : performance.total_return_pct >= 0
                    ? "positive"
                    : "negative"
              }
            />
          </section>
        )}

        {loading && !data && <TableSkeleton />}

        {error && <ErrorBox className="mb-6 rounded-lg border p-4 text-sm">{error}</ErrorBox>}

        {data && data.picks.length > 0 && (
          <div
            className="overflow-x-auto rounded-xl border"
            style={{ borderColor: c.border, boxShadow: `0 0 0 1px transparent, 0 20px 60px -30px ${c.purple}55` }}
          >
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr
                  className="border-b text-left text-xs uppercase tracking-wide"
                  style={{
                    borderColor: c.border,
                    background: `linear-gradient(90deg, ${c.gold}1A, ${c.cardAlt} 40%)`,
                    color: c.muted,
                  }}
                >
                  <th className="px-4 py-3 font-medium">Ticker</th>
                  <th className="px-4 py-3 font-medium">Live price</th>
                  <th className="px-4 py-3 font-medium">Momentum score</th>
                  <th className="px-4 py-3 font-medium">Target weight</th>
                  <th className="px-4 py-3 font-medium text-right">Suggested allocation</th>
                </tr>
              </thead>
              <tbody>
                {data.picks.map((pick, i) => (
                  <motion.tr
                    key={pick.ticker}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.25, delay: Math.min(i, 8) * 0.03 }}
                    onClick={() => selectTicker(pick.ticker)}
                    className="cursor-pointer border-b transition-colors last:border-0 hover:brightness-125"
                    style={{
                      borderColor: c.border,
                      backgroundColor: i % 2 === 0 ? c.bg : c.cardAlt,
                    }}
                  >
                    <td
                      className="px-4 py-3 font-semibold"
                      style={{ fontFamily: "var(--font-fira-code)" }}
                    >
                      {pick.ticker}
                    </td>
                    <td className="px-4 py-3">
                      <LivePrice value={quotes[pick.ticker] ?? null} />
                    </td>
                    <td className="px-4 py-3">
                      <ScoreBar value={pick.momentum_score} />
                    </td>
                    <td className="px-4 py-3 tabular-nums" style={{ color: c.muted }}>
                      {(pick.target_weight * 100).toFixed(1)}%
                    </td>
                    <td
                      className="px-4 py-3 text-right tabular-nums"
                      style={{ fontFamily: "var(--font-fira-code)", color: c.muted }}
                    >
                      {pick.suggested_dollars != null ? fmtMoney(pick.suggested_dollars) : "—"}
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {data && data.picks.length === 0 && !error && (
          <div
            className="rounded-xl border p-10 text-center text-sm"
            style={{ borderColor: c.border, backgroundColor: c.card, color: c.muted }}
          >
            No positive-momentum picks right now.
          </div>
        )}

        <section
          className="mt-10 rounded-xl border p-5"
          style={{
            borderColor: `${c.purple}40`,
            background: `linear-gradient(160deg, ${c.purple}14, ${c.card} 45%)`,
          }}
        >
          <h2
            className="bg-clip-text text-sm font-semibold text-transparent"
            style={{ fontFamily: "var(--font-fira-code)", backgroundImage: `linear-gradient(90deg, ${c.goldBright}, ${c.purpleBright})` }}
          >
            Day-trade position sizer
          </h2>
          <p className="mt-1 text-xs" style={{ color: c.muted }}>
            3% take-profit / 1.5% stop-loss, sized against real trailing volatility. Not a prediction.
          </p>

          <div className="mt-4 flex flex-wrap items-end gap-3">
            <label className="flex flex-col gap-1 text-xs" style={{ color: c.muted }}>
              Ticker
              <input
                value={previewTicker}
                onChange={(e) => setPreviewTicker(e.target.value.toUpperCase())}
                className="w-24 rounded-md border px-2.5 py-2 text-sm outline-none transition-colors focus:ring-2 focus:ring-[#8B5CF6]"
                style={{
                  borderColor: c.borderStrong,
                  backgroundColor: c.bg,
                  color: c.fg,
                  fontFamily: "var(--font-fira-code)",
                }}
              />
            </label>
            <label className="flex flex-col gap-1 text-xs" style={{ color: c.muted }}>
              Amount ($)
              <input
                value={previewAmount}
                onChange={(e) => setPreviewAmount(e.target.value)}
                type="number"
                min="1"
                className="w-32 rounded-md border px-2.5 py-2 text-sm outline-none transition-colors focus:ring-2 focus:ring-[#8B5CF6]"
                style={{
                  borderColor: c.borderStrong,
                  backgroundColor: c.bg,
                  color: c.fg,
                  fontFamily: "var(--font-fira-code)",
                }}
              />
            </label>
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.97 }}
              onClick={runPreview}
              disabled={previewLoading}
              className="flex cursor-pointer items-center gap-2 rounded-md px-4 py-2 text-sm font-semibold text-black transition-opacity disabled:cursor-not-allowed disabled:opacity-60"
              style={{ background: `linear-gradient(90deg, ${c.goldBright}, ${c.purpleBright})` }}
            >
              {previewLoading && <CircleNotch size={14} className="animate-spin" aria-hidden="true" />}
              {previewLoading ? "Loading…" : "Preview"}
            </motion.button>
          </div>

          {previewError && <ErrorBox className="mt-4 rounded-md border p-3 text-sm">{previewError}</ErrorBox>}

          {preview && (
            <>
              <div className="mt-5 mb-2 flex items-center gap-1.5 text-[11px]" style={{ color: c.mutedDim }}>
                <LivePulse size="sm" />
                Auto-updating every 30s
              </div>
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                <StatCard compact label="Entry price" value={`$${preview.entry_price.toFixed(2)}`} />
                <StatCard compact label="Shares" value={preview.shares.toFixed(4)} />
                <StatCard
                  compact
                  label={`Take-profit (${preview.take_profit_pct}%)`}
                  value={`$${preview.take_profit_price.toFixed(2)}`}
                  tone="positive"
                />
                <StatCard
                  compact
                  label={`Stop-loss (${preview.stop_loss_pct}%)`}
                  value={`$${preview.stop_loss_price.toFixed(2)}`}
                  tone="negative"
                />
                <StatCard
                  compact
                  label="Est. high (1σ, 1 day)"
                  value={`$${preview.historical_range.estimated_high_price.toFixed(2)} (+${preview.historical_range.expected_up_pct}%)`}
                  tone="positive"
                />
                <StatCard
                  compact
                  label="Est. low (1σ, 1 day)"
                  value={`$${preview.historical_range.estimated_low_price.toFixed(2)} (${preview.historical_range.expected_down_pct}%)`}
                  tone="negative"
                />
                <StatCard compact label="Trailing stop" value={`${preview.trailing_stop_pct}%`} />
              </div>
              <p className="mt-4 text-xs" style={{ color: c.mutedDim }}>
                {preview.disclaimer}
              </p>
            </>
          )}
        </section>

        </div>

        <aside className="flex flex-col gap-6 xl:sticky xl:top-10 xl:self-start">
          <MarketSessionCard status={sessionStatus} />
          <PlatformInfoCard
            trackedCount={data?.picks.length ?? null}
            source={data?.source ?? null}
            universeSize={data?.universe_size}
          />
        </aside>

        </div>

        <footer className="mt-8 text-xs" style={{ color: c.mutedDim }}>
          Research tool, not investment advice. Momentum scores are computed
          from historical price data and do not guarantee future performance.
        </footer>
      </div>
    </div>
  );
}

function AuroraBackground() {
  // Aurora UI style: large flowing gradient blobs, slow loop, subtle. Purely
  // decorative and behind all content -- respects prefers-reduced-motion
  // via the `motion` library's default reduced-motion handling.
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden="true">
      <motion.div
        className="absolute -left-32 -top-32 h-[28rem] w-[28rem] rounded-full blur-[110px]"
        style={{ background: c.purple, opacity: 0.22 }}
        animate={{ x: [0, 40, -20, 0], y: [0, 30, -10, 0] }}
        transition={{ duration: 22, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.div
        className="absolute -right-40 top-10 h-[26rem] w-[26rem] rounded-full blur-[110px]"
        style={{ background: c.gold, opacity: 0.16 }}
        animate={{ x: [0, -30, 20, 0], y: [0, 20, -30, 0] }}
        transition={{ duration: 26, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.div
        className="absolute bottom-0 left-1/3 h-[24rem] w-[24rem] rounded-full blur-[110px]"
        style={{ background: c.purpleBright, opacity: 0.14 }}
        animate={{ x: [0, 25, -25, 0], y: [0, -25, 15, 0] }}
        transition={{ duration: 30, repeat: Infinity, ease: "easeInOut" }}
      />
    </div>
  );
}

function BriefingButton() {
  const [state, setState] = useState<"idle" | "loading" | "playing" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const play = async () => {
    if (state === "loading" || state === "playing") return;
    setState("loading");
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/briefing/audio`);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audioRef.current = audio;
      audio.onended = () => setState("idle");
      audio.onerror = () => {
        setState("error");
        setError("Playback failed");
      };
      await audio.play();
      setState("playing");
    } catch (err) {
      setState("error");
      setError(err instanceof Error ? err.message : "Failed to load briefing");
    }
  };

  return (
    <div className="flex items-center gap-2">
      <button
        onClick={play}
        disabled={state === "loading"}
        className="flex cursor-pointer items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium transition-colors disabled:cursor-wait"
        style={{ borderColor: c.borderStrong, color: c.muted, backgroundColor: c.cardAlt }}
      >
        {state === "loading" ? (
          <CircleNotch size={13} className="animate-spin" aria-hidden="true" />
        ) : (
          <SpeakerHigh size={13} weight={state === "playing" ? "fill" : "regular"} aria-hidden="true" />
        )}
        {state === "playing" ? "Playing…" : "Listen"}
      </button>
      {error && (
        <span className="text-[11px]" style={{ color: c.destructive }}>
          {error}
        </span>
      )}
    </div>
  );
}

function LivePulse({ size = "md" }: { size?: "sm" | "md" }) {
  const dim = size === "sm" ? "h-1.5 w-1.5" : "h-2 w-2";
  return (
    <span className={`relative flex ${dim}`}>
      <span
        className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-75"
        style={{ backgroundColor: c.accent }}
      />
      <span className={`relative inline-flex ${dim} rounded-full`} style={{ backgroundColor: c.accent }} />
    </span>
  );
}

function MarketSessionCard({ status }: { status: SessionStatus | null }) {
  const open = status?.market_open ?? false;
  return (
    <div className="rounded-xl border p-5" style={{ borderColor: c.border, backgroundColor: c.card }}>
      <h3 className="mb-3 text-xs font-semibold uppercase tracking-wide" style={{ color: c.muted }}>
        Market session
      </h3>
      <div className="flex items-center gap-2">
        <span
          className="h-2.5 w-2.5 rounded-full"
          style={{ backgroundColor: status == null ? c.mutedDim : open ? c.accent : c.destructive }}
        />
        <span className="text-lg font-semibold" style={{ fontFamily: "var(--font-fira-code)" }}>
          {status == null ? "—" : open ? "NYSE open" : "NYSE closed"}
        </span>
      </div>
      {open && status?.minutes_to_close != null && (
        <p className="mt-2 text-sm" style={{ color: c.muted }}>
          Closes in {Math.floor(status.minutes_to_close / 60)}h {Math.floor(status.minutes_to_close % 60)}m
        </p>
      )}
      {status && (
        <div className="mt-4 flex flex-col gap-1.5 text-xs" style={{ color: c.mutedDim }}>
          <span>
            {status.can_open_new_position ? "New positions allowed" : "New positions blocked (near close)"}
          </span>
          {status.should_flatten && <span style={{ color: c.destructive }}>Flatten window active</span>}
        </div>
      )}
    </div>
  );
}

function PlatformInfoCard({
  trackedCount,
  source,
  universeSize,
}: {
  trackedCount: number | null;
  source: string | null;
  universeSize: number | null | undefined;
}) {
  const scanningFullMarket = source === "alpaca_full_market";
  return (
    <div className="rounded-xl border p-5" style={{ borderColor: c.border, backgroundColor: c.card }}>
      <h3 className="mb-3 text-xs font-semibold uppercase tracking-wide" style={{ color: c.muted }}>
        Platform
      </h3>
      <ul className="flex flex-col gap-2.5 text-sm" style={{ color: c.muted }}>
        <li className="flex items-center gap-2">
          <span className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ backgroundColor: c.purple }} />
          Alpaca paper trading account
        </li>
        <li className="flex items-center gap-2">
          <span className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ backgroundColor: c.purple }} />
          {scanningFullMarket && universeSize
            ? `Scanning ${universeSize.toLocaleString()} tickers · top ${trackedCount ?? 10} shown`
            : trackedCount != null
              ? `Ranking top ${trackedCount} of a fixed 30-name list`
              : "Loading watchlist…"}
        </li>
        {source && (
          <li className="flex items-center gap-2">
            <span className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ backgroundColor: c.purple }} />
            Prices from {source.replace(/_/g, " ")}
          </li>
        )}
        {scanningFullMarket && (
          <li className="flex items-center gap-2">
            <span className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ backgroundColor: c.accent }} />
            Excludes single-day spikes (&gt;25% gap) &mdash; no reverse-split/halt/meme pops
          </li>
        )}
        <li className="flex items-center gap-2">
          <span className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ backgroundColor: c.purple }} />
          Refreshes every 30s
        </li>
      </ul>
    </div>
  );
}

function PremarketWatchSection({ data, onSelect }: { data: PremarketWatch; onSelect: (ticker: string) => void }) {
  const lastUpdated = data.updated_at ? new Date(data.updated_at) : null;
  return (
    <section
      className="mb-8 overflow-hidden rounded-xl border p-5"
      style={{
        borderColor: `${c.gold}40`,
        background: `linear-gradient(160deg, ${c.gold}0F, ${c.card} 45%)`,
      }}
    >
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h2 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide" style={{ color: c.muted }}>
          <TrendUp size={14} weight="bold" aria-hidden="true" />
          Pre-market watch — what's moving right now, and why
        </h2>
        <span className="flex items-center gap-1.5 text-[11px]" style={{ color: c.mutedDim }} role="status" aria-live="polite">
          <LivePulse size="sm" />
          {lastUpdated ? `Updated ${lastUpdated.toLocaleTimeString()}` : "Loading…"}
          {data.scanning && " · rescanning"}
        </span>
      </div>

      <IntradayLeadersRow leaders={data.intraday_leaders} onSelect={onSelect} />

      <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
        <MoverColumn label="Gainers" movers={data.gainers} tone={c.accent} onSelect={onSelect} />
        <MoverColumn label="Losers" movers={data.losers} tone={c.destructive} onSelect={onSelect} />
      </div>

      <p className="mt-3 text-[11px]" style={{ color: c.mutedDim }}>
        Real Alpaca market movers, filtered down from the raw feed (mostly warrants and sub-$3 junk) to the same
        clean, liquid universe used everywhere else on this dashboard, with recent headlines attached where
        available. This reflects whatever's moving right now — pre-market before the bell, regular-session
        during market hours — not a prediction of what keeps moving. The % here is vs. yesterday's close (why
        it's a "mover" at all); a ticker can be a big gainer by that measure while fading within today's own
        session — click through to the Intraday Read below for that separate, same-session number.
      </p>
    </section>
  );
}

function IntradayLeadersRow({ leaders, onSelect }: { leaders: IntradayLeader[]; onSelect: (ticker: string) => void }) {
  return (
    <div className="mb-1 rounded-lg border p-3" style={{ borderColor: `${c.gold}30`, backgroundColor: `${c.gold}0A` }}>
      <div className="mb-1.5 flex items-baseline gap-1.5">
        <span className="text-xs font-semibold" style={{ color: c.gold }}>
          Intraday leaders
        </span>
        <span className="text-[11px]" style={{ color: c.mutedDim }}>
          up right now, today's own session — not just vs. prior close — with confirmed RSI/MACD and a 2%+
          session range for real room to move. Not a forecast of where it lands.
        </span>
      </div>
      {leaders.length === 0 ? (
        <span className="text-xs" style={{ color: c.mutedDim }}>
          None of today's clean gainers are currently up within today's own session with confirmed technicals and
          a 2%+ range — that's the filter working, not a data gap. A stock can gap up vs. yesterday and still be
          filtered out here if it's been sliding since today's open.
        </span>
      ) : (
        <div className="flex flex-wrap gap-1.5">
          {leaders.map((m) => (
            <button
              key={m.symbol}
              onClick={() => onSelect(m.symbol)}
              className="flex cursor-pointer items-center gap-1.5 rounded-full border py-1 pl-2.5 pr-2 text-xs transition-colors hover:opacity-90"
              style={{ borderColor: `${c.gold}55`, backgroundColor: c.cardAlt }}
              title={`vs. prior close +${m.percent_change.toFixed(1)}% · RSI ${m.intraday_trend.rsi.toFixed(0)} (${m.intraday_trend.rsi_zone}) · MACD ${m.intraday_trend.macd_state.replace("_", " ")} · today's range ${m.intraday_trend.session_range_pct.toFixed(1)}%`}
            >
              <span className="font-semibold" style={{ fontFamily: "var(--font-fira-code)" }}>
                {m.symbol}
              </span>
              <span className="tabular-nums" style={{ color: c.accent, fontFamily: "var(--font-fira-code)" }}>
                today +{m.intraday_trend.session_change_pct.toFixed(1)}%
              </span>
              <span
                className="rounded-full px-1.5 py-0.5 text-[10px] font-medium tabular-nums"
                style={{ backgroundColor: `${c.gold}22`, color: c.gold }}
              >
                range {m.intraday_trend.session_range_pct.toFixed(1)}%
              </span>
              {m.intraday_trend.macd_state === "bullish_accelerating" && (
                <span
                  className="rounded-full px-1.5 py-0.5 text-[10px] font-medium"
                  style={{ backgroundColor: `${c.accent}22`, color: c.accent }}
                >
                  building
                </span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function MoverColumn({
  label,
  movers,
  tone,
  onSelect,
}: {
  label: string;
  movers: Mover[];
  tone: string;
  onSelect: (ticker: string) => void;
}) {
  if (movers.length === 0) {
    return (
      <div>
        <div className="mb-1.5 text-xs font-semibold" style={{ color: tone }}>
          {label}
        </div>
        <span className="text-xs" style={{ color: c.mutedDim }}>
          None passed the clean-universe/liquidity filter right now.
        </span>
      </div>
    );
  }
  return (
    <div>
      <div className="mb-1.5 flex items-baseline gap-1.5">
        <span className="text-xs font-semibold" style={{ color: tone }}>
          {label}
        </span>
        <span className="text-[10px]" style={{ color: c.mutedDim }}>
          vs. prior close
        </span>
      </div>
      <div className="flex flex-col gap-1.5">
        {movers.map((m) => (
          <button
            key={m.symbol}
            onClick={() => onSelect(m.symbol)}
            className="cursor-pointer rounded-lg border p-2.5 text-left transition-colors hover:opacity-90"
            style={{ borderColor: c.border, backgroundColor: c.cardAlt }}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm font-semibold" style={{ fontFamily: "var(--font-fira-code)" }}>
                {m.symbol}
              </span>
              <span className="flex items-center gap-2 text-xs tabular-nums" style={{ fontFamily: "var(--font-fira-code)" }}>
                <span style={{ color: c.mutedDim }}>${m.price.toFixed(2)}</span>
                <span style={{ color: tone, fontWeight: 600 }}>
                  {m.percent_change >= 0 ? "+" : ""}
                  {m.percent_change.toFixed(1)}%
                </span>
              </span>
            </div>
            {m.headline && (
              <p className="mt-1 text-[11px] leading-snug" style={{ color: c.muted }}>
                {m.headline}
              </p>
            )}
            {m.news_flags.length > 0 && (
              <div className="mt-1 flex flex-wrap gap-1">
                {m.news_flags.map((f) => (
                  <span
                    key={f}
                    className="rounded-full px-1.5 py-0.5 text-[10px]"
                    style={{ backgroundColor: `${c.destructive}22`, color: c.destructive }}
                  >
                    {f}
                  </span>
                ))}
              </div>
            )}
          </button>
        ))}
      </div>
    </div>
  );
}

function SectorConcentrationBanner({ data }: { data: NonNullable<SectorConcentration> }) {
  const [open, setOpen] = useState(!data.within_limits);
  const sorted = Object.entries(data.sector_weights).sort((a, b) => b[1] - a[1]);
  const breached = Object.entries(data.breaches).sort((a, b) => b[1] - a[1]);
  const warn = !data.within_limits;

  return (
    <div
      className="mb-6 overflow-hidden rounded-xl border"
      style={{
        borderColor: warn ? `${c.destructive}55` : c.border,
        backgroundColor: warn ? `${c.destructive}0F` : c.card,
      }}
    >
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full cursor-pointer items-center justify-between gap-2 px-4 py-3 text-left"
      >
        <span className="flex items-center gap-2 text-sm font-medium" style={{ color: warn ? "#FCA5A5" : c.muted }}>
          <span
            className="h-2 w-2 shrink-0 rounded-full"
            style={{ backgroundColor: warn ? c.destructive : c.accent }}
          />
          {warn
            ? `Concentration risk: ${breached.map(([s, w]) => `${s} ${(w * 100).toFixed(0)}%`).join(", ")} of the Top 10`
            : "Sector mix within limits"}
        </span>
        <span className="text-xs" style={{ color: c.mutedDim }}>
          {open ? "Hide" : "Details"}
        </span>
      </button>
      {open && (
        <div className="border-t px-4 py-3" style={{ borderColor: warn ? `${c.destructive}30` : c.border }}>
          <div className="flex flex-wrap gap-1.5">
            {sorted.map(([sector, weight]) => (
              <span
                key={sector}
                className="rounded-full border px-2.5 py-1 text-xs"
                style={{
                  borderColor: weight > 0.4 ? `${c.destructive}55` : c.borderStrong,
                  color: weight > 0.4 ? c.destructive : c.muted,
                }}
              >
                {sector} · {(weight * 100).toFixed(0)}%
              </span>
            ))}
          </div>
          <p className="mt-3 text-[11px] leading-relaxed" style={{ color: c.mutedDim }}>
            Momentum can pick several correlated names riding the same theme — a top-10 list that looks
            diversified by ticker can still be one concentrated bet by sector. Flagged above 40% of the list in
            a single sector.
          </p>
        </div>
      )}
    </div>
  );
}

function TableSkeleton() {
  return (
    <div
      className="overflow-hidden rounded-xl border"
      style={{ borderColor: c.border }}
      role="status"
      aria-busy="true"
      aria-label="Loading momentum data"
    >
      {Array.from({ length: 6 }).map((_, i) => (
        <div
          key={i}
          className="flex items-center gap-4 border-b px-4 py-3.5 last:border-0"
          style={{ borderColor: c.border, backgroundColor: i % 2 === 0 ? c.bg : c.cardAlt }}
        >
          <div className="h-4 w-12 animate-pulse rounded" style={{ backgroundColor: c.border }} />
          <div className="h-4 w-16 animate-pulse rounded" style={{ backgroundColor: c.border }} />
          <div className="h-4 w-28 flex-1 animate-pulse rounded" style={{ backgroundColor: c.border }} />
          <div className="h-4 w-10 animate-pulse rounded" style={{ backgroundColor: c.border }} />
          <div className="h-4 w-16 animate-pulse rounded" style={{ backgroundColor: c.border }} />
        </div>
      ))}
    </div>
  );
}

// The Web Speech API isn't in TypeScript's standard DOM lib -- minimal
// ambient shape for just what's used below, not the full spec.
type SpeechRecognitionResultLike = { transcript: string };
type SpeechRecognitionLike = {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  start: () => void;
  stop: () => void;
  onresult: ((e: { results: SpeechRecognitionResultLike[][] }) => void) | null;
  onerror: ((e: { error: string }) => void) | null;
  onend: (() => void) | null;
};
type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

function TickerSearchBar({ onSearch }: { onSearch: (ticker: string) => void }) {
  const [value, setValue] = useState("");
  const [listening, setListening] = useState(false);
  const [micError, setMicError] = useState<string | null>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);

  // Checked post-mount, not during render -- window.SpeechRecognition is
  // only ever defined client-side, so evaluating this directly in the
  // render body makes the server-rendered HTML (no mic button) disagree
  // with the client's first render (button present), which is exactly
  // the "Hydration failed" class of bug.
  const [speechSupported, setSpeechSupported] = useState(false);
  useEffect(() => {
    const w = window as unknown as { SpeechRecognition?: SpeechRecognitionCtor; webkitSpeechRecognition?: SpeechRecognitionCtor };
    setSpeechSupported(Boolean(w.SpeechRecognition || w.webkitSpeechRecognition));
  }, []);

  const submit = (raw?: string) => {
    const t = (raw ?? value).trim().toUpperCase();
    if (t) onSearch(t);
  };

  const startListening = () => {
    if (listening) return;
    setMicError(null);
    const Ctor =
      (window as unknown as { SpeechRecognition?: SpeechRecognitionCtor }).SpeechRecognition ||
      (window as unknown as { webkitSpeechRecognition?: SpeechRecognitionCtor }).webkitSpeechRecognition;
    if (!Ctor) return;

    const recognition = new Ctor();
    recognition.lang = "en-US";
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.onresult = (e) => {
      // A spoken ticker is a single word -- take the first token so "NVDA
      // please" or similar doesn't get searched verbatim.
      const heard = e.results[0]?.[0]?.transcript ?? "";
      const ticker = heard.trim().split(/\s+/)[0]?.toUpperCase() ?? "";
      if (ticker) {
        setValue(ticker);
        submit(ticker);
      }
    };
    recognition.onerror = (e) => {
      setMicError(e.error === "not-allowed" ? "Microphone access denied" : "Didn't catch that");
      setListening(false);
    };
    recognition.onend = () => setListening(false);

    recognitionRef.current = recognition;
    setListening(true);
    recognition.start();
  };

  return (
    <div className="mb-6 flex items-center gap-2">
      <div
        className="flex flex-1 items-center gap-2 rounded-lg border px-3 py-2.5"
        style={{ borderColor: c.borderStrong, backgroundColor: c.card }}
      >
        <MagnifyingGlass size={18} style={{ color: c.mutedDim }} aria-hidden="true" />
        <input
          value={value}
          onChange={(e) => setValue(e.target.value.toUpperCase())}
          onKeyDown={(e) => {
            if (e.key === "Enter") submit();
          }}
          placeholder="Search any ticker — NVDA, GOOGL, JPM…"
          className="w-full bg-transparent text-sm outline-none"
          style={{ color: c.fg, fontFamily: "var(--font-fira-code)" }}
        />
        {speechSupported && (
          <button
            onClick={startListening}
            aria-label={listening ? "Listening…" : "Search by voice"}
            className="flex cursor-pointer items-center justify-center rounded-full p-1.5 transition-colors"
            style={{ backgroundColor: listening ? `${c.destructive}22` : "transparent" }}
          >
            <Microphone
              size={16}
              weight={listening ? "fill" : "regular"}
              style={{ color: listening ? c.destructive : c.mutedDim }}
              aria-hidden="true"
            />
          </button>
        )}
      </div>
      <button
        onClick={() => submit()}
        className="cursor-pointer rounded-lg px-4 py-2.5 text-sm font-semibold transition-opacity hover:opacity-90"
        style={{ background: `linear-gradient(90deg, ${c.goldBright}, ${c.purpleBright})`, color: "#000" }}
      >
        Search
      </button>
      {micError && (
        <span className="text-[11px]" style={{ color: c.destructive }}>
          {micError}
        </span>
      )}
    </div>
  );
}

function SelectedTickerPanel({
  ticker,
  company,
  data,
  error,
  loading,
  momentumScore,
  tickers,
  onSelect,
  range,
  onRangeChange,
  intraday,
  intradayLoading,
}: {
  ticker: string;
  company: CompanyInfo | null;
  data: HistoryResponse | null;
  error: string | null;
  loading: boolean;
  momentumScore: number | null;
  tickers: string[];
  onSelect: (ticker: string) => void;
  range: ChartRange;
  onRangeChange: (range: ChartRange) => void;
  intraday: IntradayAnalysis | null;
  intradayLoading: boolean;
}) {
  return (
    <section
      id="ticker-chart-panel"
      className="mb-8 overflow-hidden rounded-xl border p-5"
      style={{
        borderColor: `${c.purple}40`,
        background: `linear-gradient(160deg, ${c.purple}14, ${c.card} 40%)`,
        boxShadow: `0 20px 60px -25px ${c.purple}55`,
      }}
    >
      <div className="mb-4 flex flex-wrap gap-1.5">
        {tickers.map((t) => (
          <button
            key={t}
            onClick={() => onSelect(t)}
            className="cursor-pointer rounded-full border px-3 py-1 text-xs font-medium transition-colors"
            style={
              t === ticker
                ? { background: `linear-gradient(90deg, ${c.goldBright}, ${c.purpleBright})`, color: "#000", borderColor: "transparent" }
                : { borderColor: c.borderStrong, color: c.muted, backgroundColor: c.cardAlt }
            }
          >
            {t}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <h3 className="text-xl font-semibold" style={{ fontFamily: "var(--font-fira-code)" }}>
            {ticker}
            {company?.name && (
              <span className="font-sans text-base font-normal" style={{ color: c.muted }}>
                {" "}
                — {company.name}
              </span>
            )}
          </h3>
          {company?.sector && (
            <p className="mt-0.5 text-xs" style={{ color: c.mutedDim }}>
              {company.sector}
              {company.industry ? ` · ${company.industry}` : ""}
            </p>
          )}

          <div className="mt-3 flex flex-wrap gap-1">
            {CHART_RANGES.map((r) => (
              <button
                key={r}
                onClick={() => onRangeChange(r)}
                className="cursor-pointer rounded-md px-2.5 py-1 text-xs font-semibold transition-colors"
                style={
                  r === range
                    ? { backgroundColor: c.purple, color: "#fff" }
                    : { color: c.muted, backgroundColor: "transparent" }
                }
              >
                {r}
              </button>
            ))}
          </div>

          {loading && !data && (
            <div className="flex h-64 items-center justify-center text-sm" style={{ color: c.muted }}>
              <CircleNotch size={18} className="mr-2 animate-spin" aria-hidden="true" />
              Loading chart…
            </div>
          )}

          {error && <ErrorBox className="mt-4 rounded-md border p-3 text-sm">{error}</ErrorBox>}

          {data && data.points.length > 1 && <PriceChart points={data.points} granularity={data.granularity} />}
        </div>

        <div className="flex flex-col items-center justify-center gap-2 border-t pt-5 lg:border-l lg:border-t-0 lg:pl-6 lg:pt-0" style={{ borderColor: c.border }}>
          <MomentumGauge value={momentumScore} />
          <span className="text-xs uppercase tracking-wide" style={{ color: c.mutedDim }}>
            Momentum score
          </span>
          <MomentumScoreExplainer value={momentumScore} />

          {(range === "1D" || range === "1W") && (
            <IntradayReadCard data={intraday} loading={intradayLoading} />
          )}
        </div>
      </div>
    </section>
  );
}

function IntradayReadCard({ data, loading }: { data: IntradayAnalysis | null; loading: boolean }) {
  if (loading && !data) {
    return (
      <div className="mt-3 flex items-center gap-1.5 text-[11px]" style={{ color: c.mutedDim }}>
        <CircleNotch size={11} className="animate-spin" aria-hidden="true" />
        Reading today's session…
      </div>
    );
  }
  if (!data) return null;

  if (!data.available) {
    return (
      <p className="mt-3 max-w-[220px] text-center text-[11px]" style={{ color: c.mutedDim }}>
        {data.reason}
      </p>
    );
  }

  const bullish = data.trend.macd_state.startsWith("bullish");
  const tone = data.session_change_pct >= 0 ? c.accent : c.destructive;

  return (
    <div
      className="mt-3 w-full max-w-[240px] rounded-lg border p-3"
      style={{ borderColor: `${tone}40`, backgroundColor: c.cardAlt }}
    >
      <div className="mb-1 flex items-center justify-between text-[10px] uppercase tracking-wide" style={{ color: c.mutedDim }}>
        <span>Today's session</span>
        <span>{data.bars} bars</span>
      </div>
      <div className="flex items-center gap-1.5 text-sm font-semibold tabular-nums" style={{ color: tone, fontFamily: "var(--font-fira-code)" }}>
        {data.session_change_pct >= 0 ? (
          <TrendUp size={14} weight="bold" aria-hidden="true" />
        ) : (
          <TrendDown size={14} weight="bold" aria-hidden="true" />
        )}
        {data.session_change_pct >= 0 ? "+" : ""}
        {data.session_change_pct.toFixed(2)}% today
      </div>
      <div className="mt-1.5 flex flex-wrap gap-1 text-[10px]" style={{ color: c.muted }}>
        <span className="rounded-full border px-1.5 py-0.5" style={{ borderColor: c.borderStrong }}>
          RSI {data.trend.rsi.toFixed(0)} ({data.trend.rsi_zone})
        </span>
        <span
          className="rounded-full border px-1.5 py-0.5"
          style={{ borderColor: c.borderStrong, color: bullish ? c.accent : c.destructive }}
        >
          MACD {bullish ? "bullish" : "bearish"}
        </span>
      </div>
      <p className="mt-2 text-[10px] leading-relaxed" style={{ color: c.mutedDim }}>
        Intraday only -- a separate read from the daily Momentum Score above, and they can disagree.
      </p>
    </div>
  );
}

function momentumScoreRead(value: number): { label: string; color: string } {
  if (value >= 0.8) return { label: "Strong — top of the pack", color: c.accent };
  if (value >= 0.6) return { label: "Above average", color: c.accent };
  if (value >= 0.4) return { label: "Neutral", color: c.muted };
  if (value >= 0.2) return { label: "Below average", color: c.destructive };
  return { label: "Weak — bottom of the pack", color: c.destructive };
}

function MomentumScoreExplainer({ value }: { value: number | null }) {
  const [open, setOpen] = useState(false);
  const read = value != null ? momentumScoreRead(value) : null;

  return (
    <div className="flex flex-col items-center gap-1.5 text-center">
      {read && (
        <span className="text-xs font-semibold" style={{ color: read.color }}>
          {read.label}
        </span>
      )}
      <button
        onClick={() => setOpen((v) => !v)}
        className="cursor-pointer text-[11px] underline decoration-dotted underline-offset-2"
        style={{ color: c.mutedDim }}
      >
        {open ? "Hide" : "What's this?"}
      </button>
      {open && (
        <p className="max-w-[220px] text-[11px] leading-relaxed" style={{ color: c.mutedDim }}>
          A 0–1 rank of how strongly this ticker is trending versus the tracked watchlist right now, blending
          63-day return (35%), MACD histogram (25%), RSI (15%), price vs. 50-day average (15%), and stochastic
          %K (10%). 1.0 means it&apos;s outperforming the rest of the list on these signals today — not a
          prediction of future returns.
        </p>
      )}
    </div>
  );
}

const ZONE_COLOR: Record<string, string> = {
  overbought: "#F59E0B",
  oversold: "#38BDF8",
  neutral: undefined as unknown as string,
};

function TickerAnalysisPanel({
  data,
  loading,
  error,
}: {
  data: TickerAnalysis | null;
  loading: boolean;
  error: string | null;
}) {
  if (loading && !data) {
    return (
      <section className="mb-8 rounded-xl border p-5" style={{ borderColor: c.border, backgroundColor: c.card }}>
        <div className="flex items-center gap-2 text-sm" style={{ color: c.muted }}>
          <CircleNotch size={16} className="animate-spin" aria-hidden="true" />
          Reading trend, patterns, and risk…
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <ErrorBox as="section" className="mb-8 rounded-xl border p-5 text-sm">
        {error}
      </ErrorBox>
    );
  }

  if (!data) return null;

  const { trend, projected_range, patterns, risk_flags, ownership, bull_bear_case, news_sentiment } = data;
  const anyRiskFlag =
    risk_flags.gap_risk || risk_flags.illiquid || risk_flags.high_short_interest || risk_flags.news_flags.length > 0;

  return (
    <section
      className="mb-8 rounded-xl border p-5"
      style={{ borderColor: c.border, backgroundColor: c.card }}
    >
      <h3 className="mb-3 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide" style={{ color: c.muted }}>
        Technical read
      </h3>

      <p className="mb-4 text-sm leading-relaxed" style={{ color: c.fg }}>
        {data.summary}
      </p>

      {(bull_bear_case.bull_points.length > 0 || bull_bear_case.bear_points.length > 0) && (
        <div className="mb-4">
          <div className="mb-1.5 flex items-center gap-2 text-[11px] uppercase tracking-wide" style={{ color: c.mutedDim }}>
            Bull case vs. bear case
            {news_sentiment != null && (
              <span
                className="rounded-full px-2 py-0.5 normal-case tracking-normal"
                style={{
                  backgroundColor: news_sentiment > 0.05 ? `${c.accent}22` : news_sentiment < -0.05 ? `${c.destructive}22` : c.cardAlt,
                  color: news_sentiment > 0.05 ? c.accent : news_sentiment < -0.05 ? c.destructive : c.muted,
                }}
              >
                News sentiment {news_sentiment > 0 ? "+" : ""}
                {news_sentiment.toFixed(2)}
              </span>
            )}
          </div>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <CaseColumn label="Bull case" points={bull_bear_case.bull_points} tone={c.accent} emptyWord="bullish" />
            <CaseColumn label="Bear case" points={bull_bear_case.bear_points} tone={c.destructive} emptyWord="bearish" />
          </div>
          <p className="mt-2 text-[11px]" style={{ color: c.mutedDim }}>
            Same data, read from both sides — not two opinions, and not a recommendation either way.
          </p>
        </div>
      )}

      <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <TechStatCard
          label="RSI (14d)"
          value={trend.rsi.toFixed(0)}
          tone={trend.rsi_zone === "neutral" ? undefined : ZONE_COLOR[trend.rsi_zone]}
          sub={trend.rsi_zone}
        />
        <TechStatCard
          label="Stochastic %K"
          value={trend.stochastic_k != null ? trend.stochastic_k.toFixed(0) : "—"}
          tone={trend.stochastic_zone && trend.stochastic_zone !== "neutral" ? ZONE_COLOR[trend.stochastic_zone] : undefined}
          sub={trend.stochastic_zone ?? "unavailable"}
        />
        <TechStatCard
          label="vs. 50d avg"
          value={trend.price_vs_sma50_pct != null ? `${trend.price_vs_sma50_pct > 0 ? "+" : ""}${trend.price_vs_sma50_pct.toFixed(1)}%` : "—"}
          tone={trend.price_vs_sma50_pct != null ? (trend.price_vs_sma50_pct >= 0 ? c.accent : c.destructive) : undefined}
        />
        <TechStatCard
          label="MACD"
          value={trend.macd_state.startsWith("bullish") ? "Bullish" : "Bearish"}
          sub={trend.macd_state.endsWith("accelerating") ? "accelerating" : "fading"}
          tone={trend.macd_state.startsWith("bullish") ? c.accent : c.destructive}
        />
      </div>

      <div className="mb-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
        {projected_range.map((r) => (
          <div key={r.horizon_days} className="rounded-lg border p-3" style={{ borderColor: c.border, backgroundColor: c.cardAlt }}>
            <div className="text-[11px] uppercase tracking-wide" style={{ color: c.mutedDim }}>
              Projected range, next {r.horizon_days}d ({r.vol_model})
            </div>
            <div className="mt-1 flex items-baseline gap-2 text-sm font-semibold tabular-nums" style={{ fontFamily: "var(--font-fira-code)" }}>
              <span style={{ color: c.accent }}>+{r.expected_up_pct.toFixed(1)}%</span>
              <span style={{ color: c.mutedDim }}>/</span>
              <span style={{ color: c.destructive }}>{r.expected_down_pct.toFixed(1)}%</span>
            </div>
            <div className="mt-1 text-[11px]" style={{ color: c.mutedDim }}>
              1-standard-deviation statistical range, not a forecast of where it will land.
            </div>
          </div>
        ))}
      </div>

      {patterns.length > 0 && (
        <div className="mb-4">
          <div className="mb-1.5 text-[11px] uppercase tracking-wide" style={{ color: c.mutedDim }}>
            Recent candlestick patterns
          </div>
          <div className="flex flex-wrap gap-1.5">
            {patterns.slice(0, 6).map((p, i) => (
              <span
                key={i}
                className="rounded-full border px-2.5 py-1 text-xs"
                style={{
                  borderColor: p.bias === "bullish" ? `${c.accent}55` : p.bias === "bearish" ? `${c.destructive}55` : c.borderStrong,
                  color: p.bias === "bullish" ? c.accent : p.bias === "bearish" ? c.destructive : c.muted,
                }}
              >
                {p.pattern.replace(/_/g, " ")} · {p.date}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="mb-4">
        <div className="mb-1.5 text-[11px] uppercase tracking-wide" style={{ color: c.mutedDim }}>
          Risk flags
        </div>
        {!anyRiskFlag ? (
          <span className="text-sm" style={{ color: c.accent }}>
            No gap, liquidity, short-interest, or headline risk flags detected.
          </span>
        ) : (
          <div className="flex flex-wrap gap-1.5">
            {risk_flags.gap_risk && <RiskBadge label="Single-day gap/spike in recent history" />}
            {risk_flags.illiquid && <RiskBadge label="Thin trading liquidity" />}
            {risk_flags.high_short_interest && <RiskBadge label="Heavy short interest" />}
            {risk_flags.news_flags.map((f) => (
              <RiskBadge key={f} label={`Flagged headlines: ${f}`} />
            ))}
          </div>
        )}
      </div>

      {ownership && (ownership.short_interest_pct != null || ownership.insider_summary) && (
        <div>
          <div className="mb-1.5 text-[11px] uppercase tracking-wide" style={{ color: c.mutedDim }}>
            Ownership
          </div>
          <div className="flex flex-col gap-1 text-sm" style={{ color: c.fg }}>
            {ownership.short_interest_pct != null && (
              <span>
                Short interest:{" "}
                <span
                  className="font-semibold tabular-nums"
                  style={{ color: risk_flags.high_short_interest ? c.destructive : c.fg }}
                >
                  {(ownership.short_interest_pct * 100).toFixed(1)}%
                </span>{" "}
                of float
              </span>
            )}
            {ownership.insider_summary && (ownership.insider_summary.buys > 0 || ownership.insider_summary.sells > 0) && (
              <span>
                Insider activity ({ownership.insider_summary.lookback_days}d):{" "}
                <span style={{ color: c.accent }}>{ownership.insider_summary.buys} buy</span>
                {", "}
                <span style={{ color: c.destructive }}>{ownership.insider_summary.sells} sell</span> transaction(s)
              </span>
            )}
          </div>
        </div>
      )}

      <p className="mt-4 text-[11px]" style={{ color: c.mutedDim }}>
        Descriptive signals from historical data — not investment advice, and not a guarantee against loss.
      </p>
    </section>
  );
}

// The bull-case and bear-case boxes are the same shape (a tinted card, a
// label, a bulleted list or an empty-state line) with only the color,
// heading, and point list actually differing between them.
function CaseColumn({ label, points, tone, emptyWord }: { label: string; points: string[]; tone: string; emptyWord: string }) {
  return (
    <div className="rounded-lg border p-3" style={{ borderColor: `${tone}30`, backgroundColor: c.cardAlt }}>
      <div className="mb-1.5 text-xs font-semibold" style={{ color: tone }}>
        {label}
      </div>
      {points.length > 0 ? (
        <ul className="flex flex-col gap-1.5 text-xs" style={{ color: c.muted }}>
          {points.map((p, i) => (
            <li key={i}>{p}</li>
          ))}
        </ul>
      ) : (
        <span className="text-xs" style={{ color: c.mutedDim }}>
          No {emptyWord} points from the current data.
        </span>
      )}
    </div>
  );
}

function RiskBadge({ label }: { label: string }) {
  return (
    <span
      className="rounded-full border px-2.5 py-1 text-xs"
      style={{ borderColor: `${c.destructive}55`, backgroundColor: `${c.destructive}14`, color: "#FCA5A5" }}
    >
      {label}
    </span>
  );
}

function TechStatCard({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: string }) {
  return (
    <div className="rounded-lg border p-3" style={{ borderColor: c.border, backgroundColor: c.cardAlt }}>
      <div className="text-[11px] uppercase tracking-wide" style={{ color: c.mutedDim }}>
        {label}
      </div>
      <div
        className="mt-1 text-lg font-semibold tabular-nums"
        style={{ fontFamily: "var(--font-fira-code)", color: tone || c.fg }}
      >
        {value}
      </div>
      {sub && (
        <div className="text-[11px] capitalize" style={{ color: c.mutedDim }}>
          {sub}
        </div>
      )}
    </div>
  );
}

function MomentumGauge({ value }: { value: number | null }) {
  const pct = Math.max(0, Math.min(1, value ?? 0));
  const radius = 70;
  const circumference = Math.PI * radius; // semicircle
  const offset = circumference * (1 - pct);

  return (
    <svg width="180" height="100" viewBox="0 0 180 100">
      <defs>
        <linearGradient id="gaugeGradient" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor={c.gold} />
          <stop offset="100%" stopColor={c.purpleBright} />
        </linearGradient>
      </defs>
      <path
        d="M 20 90 A 70 70 0 0 1 160 90"
        fill="none"
        stroke={c.border}
        strokeWidth="14"
        strokeLinecap="round"
      />
      <path
        d="M 20 90 A 70 70 0 0 1 160 90"
        fill="none"
        stroke="url(#gaugeGradient)"
        strokeWidth="14"
        strokeLinecap="round"
        strokeDasharray={circumference}
        strokeDashoffset={offset}
        style={{ transition: "stroke-dashoffset 0.6s ease-out" }}
      />
      <text
        x="90"
        y="82"
        textAnchor="middle"
        fontSize="26"
        fontWeight="700"
        fill={c.fg}
        style={{ fontFamily: "var(--font-fira-code)" }}
      >
        {value != null ? value.toFixed(2) : "—"}
      </text>
    </svg>
  );
}

function formatAxisLabel(date: string, granularity: "minute" | "daily", multiDay: boolean): string {
  if (granularity === "minute") {
    const [datePart, timePart] = date.split(" ");
    if (!timePart) return date;
    if (!multiDay) return timePart;
    // 1W spans several trading days, so time-of-day alone is ambiguous --
    // prefix the date so a hovered point is unambiguous.
    const monthDay = new Date(`${datePart}T00:00:00`).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
    });
    return `${monthDay}, ${timePart}`;
  }
  return date;
}

function PriceChart({ points, granularity }: { points: HistoryPoint[]; granularity: "minute" | "daily" }) {
  const width = 640;
  const height = 260;
  const padding = 32;
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  const multiDay = granularity === "minute" && points[0].date.slice(0, 10) !== points[points.length - 1].date.slice(0, 10);

  const closes = points.map((p) => p.close);
  const min = Math.min(...closes);
  const max = Math.max(...closes);
  const range = max - min || 1;

  const x = (i: number) => padding + (i / (points.length - 1)) * (width - padding * 2);
  const y = (v: number) => height - padding - ((v - min) / range) * (height - padding * 2);

  const path = points.map((p, i) => `${i === 0 ? "M" : "L"} ${x(i).toFixed(1)} ${y(p.close).toFixed(1)}`).join(" ");
  const areaPath = `${path} L ${x(points.length - 1).toFixed(1)} ${height - padding} L ${x(0).toFixed(1)} ${height - padding} Z`;

  const first = points[0].close;
  const last = points[points.length - 1].close;

  // While hovering, the headline price/change reflect the cursor position --
  // the Robinhood pattern: the number you're looking at is the number under
  // your finger, not a separate tooltip you have to glance between.
  const shown = hoverIndex != null ? points[hoverIndex] : points[points.length - 1];
  const changePct = ((shown.close - first) / first) * 100;
  const positive = changePct >= 0;
  const lineColor = positive ? c.accent : c.destructive;

  const handleMove = (e: MouseEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const relX = ((e.clientX - rect.left) / rect.width) * width;
    const t = (relX - padding) / (width - padding * 2);
    const idx = Math.round(t * (points.length - 1));
    setHoverIndex(Math.max(0, Math.min(points.length - 1, idx)));
  };

  return (
    <div className="mt-4">
      <div className="mb-2 flex items-baseline gap-2">
        <span
          className="text-2xl font-semibold tabular-nums"
          style={{ fontFamily: "var(--font-fira-code)" }}
        >
          ${shown.close.toFixed(2)}
        </span>
        <span
          className="flex items-center gap-1 text-sm font-medium tabular-nums"
          style={{ color: lineColor }}
        >
          {positive ? <TrendUp size={14} weight="bold" aria-hidden="true" /> : <TrendDown size={14} weight="bold" aria-hidden="true" />}
          {positive ? "+" : ""}
          {changePct.toFixed(2)}%
          {hoverIndex == null ? ` over ${points.length} ${granularity === "minute" ? "bars" : "sessions"}` : ""}
        </span>
        {hoverIndex != null && (
          <span className="text-xs" style={{ color: c.mutedDim }}>
            {formatAxisLabel(points[hoverIndex].date, granularity, multiDay)}
          </span>
        )}
      </div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
        className="w-full cursor-crosshair touch-none"
        onMouseMove={handleMove}
        onMouseLeave={() => setHoverIndex(null)}
      >
        <defs>
          <linearGradient id="chartFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={lineColor} stopOpacity="0.25" />
            <stop offset="100%" stopColor={lineColor} stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={areaPath} fill="url(#chartFill)" stroke="none" />
        <path
          d={path}
          fill="none"
          stroke={lineColor}
          strokeWidth="2"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        {hoverIndex != null && (
          <>
            <line
              x1={x(hoverIndex)}
              y1={padding * 0.3}
              x2={x(hoverIndex)}
              y2={height - padding}
              stroke={c.muted}
              strokeWidth="1"
              strokeDasharray="3 3"
            />
            <circle
              cx={x(hoverIndex)}
              cy={y(points[hoverIndex].close)}
              r="4.5"
              fill={lineColor}
              stroke={c.bg}
              strokeWidth="2"
            />
          </>
        )}
      </svg>
      <div className="mt-1 flex justify-between text-[11px]" style={{ color: c.mutedDim }}>
        <span>{formatAxisLabel(points[0].date, granularity, multiDay)}</span>
        <span>{formatAxisLabel(points[points.length - 1].date, granularity, multiDay)}</span>
      </div>
    </div>
  );
}

function LivePrice({ value }: { value: number | null }) {
  const prevRef = useRef<number | null>(value);
  const [flash, setFlash] = useState<"up" | "down" | null>(null);

  useEffect(() => {
    if (value == null || prevRef.current == null) {
      prevRef.current = value;
      return;
    }
    if (value !== prevRef.current) {
      setFlash(value > prevRef.current ? "up" : "down");
      prevRef.current = value;
      const t = setTimeout(() => setFlash(null), 700);
      return () => clearTimeout(t);
    }
  }, [value]);

  if (value == null) return <span style={{ color: c.mutedDim }}>—</span>;

  const flashColor = flash === "up" ? c.accent : flash === "down" ? c.destructive : null;

  return (
    <motion.span
      animate={{
        backgroundColor: flashColor ? `${flashColor}30` : "rgba(0,0,0,0)",
      }}
      transition={{ duration: 0.7 }}
      className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 tabular-nums"
      style={{ fontFamily: "var(--font-fira-code)" }}
    >
      {flash === "up" && <TrendUp size={12} weight="bold" style={{ color: c.accent }} aria-hidden="true" />}
      {flash === "down" && <TrendDown size={12} weight="bold" style={{ color: c.destructive }} aria-hidden="true" />}
      ${value.toFixed(2)}
    </motion.span>
  );
}

function ScoreBar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-24 overflow-hidden rounded-full" style={{ backgroundColor: c.border }}>
        <motion.div
          className="h-full rounded-full"
          style={{ backgroundColor: c.accent }}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        />
      </div>
      <span className="tabular-nums" style={{ color: c.muted, fontFamily: "var(--font-fira-code)" }}>
        {value.toFixed(2)}
      </span>
    </div>
  );
}

type Tone = "positive" | "negative" | "neutral";
const toneColor = (tone: Tone) => (tone === "positive" ? c.accent : tone === "negative" ? c.destructive : c.fg);

// StatCard (boxed, larger) and the old PreviewStat (bare, compact) were the
// same {label, value, tone} card with two size presets -- one component,
// one `compact` switch, instead of two near-duplicate definitions.
function StatCard({
  label,
  value,
  tone = "neutral",
  compact = false,
}: {
  label: string;
  value: string;
  tone?: Tone;
  compact?: boolean;
}) {
  const body = (
    <>
      <div className={compact ? "text-[11px] uppercase tracking-wide" : "text-xs uppercase tracking-wide"} style={{ color: c.muted }}>
        {label}
      </div>
      <div
        className={compact ? "mt-0.5 text-sm font-medium tabular-nums" : "mt-1 text-xl font-semibold tabular-nums"}
        style={{ color: toneColor(tone), fontFamily: "var(--font-fira-code)" }}
      >
        {value}
      </div>
    </>
  );
  return compact ? (
    <div>{body}</div>
  ) : (
    <div className="rounded-xl border p-4" style={{ borderColor: c.border, backgroundColor: c.card }}>
      {body}
    </div>
  );
}

// The same destructive-tinted message box (border/background/text color)
// used to be copy-pasted at every error site with only the box model
// (div/section, margin, radius, padding) actually differing.
function ErrorBox({ as: As = "div", className, children }: { as?: "div" | "section"; className: string; children: ReactNode }) {
  return (
    <As className={className} style={{ borderColor: `${c.destructive}40`, backgroundColor: `${c.destructive}14`, color: "#FCA5A5" }}>
      {children}
    </As>
  );
}

function fmtMoney(value: number | null | undefined) {
  if (value == null) return "—";
  return value.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
}
