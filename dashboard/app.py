"""Audit dashboard: uv run streamlit run dashboard/app.py"""

import pandas as pd
import plotly.graph_objects as go
import psycopg
import streamlit as st
from plotly.subplots import make_subplots

from alpha import audit
from alpha.config import PERP_SUFFIX, get_settings

st.set_page_config(page_title="Alpha Centure Data Audit", layout="wide")
settings = get_settings()

UP, DOWN = "#1a9e77", "#d6455d"
ALL_SYMBOLS = list(dict.fromkeys(f.db_symbol for f in settings.candle_feeds()))


@st.cache_resource
def connection() -> psycopg.Connection:
    return psycopg.connect(settings.database_url, autocommit=True)


def conn() -> psycopg.Connection:
    c = connection()
    if c.closed or c.broken:
        connection.clear()
        c = connection()
    return c


st.title("Alpha Centure: Data Audit")
st.caption(f"Perp (.P): {', '.join(settings.symbols)} × {', '.join(settings.perp_intervals) or 'off'} · "
           f"Spot: {', '.join(settings.intervals) or 'off'} · auto-refresh 5s")

with st.sidebar:
    st.header("Filters")
    f_symbol = st.selectbox("Symbol (last-fetched table)", ["All", *ALL_SYMBOLS])
    f_interval = st.selectbox("Interval (last-fetched table)", ["All", *settings.all_intervals])
    n_rows = st.slider("Rows", 10, 100, 10, step=10)


@st.fragment(run_every="5s")
def live_section() -> None:
    c = conn()
    health = audit.candle_health(c, settings.all_intervals, ALL_SYMBOLS)
    gaps = audit.candle_gaps(c, settings.all_intervals, ALL_SYMBOLS)
    bad = audit.ohlc_violations(c)
    mism = audit.resample_mismatch(c)

    stale = int(health["stale"].sum()) if len(health) else 0
    expected = len(settings.candle_feeds())
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Streams live", f"{len(health) - stale}/{expected}")
    k2.metric("Missing candles (7d)", int(gaps["missing"].sum()) if len(gaps) else 0)
    k3.metric("OHLC violations (24h)", len(bad))
    k4.metric("1m→5m mismatches (24h)", len(mism))
    k5.metric("Total candles (span)", f"{int(health['bars'].sum()):,}" if len(health) else 0)

    st.subheader(f"Last {n_rows} candles fetched")
    st.dataframe(
        audit.last_candles(c, n_rows, None if f_symbol == "All" else f_symbol,
                           None if f_interval == "All" else f_interval),
        hide_index=True, width="stretch",
    )

    st.subheader(f"Last {n_rows} fetch events (all feeds)")
    st.dataframe(audit.last_fetches(c, n_rows), hide_index=True, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Candle health")
        if len(health):
            health = health.assign(status=health["stale"].map({True: "🔴 stale", False: "🟢 live"}))
            st.dataframe(health[["status", "symbol", "interval", "last_close", "bars", "first_open"]],
                         hide_index=True, width="stretch")
    with right:
        st.subheader("Other feeds")
        st.dataframe(audit.feed_health(c), hide_index=True, width="stretch")
        st.subheader("Websocket events (24h)")
        st.dataframe(audit.ws_events(c), hide_index=True, width="stretch")


live_section()

st.divider()
with st.expander("Audit details: gaps, OHLC violations, resample mismatches", expanded=False):
    c = conn()
    st.markdown("**Gaps in last 7 days.** Exchange outages can't be repaired; anything else is retried every 30 min.")
    st.dataframe(audit.candle_gaps(c, settings.all_intervals, ALL_SYMBOLS), hide_index=True, width="stretch")
    st.markdown("**OHLC sanity violations (24h)**")
    st.dataframe(audit.ohlc_violations(c), hide_index=True, width="stretch")
    st.markdown("**5m candles rebuilt from 1m that differ from Binance 5m (24h)**")
    st.dataframe(audit.resample_mismatch(c), hide_index=True, width="stretch")

st.subheader("Chart")
cc1, cc2, cc3 = st.columns(3)
ch_symbol = cc1.selectbox("Symbol", ALL_SYMBOLS)
ch_interval = cc2.selectbox("Interval", settings.all_intervals, index=min(3, len(settings.all_intervals) - 1))
ch_bars = cc3.slider("Bars", 50, 1000, 200, step=50)


def load_chart(symbol: str, interval: str, bars: int):
    c = conn()
    candles = audit._df(c, """
        SELECT open_time, open::float8, high::float8, low::float8, close::float8, volume::float8
        FROM candles WHERE symbol = %s AND interval = %s ORDER BY open_time DESC LIMIT %s
    """, (symbol, interval, bars)).sort_values("open_time")
    if candles.empty:
        return candles, pd.DataFrame(), pd.DataFrame()
    start = candles["open_time"].iloc[0]
    base = symbol.removesuffix(PERP_SUFFIX)  # futures stats are keyed by the plain symbol
    oi = audit._df(c, "SELECT ts, sum_open_interest_value FROM open_interest WHERE symbol = %s AND ts >= %s ORDER BY ts",
                   (base, start))
    fund = audit._df(c, "SELECT ts, last_funding_rate FROM premium_snap WHERE symbol = %s AND ts >= %s ORDER BY ts",
                     (base, start))
    return candles, oi, fund


candles, oi, fund = load_chart(ch_symbol, ch_interval, ch_bars)
if candles.empty:
    st.info("No candles yet. Is the collector running?")
else:
    rows = 2 + (not oi.empty) + (not fund.empty)
    heights = [0.55, 0.15] + [0.15] * (rows - 2)
    titles = [f"{ch_symbol} {ch_interval}", "Volume"] + (["Open interest (USD)"] if not oi.empty else []) \
        + (["Funding rate (predicted)"] if not fund.empty else [])
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=heights, subplot_titles=titles)
    fig.add_trace(go.Candlestick(
        x=candles["open_time"], open=candles["open"], high=candles["high"], low=candles["low"],
        close=candles["close"], name="Price", increasing_line_color=UP, decreasing_line_color=DOWN,
    ), row=1, col=1)
    vol_colors = [UP if c >= o else DOWN for o, c in zip(candles["open"], candles["close"])]
    fig.add_trace(go.Bar(x=candles["open_time"], y=candles["volume"], marker_color=vol_colors, name="Volume"),
                  row=2, col=1)
    r = 3
    if not oi.empty:
        fig.add_trace(go.Scatter(x=oi["ts"], y=oi["sum_open_interest_value"], line=dict(width=2), name="OI"),
                      row=r, col=1)
        r += 1
    if not fund.empty:
        fig.add_trace(go.Scatter(x=fund["ts"], y=fund["last_funding_rate"] * 100, line=dict(width=2),
                                 name="Funding %"), row=r, col=1)
    fig.update_layout(height=300 + 160 * rows, showlegend=False, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
