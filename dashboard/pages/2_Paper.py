"""Paper trading page: mainnet paper account (V4_carry), decisions, orders, fills, execution quality."""

import pandas as pd
import plotly.graph_objects as go
import psycopg
import streamlit as st

from alpha.audit import _df
from alpha.config import get_settings

st.set_page_config(page_title="Alpha Centure Paper", layout="wide")
settings = get_settings()
UP, DOWN = "#1a9e77", "#d6455d"


@st.cache_resource
def connection() -> psycopg.Connection:
    return psycopg.connect(settings.database_url, autocommit=True)


def conn() -> psycopg.Connection:
    c = connection()
    if c.closed or c.broken:
        connection.clear()
        c = connection()
    return c


st.title("Alpha Centure: Paper trading (mainnet data, simulated fills)")
st.caption("No real orders are ever sent. Profitability is not decidable from a 15-30 day run; "
           "see `uv run python -m alpha.live.report` for the engineering gate and paper vs shadow backtest.")

eq = _df(conn(), "SELECT ts, equity, gross_lev, net_lev, margin_ratio, fees, funding FROM paper_equity ORDER BY ts")
if eq.empty:
    st.info("No paper data yet. Start the engine: `uv run python -m alpha.live.engine` (systemd: alpha-paper).")
    st.stop()

last = eq.iloc[-1]
start = eq["equity"].iloc[0]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Equity", f"${last['equity']:,.2f}", f"{last['equity'] / start - 1:+.2%}")
peak = eq["equity"].cummax()
c2.metric("Drawdown from peak", f"{1 - last['equity'] / peak.iloc[-1]:.2%}")
c3.metric("Gross leverage", f"{last['gross_lev']:.2f}x", f"net {last['net_lev']:+.2f}x")
c4.metric("Fees paid", f"${last['fees']:,.2f}")
c5.metric("Funding paid", f"${last['funding']:,.2f}")

fig = go.Figure(go.Scatter(x=eq["ts"], y=eq["equity"], line=dict(color=UP, width=1.5), name="equity"))
fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="USDT")
st.plotly_chart(fig, width="stretch")

mon = conn().execute("SELECT value FROM paper_state WHERE key = 'monitor'").fetchone()
if mon:
    m = mon[0]
    (st.success if m.get("active", True) else st.warning)(
        f"Strategy monitor: {m.get('reason')} (checked {m.get('checked') or '-'})")

st.subheader("Decisions")
st.dataframe(_df(conn(), """SELECT ts, bar_time, action, reason, round(equity::numeric, 2) AS equity, targets, current
                            FROM paper_decisions ORDER BY id DESC LIMIT 50"""), width="stretch")

st.subheader("Fills")
fills = _df(conn(), """SELECT f.ts, f.symbol, CASE WHEN f.side > 0 THEN 'BUY' ELSE 'SELL' END AS side, f.qty, f.price,
                              f.liquidity, round(f.fee::numeric, 4) AS fee, round(f.slippage_bps::numeric, 2) AS slip_bps,
                              f.latency_ms, f.beyond_book
                       FROM paper_fills f ORDER BY f.id DESC LIMIT 200""")
st.dataframe(fills, width="stretch")
if not fills.empty:
    notional = (fills["qty"] * fills["price"])
    maker = notional[fills["liquidity"] == "maker"].sum() / notional.sum()
    st.caption(f"Maker share of traded notional: {maker:.0%} (model 60%) · "
               f"avg slippage vs arrival mid: {fills['slip_bps'].astype(float).mean():.2f} bps")

st.subheader("Orders")
st.dataframe(_df(conn(), """SELECT ts, symbol, kind, side, price, qty, filled, status, queue_ahead, closed_at
                            FROM paper_orders ORDER BY id DESC LIMIT 100"""), width="stretch")
