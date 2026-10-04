"""Signals page: live LONG / SHORT / PASS decisions, their reasons, and how they turned out."""

import json

import pandas as pd
import plotly.graph_objects as go
import psycopg
import streamlit as st

from alpha.audit import _df
from alpha.config import PERP_SUFFIX, get_settings

st.set_page_config(page_title="Alpha Centure Signals", layout="wide")
settings = get_settings()
UP, DOWN, MUTED = "#1a9e77", "#d6455d", "#8a8f98"
PERPS = [s + PERP_SUFFIX for s in settings.symbols]


@st.cache_resource
def connection() -> psycopg.Connection:
    return psycopg.connect(settings.database_url, autocommit=True)


def conn() -> psycopg.Connection:
    c = connection()
    if c.closed or c.broken:
        connection.clear()
        c = connection()
    return c


st.title("Alpha Centure: Signals")

with st.sidebar:
    st.header("Signals")
    days = st.select_slider("Window", options=[1, 3, 7, 14, 30], value=7, format_func=lambda d: f"{d}d")
    n_rows = st.slider("Rows", 10, 200, 20, step=10)
    only_setups = st.checkbox("Hide bars with no setup", value=True)


@st.fragment(run_every="15s")
def signals_section() -> None:
    c = conn()
    model = _df(c, """SELECT version, status, created_at, train_end, policy, metrics, note FROM model_registry
                      WHERE status IN ('champion', 'degraded') ORDER BY created_at DESC LIMIT 1""")
    if model.empty:
        st.warning("No champion model yet. Train one with `uv run python -m alpha.trainer`.")
        return
    m = model.iloc[0]
    metrics = m["metrics"] if isinstance(m["metrics"], dict) else json.loads(m["metrics"])
    hold = metrics.get("holdout") or {}
    badge = "🟢 champion" if m["status"] == "champion" else "🔴 degraded (all PASS)"
    st.markdown(f"**Model** `{m['version']}` · {badge} · trained until {m['train_end']:%Y-%m-%d} · "
                f"EV threshold {metrics.get('threshold')} · {metrics.get('enabled_cells')} strategy×regime cells enabled")
    if hold:
        avg_r = f"{hold['avg_r']:+.3f}" if hold.get("avg_r") is not None else "n/a"
        st.caption(f"Holdout before promotion: {hold.get('trades', 0)} trades, avg R {avg_r}, "
                   f"max DD {(hold.get('max_dd') or 0):.1%} · {m['note'] or ''}")

    sig = _df(c, """SELECT bar_time, symbol, tf, strategy, side, action, reason, regime, p_win, ev_r, close_px, stop,
                           target, outcome, r, exit_time, scored_at, created_at
                    FROM signals WHERE model_version = %s AND bar_time > now() - make_interval(days => %s)
                    ORDER BY bar_time DESC, symbol, tf""", (m["version"], days))
    if sig.empty:
        st.info("No signals yet in this window. Is `alpha.predict` running?")
        return
    setups = sig[sig["strategy"] != "none"]
    taken = setups[setups["action"] != "PASS"]
    passed = setups[setups["action"] == "PASS"]
    t_scored, p_scored = taken.dropna(subset=["r"]), passed.dropna(subset=["r"])

    k = st.columns(6)
    k[0].metric("Bars evaluated", f"{sig.groupby(['symbol', 'tf', 'bar_time']).ngroups:,}")
    k[1].metric("Setups", f"{len(setups):,}")
    k[2].metric("Trades taken", len(taken))
    k[3].metric("PASS rate (setups)", f"{len(passed) / max(1, len(setups)):.0%}")
    k[4].metric("Taken: avg R (scored)", f"{t_scored['r'].mean():+.2f}" if len(t_scored) else "–",
                help=f"{len(t_scored)} scored trades")
    k[5].metric("Passed: avg R (would-have)", f"{p_scored['r'].mean():+.2f}" if len(p_scored) else "–",
                help="Counterfactual result of setups we passed on. Negative = passing saved money.")

    st.subheader(f"Last {n_rows} decisions")
    view = setups if only_setups else sig
    show = view.head(n_rows).copy()
    show["decision"] = show["action"].map({"LONG": "🟢 LONG", "SHORT": "🔴 SHORT", "PASS": "⚪ PASS"})
    st.dataframe(show[["bar_time", "symbol", "tf", "decision", "strategy", "reason", "regime", "p_win", "ev_r",
                       "close_px", "stop", "target", "outcome", "r"]],
                 hide_index=True, width="stretch",
                 column_config={"p_win": st.column_config.NumberColumn(format="%.2f"),
                                "ev_r": st.column_config.NumberColumn(format="%+.2f"),
                                "r": st.column_config.NumberColumn(format="%+.2f")})

    left, right = st.columns(2)
    with left:
        st.subheader("Current regime")
        latest = sig.dropna(subset=["regime"]).sort_values("bar_time").groupby("symbol").tail(1)
        st.dataframe(latest[["symbol", "regime", "bar_time"]], hide_index=True, width="stretch")
        st.subheader("Why we passed")
        reasons = setups.loc[setups["action"] == "PASS", "reason"].value_counts().rename("setups")
        if len(p_scored):
            reasons = pd.concat([reasons, p_scored.groupby("reason")["r"].mean().rename("would-have avg R")], axis=1)
        st.dataframe(reasons.round(3), width="stretch")
    with right:
        st.subheader("Live results by strategy (taken, scored)")
        if len(t_scored):
            g = t_scored.groupby(["strategy", "regime"])
            st.dataframe(pd.DataFrame({"trades": g.size(), "win_rate": g["r"].apply(lambda r: (r > 0).mean()),
                                       "avg_r": g["r"].mean(), "total_r": g["r"].sum()}).round(3), width="stretch")
        else:
            st.caption("No scored trades yet: outcomes appear once each trade's stop, target or time limit is reached.")
        st.subheader("Setups by strategy")
        st.dataframe(setups.pivot_table(index="strategy", columns="action", values="tf", aggfunc="count",
                                        fill_value=0), width="stretch")


signals_section()

st.divider()
st.subheader("Trades on the chart")
c1, c2, c3 = st.columns(3)
ch_symbol = c1.selectbox("Symbol", PERPS)
ch_tf = c2.selectbox("Timeframe", ["15m", "1h", "5m"])
ch_bars = c3.slider("Bars", 50, 600, 200, step=50)
c = conn()
candles = _df(c, """SELECT open_time, open::float8, high::float8, low::float8, close::float8 FROM candles
                    WHERE symbol = %s AND interval = %s ORDER BY open_time DESC LIMIT %s""",
              (ch_symbol, ch_tf, ch_bars)).sort_values("open_time")
if candles.empty:
    st.info("No candles for this selection.")
else:
    trades = _df(c, """SELECT bar_time, action, strategy, stop, target, close_px, exit_time, r FROM signals
                       WHERE symbol = %s AND tf = %s AND action <> 'PASS' AND bar_time >= %s""",
                 (ch_symbol, ch_tf, candles["open_time"].iloc[0]))
    fig = go.Figure(go.Candlestick(x=candles["open_time"], open=candles["open"], high=candles["high"],
                                   low=candles["low"], close=candles["close"], name=ch_symbol,
                                   increasing_line_color=MUTED, decreasing_line_color=MUTED))
    for t in trades.itertuples():
        color = UP if t.action == "LONG" else DOWN
        end = t.exit_time if pd.notna(t.exit_time) else candles["open_time"].iloc[-1]
        fig.add_trace(go.Scatter(x=[t.bar_time], y=[t.close_px], mode="markers", name=f"{t.action} {t.strategy}",
                                 marker=dict(size=12, color=color, symbol="triangle-up" if t.action == "LONG" else "triangle-down"),
                                 hovertext=f"{t.action} {t.strategy} · R={t.r if pd.notna(t.r) else 'open'}"))
        for y, dash in ((t.stop, "dot"), (t.target, "dash")):
            fig.add_trace(go.Scatter(x=[t.bar_time, end], y=[y, y], mode="lines", showlegend=False,
                                     line=dict(color=color, width=1.5, dash=dash), hoverinfo="skip"))
    fig.update_layout(height=520, showlegend=False, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("▲ LONG / ▼ SHORT entries · dotted = stop · dashed = target")
