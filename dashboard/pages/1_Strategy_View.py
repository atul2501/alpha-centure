"""Strategy View: how the strategy turns signals into positions, drawn from the decision's StrategyTrace.

Strategy-agnostic: everything below is read from paper_decisions.trace (alpha.strategy.trace), so a new strategy that
fills the same trace shows up here without changes.
"""

import pandas as pd
import plotly.graph_objects as go
import psycopg
import streamlit as st

from alpha.audit import _df
from alpha.config import get_settings
from alpha.live.flowviz import COLOR, LONG, SHORT, cycle_progress, strategy_svg
from alpha.live.workflow import FAIL, IDLE, OK, WARN, fmt_age
from alpha.strategy.trace import BOOK, SIGNAL, StrategyTrace, contributions, gross, scale_factor

st.set_page_config(page_title="Alpha Centure · Strategy View", layout="wide")
settings = get_settings()
BOOK_COLORS = ["#4c9be8", "#d99a00", "#9b59b6", "#2bb5a0", "#e07b39", "#8a8f98"]
ACTION_COLOR = {"REBALANCE": COLOR[OK], "HOLD": COLOR[IDLE], "NO_TRADE": COLOR[WARN], "FLATTEN": COLOR[FAIL],
                "PAUSE": COLOR[WARN], "RESUME": "#5b6ee1", "SNAPSHOT": "#9b59b6"}
DIVERGING = [[0, SHORT], [0.5, "rgba(138,143,152,0.15)"], [1, LONG]]


@st.cache_resource
def connection() -> psycopg.Connection:
    return psycopg.connect(settings.database_url, autocommit=True)


def conn() -> psycopg.Connection:
    c = connection()
    if c.closed or c.broken:
        connection.clear()
        c = connection()
    return c


def short(s: str) -> str:
    return s.replace("USDT", "")


def layout(fig: go.Figure, h: int) -> go.Figure:
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.08))
    return fig


st.title("Strategy View")
st.caption("How the strategy turns market data into positions, step by step. Read-only: built from what the paper "
           "engine recorded at each rebalance.")

eng = (conn().execute("SELECT value FROM paper_state WHERE key = 'engine'").fetchone() or [{}])[0] or {}
c1, c2, c3, c4 = st.columns(4)
c1.metric("Strategy", eng.get("model", "–"))
c2.metric("Rebalances", f"every {eng.get('rebalance_hours', '–')}h")
nxt = eng.get("next_rebalance_decision")
if nxt:
    left = (pd.Timestamp(nxt) - pd.Timestamp.now(tz="UTC")).total_seconds() / 3600
    c3.metric("Next rebalance", f"in {left:.1f}h", str(nxt)[:16] + " UTC", delta_color="off")
    frac = cycle_progress(nxt, eng.get("rebalance_hours") or 0)
    if frac is not None:
        c4.metric("Cycle", f"{frac:.0%} elapsed")
        c4.progress(frac)
else:
    c3.metric("Next rebalance", "unknown")

decs = _df(conn(), """SELECT id, ts, bar_time, action, reason, trace FROM paper_decisions
                      WHERE trace IS NOT NULL ORDER BY id DESC LIMIT 60""")

# ---------------- 1. animated decision flow ----------------
st.subheader("1 · How the decision was built")
if decs.empty:
    st.info(f"No breakdown recorded yet. The engine records one at every rebalance "
            f"(next: {str(nxt)[:16] + ' UTC' if nxt else 'unknown'}). To see it now, run on the server:\n\n"
            "`sudo -u alpha bash -c 'cd /opt/alpha && set -a && . /etc/alpha/.env && "
            "~/.local/bin/uv run python -m alpha.strategy.trace --snapshot'`")
    t = None
else:
    labels = {r.id: f"{r.action} · bar {pd.Timestamp(r.bar_time):%Y-%m-%d %H:%M} UTC · {fmt_age((pd.Timestamp.now(tz='UTC') - pd.Timestamp(r.ts)).total_seconds())}"
              for r in decs.itertuples()}
    pick, replay = st.columns([5, 1])
    did = pick.selectbox("Decision", list(labels), format_func=labels.get, label_visibility="collapsed")
    if replay.button("▶ Replay", width="stretch"):
        st.session_state["replay"] = st.session_state.get("replay", 0) + 1
    row = decs[decs["id"] == did].iloc[0]
    t = StrategyTrace.from_json(row["trace"])
    if t.notes:
        st.caption(t.notes)
    st.html(strategy_svg(t, nonce=st.session_state.get("replay", 0)))
    st.caption("Read left to right: the signals feed each sub-strategy, the sub-strategies are mixed, the mix is "
               "scaled to the risk target, and the result is the position in each coin (green = long, red = short, "
               "thicker = bigger). Hover a box for what it does.")

if t is not None:
    syms = [s for s in t.symbols() if any(abs(x.values.get(s, 0)) > 1e-9 for x in t.layers if x.kind != SIGNAL)
            or abs(t.final.get(s, 0)) > 1e-9] or t.symbols()

    # ---------------- 2. per-coin build-up ----------------
    st.subheader("2 · Why each coin is long or short")
    eq = conn().execute("SELECT equity, positions FROM paper_equity ORDER BY ts DESC LIMIT 1").fetchone()
    held = {}
    if eq and eq[0]:
        for s, p in (eq[1] or {}).items():
            if p.get("qty") and p.get("mark"):
                held[s] = p["qty"] * p["mark"] / float(eq[0])
    contrib = contributions(t)
    fig = go.Figure()
    for i, (k, vals) in enumerate(contrib.items()):
        layer = next(x for x in t.layers if x.key == k)
        fig.add_bar(y=[short(s) for s in syms], x=[vals.get(s, 0.0) for s in syms], name=layer.title,
                    orientation="h", marker_color=BOOK_COLORS[i % len(BOOK_COLORS)])
    fig.add_scatter(y=[short(s) for s in syms], x=[t.final.get(s, 0.0) for s in syms], mode="markers",
                    name="final target (after scaling)", marker=dict(symbol="diamond", size=11, color="#e8e8e8",
                                                                       line=dict(color="#333", width=1.5)))
    if held:
        fig.add_scatter(y=[short(s) for s in syms], x=[held.get(s, 0.0) for s in syms], mode="markers",
                        name="held now", marker=dict(symbol="line-ns", size=16, line=dict(width=3, color="#d99a00")))
    fig.update_layout(barmode="relative", xaxis_title="weight (× equity)", yaxis=dict(autorange="reversed"))
    st.plotly_chart(layout(fig, 60 + 26 * len(syms)), width="stretch")
    sf = scale_factor(t)
    st.caption("Bars = each sub-strategy's share before scaling (they add up to the blend). Diamond = the final "
               "target after the risk scaling" + (f" (×{sf:.2f})" if sf else "") + ". Orange tick = position held now.")

    # ---------------- 3. signals ----------------
    sigs = t.of(SIGNAL)
    if sigs:
        st.subheader("3 · What the signals say")
        z = []
        for x in sigs:
            v = pd.Series({s: x.values.get(s) for s in syms}, dtype=float)
            sd = v.std()
            z.append(((v - v.mean()) / sd).clip(-3, 3).tolist() if sd and sd > 0 else [0.0] * len(syms))
        raw = [[x.values.get(s) for s in syms] for x in sigs]
        fig = go.Figure(go.Heatmap(z=z, x=[short(s) for s in syms], y=[x.title for x in sigs], zmid=0, zmin=-3, zmax=3,
                                   colorscale=DIVERGING, customdata=raw,
                                   hovertemplate="%{y} · %{x}: %{customdata:.3f}<extra></extra>",
                                   colorbar=dict(title="vs other coins", tickvals=[-3, 0, 3], ticktext=["low", "avg", "high"])))
        st.plotly_chart(layout(fig, 70 + 34 * len(sigs)), width="stretch")
        st.caption("Each row compared across coins: green = high, red = low. Hover for the raw value. "
                   "Volatility is not a direction: higher volatility only means a smaller position.")

        feats = next((x.features for x in sigs if x.features), None)
        if feats:
            st.subheader("4 · Inside the forecast model")
            names = list(next(iter(feats.values())).keys())
            imp = sorted(names, key=lambda f: -sum(abs(feats[s].get(f, 0)) for s in feats))
            fs = [s for s in syms if s in feats]
            fig = go.Figure(go.Heatmap(z=[[feats[s].get(f, 0.0) for s in fs] for f in imp], x=[short(s) for s in fs],
                                       y=imp, zmid=0, colorscale=DIVERGING,
                                       hovertemplate="%{y} → %{x}: %{z:+.3f}<extra></extra>",
                                       colorbar=dict(title="push on<br>forecast")))
            st.plotly_chart(layout(fig, 80 + 22 * len(imp)), width="stretch")
            st.caption("How much each input pushed the forecast up (green) or down (red) for each coin; most "
                       "influential inputs on top. The column adds up to that coin's forecast (minus a constant).")

# ---------------- 5. decision timeline ----------------
st.subheader("5 · Every hourly decision (last 14 days)")
tl = _df(conn(), """SELECT ts, bar_time, action, reason FROM paper_decisions
                    WHERE ts > now() - interval '14 days' ORDER BY ts""")
if tl.empty:
    st.info("No decisions yet.")
else:
    fig = go.Figure()
    for a, g in tl.groupby("action"):
        fig.add_scatter(x=g["ts"], y=g["action"], mode="markers", name=a, text=g["reason"],
                        marker=dict(size=9 if a == "HOLD" else 14, color=ACTION_COLOR.get(a, "#888")),
                        hovertemplate="%{x|%Y-%m-%d %H:%M} · %{text}<extra></extra>")
    fig.update_layout(showlegend=False, yaxis_title=None)
    st.plotly_chart(layout(fig, 90 + 30 * tl["action"].nunique()), width="stretch")
    st.caption("One dot per hour. HOLD = waiting for the next rebalance bar; NO_TRADE = skipped for a reason "
               "(hover); gaps = the engine was down.")

# ---------------- 6. target history ----------------
st.subheader("6 · Targets at each rebalance")
th = _df(conn(), """SELECT bar_time, action, targets FROM paper_decisions
                    WHERE action IN ('REBALANCE', 'FLATTEN') ORDER BY id DESC LIMIT 40""")
if th.empty:
    st.info("No rebalances yet.")
else:
    th = th.iloc[::-1]
    w = pd.DataFrame([r or {} for r in th["targets"]], index=pd.to_datetime(th["bar_time"], utc=True)).fillna(0.0)
    w = w[w.abs().max().sort_values(ascending=False).index]
    fig = go.Figure()
    for i, s in enumerate(w.columns):
        fig.add_bar(x=w.index, y=w[s], name=short(s))
    fig.update_layout(barmode="relative", yaxis_title="target weight (× equity)")
    st.plotly_chart(layout(fig, 360), width="stretch")
    st.caption("Above zero = long, below = short. Each bar is one rebalance.")

# ---------------- 7. audit ----------------
if t is not None:
    with st.expander("Audit: every number in this decision", expanded=False):
        tab = pd.DataFrame({x.title: {short(s): x.values.get(s) for s in t.symbols()} for x in t.layers})
        st.dataframe(tab, width="stretch")
        st.json(t.params)
        st.download_button("Download CSV", tab.to_csv().encode(), f"{t.strategy}_{t.bar_time[:13]}.csv", "text/csv")
        st.caption(f"Gross: blend {gross(t.of('blend')[0].values) if t.of('blend') else 0:.2f}x → final "
                   f"{gross(t.final):.2f}x · books: {', '.join(x.title for x in t.of(BOOK))}")
