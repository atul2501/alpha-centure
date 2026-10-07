"""Shared data access + chart helpers for the SOL 15m tournament pages (reads only the `tournament` schema)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psycopg
import streamlit as st

from alpha.config import get_settings

EQUITY = 30_000.0
# reference categorical palette (fixed order, never cycled) and diverging poles for gain / loss
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GAIN, LOSS, NEUTRAL = "#2a78d6", "#e34948", "#8a8984"
ANALYSIS = Path("data/experiments/sol15_analysis.json")


@st.cache_resource
def _connection() -> psycopg.Connection:
    return psycopg.connect(get_settings().database_url, autocommit=True)


def conn() -> psycopg.Connection:
    c = _connection()
    if c.closed or c.broken:
        _connection.clear()
        c = _connection()
    return c


def q(sql: str, params=None) -> pd.DataFrame:
    cur = conn().execute(sql, params)
    out = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
    for c in out.columns:
        if isinstance(out[c].dtype, pd.DatetimeTZDtype):
            out[c] = out[c].dt.tz_convert("UTC")
    return out


@st.cache_data(ttl=60)
def board(variant: str = "policy") -> pd.DataFrame:
    d = q("""SELECT e.experiment_id, e.stage, e.entry, e.model_id, m.family, m.complexity, m.experimental,
                    e.feature_set_id, e.n_configs, o.summary,
                    (SELECT status FROM tournament.registry r WHERE r.experiment_id = e.experiment_id
                     ORDER BY changed_at DESC LIMIT 1) AS registry_status
             FROM tournament.experiments e JOIN tournament.models m USING (model_id)
             JOIN tournament.oos_runs o USING (experiment_id)
             WHERE e.status = 'done' AND NOT e.shuffled_control AND e.stage NOT LIKE 'smoke%%'
               AND o.variant = %s""", (variant,))
    if d.empty:
        return d
    s = pd.json_normalize(d.pop("summary").tolist())
    d = pd.concat([d.reset_index(drop=True), s], axis=1)
    d["net_usd"] = d["net"] * EQUITY
    return d.sort_values("net", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=60)
def analysis() -> dict:
    return json.loads(ANALYSIS.read_text()) if ANALYSIS.exists() else {}


@st.cache_data(ttl=60)
def trades(eid: str, variant: str = "policy") -> pd.DataFrame:
    t = q("SELECT * FROM tournament.trades WHERE experiment_id = %s AND variant = %s ORDER BY trade_no", (eid, variant))
    if t.empty:
        return t
    reg = pd.json_normalize(t["regime"].apply(lambda r: r or {}).tolist())
    t = pd.concat([t.drop(columns=["regime"]), reg], axis=1)
    t["net"] = t["net_bps"] / 1e4
    t["side_name"] = np.where(t["side"] > 0, "long", "short")
    return t


def filter_trades(t: pd.DataFrame, key: str) -> pd.DataFrame:
    """The shared trade filters: date range, side, regime, fold."""
    if t.empty:
        return t
    c = st.columns(5)
    lo, hi = t["exit_ts"].min().date(), t["exit_ts"].max().date()
    dr = c[0].date_input("Date range", (lo, hi), min_value=lo, max_value=hi, key=f"{key}_d")
    side = c[1].selectbox("Long / short", ["both", "long", "short"], key=f"{key}_s")
    dim = c[2].selectbox("Regime dimension", ["(all)", "trend", "vol", "volume", "character"], key=f"{key}_r")
    val = c[3].selectbox("Regime", ["(all)"] + (sorted(t[dim].dropna().unique()) if dim != "(all)" and dim in t
                                                 else []), key=f"{key}_v")
    folds = sorted(t["fold"].dropna().unique().astype(int))
    fsel = c[4].multiselect("Folds", folds, default=folds, key=f"{key}_f")
    m = pd.Series(True, index=t.index)
    if isinstance(dr, tuple) and len(dr) == 2:
        m &= (t["exit_ts"].dt.date >= dr[0]) & (t["exit_ts"].dt.date <= dr[1])
    if side != "both":
        m &= t["side_name"] == side
    if dim != "(all)" and val != "(all)":
        m &= t[dim] == val
    m &= t["fold"].isin(fsel)
    return t[m]


def style(fig: go.Figure, title: str = "", y: str = "", h: int = 340) -> go.Figure:
    fig.update_layout(title=title, height=h, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.08, x=0), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False, linecolor="rgba(128,128,128,0.4)")
    fig.update_yaxes(title=y, gridcolor="rgba(128,128,128,0.15)", zerolinecolor="rgba(128,128,128,0.4)")
    return fig


def equity_fig(series: dict[str, pd.Series], title: str = "Cumulative net P&L ($, $30k equity)") -> go.Figure:
    fig = go.Figure()
    for i, (name, s) in enumerate(series.items()):
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name, mode="lines",
                                 line=dict(color=SERIES[i % len(SERIES)] if i < len(SERIES) else NEUTRAL, width=2)))
    return style(fig, title, "$")


def bars_fig(s: pd.Series, title: str, y: str = "$") -> go.Figure:
    fig = go.Figure(go.Bar(x=s.index.astype(str), y=s.values, marker_color=np.where(s.values >= 0, GAIN, LOSS),
                           marker_line_width=0, hovertemplate="%{x}: %{y:,.0f}<extra></extra>"))
    return style(fig, title, y)


def table(df: pd.DataFrame, **kw):
    st.dataframe(df, use_container_width=True, hide_index=True, **kw)
