"""SOL 15m tournament: Experiment Detail, Walk Forward, OOS Analysis, Regime Analysis, Trade Analysis, Cost
Analysis, Feature Importance."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tournament_views as T  # noqa: E402

st.set_page_config(page_title="Tournament experiment", layout="wide")
st.title("Experiment detail")
b = T.board("policy")
if b.empty:
    st.info("No finished experiments.")
    st.stop()
entry = st.selectbox("Experiment", b["entry"] + "  ·  " + b["experiment_id"])
eid = entry.split("  ·  ")[1]
row = b.set_index("experiment_id").loc[eid]
ex = T.q("SELECT * FROM tournament.experiments WHERE experiment_id = %s", (eid,)).iloc[0]
tabs = st.tabs(["Summary", "Walk forward", "OOS analysis", "Regimes", "Trades", "Costs", "Feature importance"])

with tabs[0]:
    c = st.columns(6)
    c[0].metric("Net P&L", f"${row['net_usd']:,.0f}")
    c[1].metric("Profit factor", f"{row['pf']:.2f}")
    c[2].metric("Sharpe", f"{row['sharpe']:.2f}")
    c[3].metric("Max drawdown", f"{100 * row['max_dd']:.1f}%")
    c[4].metric("Trades", int(row["trades"]))
    c[5].metric("t-stat", f"{row['t_stat']:.2f}")
    st.json({k: ex[k] for k in ("stage", "entry", "model_id", "dataset_id", "feature_set_id", "n_configs", "seed",
                                "git_commit", "git_dirty_hash", "config_hash", "status", "status_reason")},
            expanded=False)
    st.write("Grid (one config is chosen per fold on validation):")
    st.json(ex["grid"], expanded=False)
    stat = T.q("SELECT metric, value FROM tournament.performance_metrics WHERE experiment_id = %s AND scope = 'stat'",
               (eid,))
    if not stat.empty:
        st.write("Significance / robustness / Monte Carlo:")
        T.table(stat.round(4))

t_all = T.trades(eid)
with tabs[1]:
    wf = T.q("SELECT * FROM tournament.walk_forward_runs WHERE experiment_id = %s ORDER BY fold", (eid,))
    T.table(wf.drop(columns=["experiment_id"]).round(2))
    if not wf.empty:
        st.plotly_chart(T.bars_fig(pd.Series(wf["net_bps"].fillna(0).to_numpy() / 1e4 * T.EQUITY,
                                             index=wf["fold"]), "Net P&L per test fold"), use_container_width=True)
    va = T.q("SELECT fold, config_idx, threshold_idx, val_net_bps, val_trades, chosen FROM tournament.validation_runs "
             "WHERE experiment_id = %s ORDER BY fold, config_idx, threshold_idx", (eid,))
    with st.expander("Validation candidates (selection is made here, never on test)"):
        T.table(va)

with tabs[2]:
    t = T.filter_trades(t_all, "oos")
    if t.empty:
        st.info("No trades (the model abstained everywhere or the filter is empty).")
    else:
        eq = (t.set_index("exit_ts")["net"] * T.EQUITY).cumsum()
        st.plotly_chart(T.equity_fig({"net": eq}), use_container_width=True)
        st.plotly_chart(T.equity_fig({"drawdown": eq - eq.cummax().clip(lower=0)}, "Drawdown ($)"),
                        use_container_width=True)
        m = t.set_index("exit_ts")["net"].resample("MS").sum() * T.EQUITY
        st.plotly_chart(T.bars_fig(m.set_axis(m.index.strftime("%Y-%m")), "Monthly net P&L"), use_container_width=True)
        w = t.set_index("exit_ts")["net"].resample("W").sum() * T.EQUITY
        st.plotly_chart(T.bars_fig(w.set_axis(w.index.strftime("%Y-%m-%d")), "Weekly net P&L"),
                        use_container_width=True)
        fig = go.Figure(go.Histogram(x=t["net_bps"], nbinsx=60, marker_color=T.SERIES[0], marker_line_width=0))
        st.plotly_chart(T.style(fig, "Trade net P&L distribution (bps)", "trades"), use_container_width=True)
        ls = t.groupby("side_name")["net"].agg(["sum", "count"])
        st.plotly_chart(T.bars_fig(ls["sum"] * T.EQUITY, "Long vs short net P&L"), use_container_width=True)

with tabs[3]:
    t = t_all
    if t.empty:
        st.info("No trades.")
    else:
        for dim in ("trend", "vol", "volume", "character"):
            if dim not in t:
                continue
            g = t.groupby(dim)
            tab = pd.DataFrame({"trades": g.size(), "net $": g["net"].sum() * T.EQUITY,
                                "win rate": g["net"].apply(lambda x: (x > 0).mean()),
                                "PF": g["net"].apply(lambda x: x[x > 0].sum() / -x[x < 0].sum() if (x < 0).any()
                                                     else np.nan)})
            dd = {}
            for k, gg in g:
                e = gg.set_index("exit_ts")["net"].cumsum()
                dd[k] = float((e.cummax().clip(lower=0) - e).max())
            tab["max DD"] = pd.Series(dd)
            st.write(f"**{dim}**")
            T.table(tab.reset_index().round(3))

with tabs[4]:
    t = T.filter_trades(t_all, "tr")
    T.table(t[["trade_no", "entry_ts", "exit_ts", "side_name", "bars", "gross_bps", "fee_bps", "slip_bps",
               "funding_bps", "net_bps", "fold", "trend", "vol"]].round(2) if not t.empty else t, height=500)
    if not t.empty:
        fig = go.Figure(go.Histogram(x=t["bars"], nbinsx=40, marker_color=T.SERIES[0], marker_line_width=0))
        st.plotly_chart(T.style(fig, "Holding time (15m bars)", "trades"), use_container_width=True)

with tabs[5]:
    if t_all.empty:
        st.info("No trades.")
    else:
        tot = pd.Series({"Gross": t_all["gross_bps"].sum(), "Fees": -t_all["fee_bps"].sum(),
                         "Slippage": -t_all["slip_bps"].sum(), "Funding": -t_all["funding_bps"].sum(),
                         "Net": t_all["net_bps"].sum()}) / 1e4 * T.EQUITY
        st.plotly_chart(T.bars_fig(tot, "Gross → fees → slippage → funding → net ($)"), use_container_width=True)
        st.write("Every cost configuration (same positions):")
        oos = T.q("SELECT variant, summary FROM tournament.oos_runs WHERE experiment_id = %s", (eid,))
        s = pd.json_normalize(oos["summary"].tolist())
        s.insert(0, "variant", oos["variant"])
        s["net $"] = s["net"] * T.EQUITY
        T.table(s[["variant", "net $", "trades", "gross_bps", "cost_bps", "funding_bps", "pf", "sharpe"]].round(2))

with tabs[6]:
    fi = T.q("SELECT feature, avg(importance) AS importance, count(*) AS folds, min(kind) AS kind "
             "FROM tournament.feature_importance WHERE experiment_id = %s GROUP BY feature ORDER BY 2 DESC LIMIT 30",
             (eid,))
    if fi.empty:
        st.info("This model type reports no importances.")
    else:
        fig = go.Figure(go.Bar(y=fi["feature"][::-1], x=fi["importance"][::-1], orientation="h",
                               marker_color=T.SERIES[0], marker_line_width=0))
        st.plotly_chart(T.style(fig, "Mean importance over folds (top 30)", h=650), use_container_width=True)
