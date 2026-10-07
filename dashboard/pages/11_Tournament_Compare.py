"""SOL 15m tournament: Model Comparison, Equity Curves, Drawdown."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tournament_views as T  # noqa: E402

st.set_page_config(page_title="Tournament compare", layout="wide")
st.title("Compare models")
b = T.board("policy")
if b.empty:
    st.info("No finished experiments.")
    st.stop()
default = list(b["entry"].head(3)) + (["buy_hold"] if "buy_hold" in set(b["entry"]) else [])
c = st.columns([3, 1])
pick = c[0].multiselect("Models (max 8)", b["entry"].tolist(), default=default[:4], max_selections=8)
variant = c[1].selectbox("Cost configuration", ["policy", "forced", "cost_x1.5", "cost_x2", "slip_x2", "mixed_maker"])
if not pick:
    st.stop()
sel = T.board(variant).set_index("entry").loc[pick].reset_index()
cols = ["entry", "family", "net_usd", "pf", "sharpe", "max_dd", "net_bps", "win_rate", "trades", "fees", "slippage",
        "funding", "folds_positive", "t_stat"]
T.table(sel[cols].round(3))
curves, dds = {}, {}
for _, r in sel.iterrows():
    t = T.trades(r["experiment_id"], variant if variant in ("policy", "forced") else "policy")
    if t.empty:
        continue
    s = (t.set_index("exit_ts")["net"] * T.EQUITY).cumsum()
    curves[r["entry"]] = s
    dds[r["entry"]] = s - s.cummax().clip(lower=0)
st.plotly_chart(T.equity_fig(curves), use_container_width=True)
st.plotly_chart(T.equity_fig(dds, "Drawdown ($)"), use_container_width=True)
if variant not in ("policy", "forced"):
    st.caption("Curves use the policy trades; the table above shows the stressed cost configuration's totals.")
st.subheader("Walk-forward stability: net P&L per test fold ($)")
wf = T.q("SELECT e.entry, w.fold, w.net_bps FROM tournament.walk_forward_runs w JOIN tournament.experiments e "
         "USING (experiment_id) WHERE experiment_id = ANY(%s)", (sel["experiment_id"].tolist(),))
if not wf.empty:
    fig = go.Figure()
    for i, (name, g) in enumerate(wf.groupby("entry")):
        fig.add_trace(go.Bar(x=g["fold"], y=g["net_bps"] / 1e4 * T.EQUITY, name=name, marker_color=T.SERIES[i]))
    st.plotly_chart(T.style(fig, "", "$").update_layout(barmode="group", bargap=0.25, bargroupgap=0.05),
                    use_container_width=True)
