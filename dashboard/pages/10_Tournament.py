"""SOL 15m tournament: Overview, Leaderboard, Model Registry, Experiment History."""

import numpy as np
import pandas as pd
import streamlit as st

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tournament_views as T  # noqa: E402
from alpha.tournament.analytics.metrics import composite

st.set_page_config(page_title="Tournament", layout="wide")
st.title("SOL 15m model tournament")
st.caption("DEV-OOS = 12 quarterly walk-forward test folds, 2021-07 → 2024-07. Same data, folds, costs, policy "
           "and sizing for every model. VALID-A/B are never loaded; the only true OOS is forward shadow trading.")

an = T.analysis()
b = T.board("policy")
tabs = st.tabs(["Overview", "Leaderboard", "Model Registry", "Experiment History"])

with tabs[0]:
    if b.empty:
        st.info("No finished experiments yet: `uv run python -m alpha.tournament run configs/tournament/stage_2.yaml`")
        st.stop()
    c = st.columns(5)
    c[0].metric("Verdict", an.get("verdict", "analysis not run"))
    c[1].metric("Experiments", len(b))
    c[2].metric("PBO (all experiments)", f"{an.get('pbo', float('nan')):.2f}")
    c[3].metric("SPA p (best vs flat)", f"{(an.get('spa') or {}).get('spa_p', float('nan')):.3f}")
    c[4].metric("Gate G1", "open" if an.get("g1_open") else "closed")
    fam = b.groupby("family").agg(experiments=("entry", "size"), best_net_usd=("net_usd", "max"),
                                  median_net_usd=("net_usd", "median"), best_t=("t_stat", "max")).reset_index()
    st.subheader("By family")
    T.table(fam.round(2))
    st.subheader("Net P&L vs gross edge per trade")
    import plotly.graph_objects as go

    fig = go.Figure(go.Scatter(x=b["gross_bps"] - b["cost_bps"] - b["funding_bps"].clip(lower=0), y=b["net_usd"],
                               mode="markers", marker=dict(size=9, color=T.SERIES[0], line=dict(width=2, color="white")),
                               text=b["entry"], hovertemplate="%{text}<br>net/trade %{x:.1f} bps<br>$%{y:,.0f}"
                                                              "<extra></extra>"))
    st.plotly_chart(T.style(fig, "Each dot = one experiment", "$ net (DEV-OOS)").update_xaxes(
        title="mean net bps per trade"), use_container_width=True)

with tabs[1]:
    c = st.columns(4)
    fams = c[0].multiselect("Family", sorted(b["family"].unique()), default=sorted(b["family"].unique()))
    models = c[1].multiselect("Model", sorted(b["model_id"].unique()))
    fsets = c[2].multiselect("Feature set", sorted(b["feature_set_id"].dropna().unique()))
    variant = c[3].selectbox("Cost configuration", ["policy", "forced", "cost_x1.25", "cost_x1.5", "cost_x2", "cost_x3",
                                                    "slip_x1.5", "slip_x2", "slip_x3", "mixed_maker"])
    bv = T.board(variant)
    bv = bv[bv["family"].isin(fams)]
    if models:
        bv = bv[bv["model_id"].isin(models)]
    if fsets:
        bv = bv[bv["feature_set_id"].isin(fsets)]
    res = {x["experiment_id"]: x for x in an.get("results", [])}
    bv["dsr"] = bv["experiment_id"].map(lambda e: res.get(e, {}).get("dsr", np.nan))
    bv["score (display only)"] = composite(bv)
    cols = {"entry": "Model", "family": "Family", "net_usd": "Net P&L $", "pf": "PF", "win_rate": "Win rate",
            "net_bps": "Expectancy bps", "sharpe": "Sharpe", "max_dd": "Max DD", "trades": "Trades",
            "fees": "Fees", "slippage": "Slippage", "funding": "Funding", "t_stat": "t", "folds_positive": "Folds +",
            "dsr": "DSR", "registry_status": "Registry", "score (display only)": "Score"}
    lb = bv[list(cols)].rename(columns=cols)
    for k in ("Fees", "Slippage", "Funding"):
        lb[k] = lb[k] * T.EQUITY
    lb.insert(0, "Rank", np.arange(1, len(lb) + 1))
    T.table(lb.round(3), height=600)
    st.caption("Raw numbers first; the composite score (35/20/15/10/8/5/5/2) is shown for orientation only and is "
               "never used for selection. Ranking = net P&L of the chosen cost configuration.")

with tabs[2]:
    reg = T.q("""SELECT DISTINCT ON (r.experiment_id) r.experiment_id, e.entry, e.model_id, r.status, r.reason,
                        r.changed_at FROM tournament.registry r JOIN tournament.experiments e USING (experiment_id)
                 WHERE e.stage NOT LIKE 'smoke%%' ORDER BY r.experiment_id, r.changed_at DESC""")
    if reg.empty:
        st.info("Registry empty.")
    else:
        st.bar_chart(reg["status"].value_counts())
        T.table(reg.sort_values(["status", "entry"]))
        st.caption("Statuses: EXPERIMENTAL → VALIDATING → PASSED / REJECTED → PAPER → PRODUCTION → RETIRED. "
                   "PAPER and PRODUCTION are set by hand only (never from backtest P&L).")

with tabs[3]:
    h = T.q("""SELECT experiment_id, stage, entry, model_id, status, status_reason, n_configs, seed, git_commit,
                      git_dirty_hash, config_hash, created_at, finished_at FROM tournament.experiments
               ORDER BY created_at DESC""")
    stages = st.multiselect("Stage", sorted(h["stage"].unique()), default=sorted(h["stage"].unique()))
    T.table(h[h["stage"].isin(stages)], height=600)
