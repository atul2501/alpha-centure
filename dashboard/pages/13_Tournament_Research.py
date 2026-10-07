"""SOL 15m tournament: Feature Ablation, Order Book Analysis, Robustness, Monte Carlo, Significance."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tournament_views as T  # noqa: E402

st.set_page_config(page_title="Tournament research", layout="wide")
st.title("Ablation, order book, robustness")
b = T.board("policy")
f = T.board("forced")
an = T.analysis()
if b.empty:
    st.info("No finished experiments.")
    st.stop()
tabs = st.tabs(["Feature ablation", "Order book", "Robustness", "Monte Carlo", "Significance"])


def edge_table(mask_entries):
    x = b[b["entry"].str.startswith(mask_entries)][["experiment_id", "entry", "net_usd", "pf", "t_stat", "trades"]]
    y = f.set_index("experiment_id")[["gross_bps", "cost_bps"]].rename(columns=lambda c: "forced " + c)
    x = x.join(y, on="experiment_id")
    x["forced gross / cost"] = x["forced gross_bps"] / x["forced cost_bps"]
    return x.drop(columns=["experiment_id"]).sort_values("entry")


with tabs[0]:
    x = edge_table(("abl_", "loo_"))
    T.table(x.round(3))
    if not x.empty:
        fig = go.Figure(go.Bar(x=x["entry"], y=x["forced gross / cost"], marker_color=T.SERIES[0],
                               marker_line_width=0))
        fig.add_hline(y=1.0, line_dash="dot", line_color=T.NEUTRAL, annotation_text="gross = cost")
        st.plotly_chart(T.style(fig, "Gross edge per trade / round-trip cost (forced variant)", "ratio"),
                        use_container_width=True)
    st.caption("abl_* add groups incrementally; loo_* drop one group from core. A group 'creates edge' only if "
               "adding it raises the gross / cost ratio towards and past 1.")

with tabs[1]:
    x = edge_table(("ob_",))
    T.table(x.round(3))
    st.caption("Same window for with / without (test folds 2023-10 → 2024-07; book_depth_5m starts 2023-01). "
               "Top-of-book, microprice and L2 OFI are not testable historically (collected since 2026-10-04).")

res = {x["experiment_id"]: x for x in an.get("results", [])}
top = [e for e in an.get("top", []) if e in set(b["experiment_id"])]
with tabs[2]:
    if not top:
        st.info("Run `uv run python -m alpha.tournament analyze` first.")
    else:
        rows = []
        for e in top:
            r = T.q("SELECT variant, (summary->>'net')::float8 AS net FROM tournament.oos_runs WHERE experiment_id = %s",
                    (e,)).set_index("variant")["net"]
            s = res.get(e, {})
            rows.append({"model": b.set_index("experiment_id").loc[e, "entry"], **{k: r.get(k) for k in (
                "policy", "cost_x1.25", "cost_x1.5", "cost_x2", "cost_x3", "slip_x1.5", "slip_x2", "slip_x3",
                "mixed_maker")}, **{k: s.get(k) for k in ("th_x0.8", "th_x0.9", "th_x1.1", "th_x1.2", "hold_x0.5",
                                                          "hold_x2", "delay_1bar", "param_stable")}})
        d = pd.DataFrame(rows)
        T.table(d.round(3))
        fig = go.Figure()
        stress = ["policy", "cost_x1.25", "cost_x1.5", "cost_x2", "cost_x3"]
        for i, r in d.head(8).iterrows():
            fig.add_trace(go.Scatter(x=[1, 1.25, 1.5, 2, 3], y=[r[k] * T.EQUITY for k in stress], name=r["model"],
                                     mode="lines+markers", line=dict(width=2, color=T.SERIES[i % 8]),
                                     marker=dict(size=8)))
        fig.add_hline(y=0, line_color=T.NEUTRAL)
        st.plotly_chart(T.style(fig, "Net P&L as costs scale", "$").update_xaxes(title="cost multiplier"),
                        use_container_width=True)

with tabs[3]:
    rows = [{"model": b.set_index("experiment_id").loc[e, "entry"],
             **{k: res.get(e, {}).get(k) for k in ("mc_p_profit", "mc_p_ruin", "mc_expected_dd", "mc_worst_dd",
                                                    "mc_pnl_p05", "mc_pnl_p50", "mc_pnl_p95")}} for e in top]
    T.table(pd.DataFrame(rows).round(4))
    st.caption("5,000 runs each: 5-20% missed trades, fees x U(0.9, 1.5), slippage x lognormal(0, 0.35), return "
               "noise, reshuffled order. Ruin = -30% of equity.")

with tabs[4]:
    c = st.columns(3)
    c[0].metric("Experiments (N for DSR)", an.get("n_experiments", 0))
    c[1].metric("PBO", f"{an.get('pbo', np.nan):.2f}")
    c[2].metric("SPA p / RC p", f"{(an.get('spa') or {}).get('spa_p', np.nan):.3f} / "
                                f"{(an.get('spa') or {}).get('rc_p', np.nan):.3f}")
    rows = [{"model": b.set_index("experiment_id").loc[e, "entry"],
             **{k: res.get(e, {}).get(k) for k in ("dsr", "net_ci_lo", "net_ci_hi", "sharpe_ci_lo", "sharpe_ci_hi",
                                                    "p_net_le_0", "shuffled_t")},
             "failed gates": "; ".join(res.get(e, {}).get("failed", []))} for e in top]
    T.table(pd.DataFrame(rows).round(3))
    st.write("Gate G1 candidates:")
    T.table(pd.DataFrame(an.get("g1", [])))
