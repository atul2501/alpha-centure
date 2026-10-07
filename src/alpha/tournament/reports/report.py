"""reports/MODEL_TOURNAMENT_REPORT.md from the tournament database + data/experiments/sol15_analysis.json.

    uv run python -m alpha.tournament report
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.tournament.analytics.selection import OUT as ANALYSIS, load_board
from alpha.tournament.database.repo import Repo
from alpha.tournament.execution.fills import CostConfig, side_costs
from alpha.tournament.features.groups import NOT_TESTABLE

REPORT = Path("reports/MODEL_TOURNAMENT_REPORT.md")
EQUITY = 30_000.0
PRIOR_TRIALS = 160  # earlier project research (memory: research-outcome-2026-10), not part of this tournament's N


def _f(x, d=2, pct=False, usd=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    if usd:
        return f"${x * EQUITY:,.0f}"
    if pct:
        return f"{100 * x:.1f}%"
    return f"{x:.{d}f}" if isinstance(x, (float, np.floating)) else str(x)


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_none_\n"
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(out) + "\n"


def _row(r, stat=None) -> dict:
    stat = stat or {}
    return {"Model": r["entry"], "Family": r["family"], "Net": _f(r["net"], usd=True), "Net %": _f(r["net"], pct=True),
            "PF": _f(r["pf"]), "Win": _f(r["win_rate"], pct=True), "Exp (bps)": _f(r["net_bps"], 1),
            "Sharpe": _f(r["sharpe"]), "Max DD": _f(r["max_dd"], pct=True), "Trades": int(r["trades"]),
            "Fees": _f(r["fees"], usd=True), "Slippage": _f(r["slippage"], usd=True),
            "Funding": _f(r["funding"], usd=True), "t": _f(r["t_stat"]), "Folds+": _f(r.get("folds_positive"), pct=True),
            "DSR": _f(stat.get("dsr")), "Status": stat.get("status", "")}


def leakage_tests() -> tuple[str, bool]:
    p = subprocess.run(["uv", "run", "pytest", "tests/tournament", "-q", "-p", "no:warnings"], capture_output=True,
                       text=True)
    tail = [l for l in p.stdout.splitlines() if l.strip()][-1:] or ["(no output)"]
    return tail[0], p.returncode == 0


def write_report(conn) -> Path:
    repo = Repo(conn)
    b = load_board(repo)
    an = json.loads(ANALYSIS.read_text()) if ANALYSIS.exists() else {}
    res = {x["experiment_id"]: x for x in an.get("results", [])}
    ex = repo.df("SELECT e.stage, e.entry, e.model_id, m.family, m.experimental, e.status, e.status_reason "
                 "FROM tournament.experiments e JOIN tournament.models m USING (model_id) "
                 "WHERE e.stage NOT LIKE 'smoke%%' AND e.stage <> 'control' ORDER BY e.stage, e.entry")
    ds = repo.df("SELECT * FROM tournament.datasets ORDER BY created_at DESC LIMIT 1")
    L = [f"# SOL 15-min Model Tournament Report", "",
         f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by `alpha.tournament report`._", ""]
    verdict = an.get("verdict", "analysis not run")
    n = an.get("n_experiments", len(b))
    best = b.iloc[0] if len(b) else None
    stat = lambda e: {**res.get(e, {}), "status": res.get(e, {}).get("status", "")}
    # ---------------- executive summary
    L += ["## Executive Summary", "", f"**Verdict: {verdict}.**", ""]
    bh = b[b["entry"] == "buy_hold"]
    bh_net = float(bh["net"].iloc[0]) if len(bh) else np.nan
    rnd = b[b["entry"] == "random"]
    L += [f"- {n} walk-forward experiments were evaluated on SOLUSDT perp 15m DEV data (DEV-OOS = 12 quarterly test "
          f"folds, 2021-07 → 2024-07), all with the same data, folds, cost model, policy and sizing. "
          f"{(ex['status'] == 'gated_out').sum()} entries were gated out (gate G1) and "
          f"{(ex['status'] == 'not_run').sum()} could not run in this environment.",
          f"- Trials for the deflated Sharpe: N = {n} (this tournament). The project ran ~{PRIOR_TRIALS} earlier "
          "trials on overlapping data (other strategies); they are disclosed, not added to N.",
          f"- Probability of backtest overfitting across all experiments (CSCV): **{_f(an.get('pbo'))}**; Hansen SPA "
          f"p-value of the best vs FLAT: **{_f((an.get('spa') or {}).get('spa_p'), 3)}** (White RC p "
          f"{_f((an.get('spa') or {}).get('rc_p'), 3)}).",
          f"- Benchmarks on the same window: buy-and-hold SOL {_f(bh_net, pct=True)} "
          f"({_f(bh_net, usd=True)} on $30k), random entries at matched turnover "
          f"{_f(float(rnd['net'].iloc[0]) if len(rnd) else np.nan, pct=True)}."]
    if best is not None:
        L.append(f"- Highest DEV-OOS net P&L: **{best['entry']}** ({best['family']}): {_f(best['net'], usd=True)} "
                 f"({_f(best['net'], pct=True)}), PF {_f(best['pf'])}, t {_f(best['t_stat'])}, "
                 f"DSR {_f(stat(best['experiment_id']).get('dsr'))}; status {stat(best['experiment_id'])['status']}.")
    g1 = an.get("g1", [])
    L.append(f"- Gate G1 (escalate deep/transformer/SSM/foundation/RL to a full sweep): "
             f"**{'OPEN' if an.get('g1_open') else 'CLOSED'}**. Best cheap gross-edge/cost ratios: "
             + ", ".join(f"{x['entry']} {_f(x['edge_to_cost_forced'])}x (shuffled t {_f(x['shuffled_t'])})" for x in g1))
    L += ["", "Prior evidence (stated before the tournament, project memory 2026-10): an earlier 23-coin screen "
          "found that 0 of 60 order-flow / book features covered costs at 15m / 1h / 4h, and LightGBM did worse than "
          "its shuffled-label control on hourly data. This tournament is an independent re-test on SOL alone at 15m.", ""]
    # ---------------- dataset
    L += ["## Dataset", ""]
    if len(ds):
        d = ds.iloc[0]
        L += [f"- `{d['dataset_id']}`: {d['n_rows']:,} bars of SOLUSDT.P 15m, {d['start_ts']:%Y-%m-%d} → "
              f"{d['end_ts']:%Y-%m-%d} (exclusive). DEV only: the guard (`validation/guard.py`) refuses any bar "
              "≥ 2024-07-01; VALID-A / VALID-B are consumed and never loaded.", "",
              "| Feature group | First full bar | Share of bars |", "|---|---|---|"]
        for g, c in (d["coverage"] or {}).items():
            L.append(f"| {g} | {c.get('first') or '–'} | {_f(c.get('share'), pct=True)} |")
        L += ["| REGIME | built per fold by hybrids (train-only fits) | – |"]
        for k, v in NOT_TESTABLE.items():
            L.append(f"| {k} | not testable historically | {v} |")
    L += ["", "Folds: expanding train from 2020-09-14; 90-day validation; quarterly test folds 2021-07 → 2024-07; "
          "purge = 18 bars (max label horizon 16 + 2) + 16-bar embargo between train / validation / test. "
          "Groups with late history (OPEN_INTEREST / DERIVATIVES from 2021-12, ORDER_BOOK from 2023-01) only get "
          "folds with ≥ 180 days of that history in training.", ""]
    # ---------------- costs
    cfg = CostConfig()
    L += ["## Cost Assumptions", "",
          f"- Execution: taker on every side (primary). Fees VIP0: taker {cfg.model.fee_taker_bps} bps, maker "
          f"{cfg.model.fee_maker_bps} bps. Spread: 2 × live median ({cfg.spread_bps:.2f} bps full spread). Impact: "
          f"linear book on $30k notional vs the thinner side's depth within 1% (capped at the DEV 5th percentile). "
          f"Latency: {cfg.model.latency_ms:.0f} ms drift. Funding: actual settlements crossed, long pays a "
          "positive rate.", "- Decision at the 15m close, fill at the next bar's open; one SOL position, notional "
          "= 1 × $30k equity (leverage cap 3× never reached).",
          "- Variants run for every experiment: costs ×1.25 / 1.5 / 2 / 3, slippage ×1.5 / 2 / 3, and a mixed "
          "maker execution (60% passive fills at 2 bps with adverse selection = half spread, 40% chased).", ""]
    try:
        from alpha.tournament.experiments.matrix import context

        ctx = context("real", conn)
        sc = side_costs(ctx.raw, cfg)
        L += [f"- Realized per-side cost on DEV bars: median {np.median(sc.total):.2f} bps "
              f"(fee {np.median(sc.fee):.2f} + slippage {np.median(sc.slip):.2f}); round trip median "
              f"{2 * np.median(sc.total):.2f} bps, 95th percentile {2 * np.quantile(sc.total, 0.95):.2f} bps.", ""]
    except Exception:
        pass
    # ---------------- models tested
    L += ["## Models Tested", ""]
    t = ex.groupby(["family", "status"]).size().unstack(fill_value=0)
    L += [_table(t.reset_index()), ""]
    L += ["Full list (stage / entry / model / status):", "", "<details><summary>expand</summary>", ""]
    L += [_table(ex[["stage", "entry", "model_id", "family", "status"]]), "</details>", ""]
    # ---------------- top tables
    L += ["## Top 20 Models (DEV-OOS net P&L, policy variant)", "",
          _table(pd.DataFrame([_row(r, stat(r["experiment_id"])) for _, r in b.head(20).iterrows()])), ""]
    b["dsr"] = b["experiment_id"].map(lambda e: res.get(e, {}).get("dsr", np.nan))
    L += ["## Top 10 OOS Models (by deflated Sharpe, multiple-testing adjusted)", "",
          _table(pd.DataFrame([_row(r, stat(r["experiment_id"]))
                               for _, r in b.sort_values("dsr", ascending=False).head(10).iterrows()])), ""]

    def best_of(mask, title):
        s = b[mask]
        if s.empty:
            return [f"## {title}", "", "_no finished experiment in this category_", ""]
        r = s.iloc[0]
        st = stat(r["experiment_id"])
        return [f"## {title}", "", _table(pd.DataFrame([_row(r, st)])),
                f"Long / short: {_side(repo, r['experiment_id'])}. Failed gates: "
                f"{'; '.join(st.get('failed', [])) or 'none'}.", ""]

    fam = b["family"]
    bench = b["entry"].isin(["flat", "buy_hold", "random"])
    L += best_of(~fam.isin(["hybrid", "ensemble", "meta"]) & ~bench, "Best Individual Model")
    L += best_of(fam == "regime", "Best Regime Model")
    L += best_of(fam == "tree", "Best Tree Model")
    L += best_of(fam == "deep", "Best Deep Model")
    L += best_of(fam == "transformer", "Best Transformer")
    L += best_of(fam == "ssm", "Best Mamba/SSM")
    L += best_of(fam == "hybrid", "Best Hybrid")
    L += best_of(fam == "ensemble", "Best Ensemble")
    L += best_of(b["entry"].str.startswith("ob_with"), "Best Order-Book Model")
    L += best_of(fam == "meta", "Best Meta-Label Model")
    # ---------------- order book A/B and ablation
    L += ["## Order Book: with vs without (same window, test folds 2023-10 → 2024-07)", ""]
    ob = b[b["entry"].str.startswith("ob_")]
    L += [_table(pd.DataFrame([{"Entry": r["entry"], "Net": _f(r["net"], usd=True), "Gross bps": _f(r["gross_bps"], 1),
                                "Cost bps": _f(r["cost_bps"], 1), "Trades": int(r["trades"]), "t": _f(r["t_stat"]),
                                "Forced gross/cost": _f(r["forced:gross_bps"] / r["forced:cost_bps"])}
                               for _, r in ob.sort_values("entry").iterrows()])),
          "Only 3 test quarters have order-book history (book_depth_5m starts 2023-01): this comparison has low power.",
          "Top-of-book / microprice / L2 order-flow imbalance cannot be tested historically (collected since "
          "2026-10-04).", ""]
    L += ["## Feature Group Ablation", ""]
    ab = b[b["entry"].str.startswith(("abl_", "loo_"))]
    L += [_table(pd.DataFrame([{"Entry": r["entry"], "Net": _f(r["net"], usd=True), "PF": _f(r["pf"]),
                                "t": _f(r["t_stat"]), "Forced gross bps": _f(r["forced:gross_bps"], 1),
                                "Forced cost bps": _f(r["forced:cost_bps"], 1), "Trades": int(r["trades"])}
                               for _, r in ab.sort_values("entry").iterrows()])),
          "core_deriv has a shorter window (OI / derivatives from 2021-12); compare its folds, not its total.", ""]
    # ---------------- robustness / MC
    top = an.get("top", [])
    rows, mc = [], []
    for e in top:
        r = b.set_index("experiment_id").loc[e] if e in set(b["experiment_id"]) else None
        s = res.get(e, {})
        if r is None:
            continue
        rows.append({"Model": r["entry"], "Net": _f(r["net"], pct=True), "cost×1.5": _f(r.get("cost_x1.5:net"), pct=True),
                     "cost×2": _f(r.get("cost_x2:net"), pct=True), "cost×3": _f(r.get("cost_x3:net"), pct=True),
                     "slip×2": _f(r.get("slip_x2:net"), pct=True), "slip×3": _f(r.get("slip_x3:net"), pct=True),
                     "maker mix": _f(r.get("mixed_maker:net"), pct=True), "θ×0.8": _f(s.get("th_x0.8"), pct=True),
                     "θ×1.2": _f(s.get("th_x1.2"), pct=True), "hold×0.5": _f(s.get("hold_x0.5"), pct=True),
                     "hold×2": _f(s.get("hold_x2"), pct=True), "1-bar delay": _f(s.get("delay_1bar"), pct=True),
                     "stable": s.get("param_stable")})
        mc.append({"Model": r["entry"], "P(profit)": _f(s.get("mc_p_profit"), pct=True),
                   "P(ruin −30%)": _f(s.get("mc_p_ruin"), pct=True), "E[max DD]": _f(s.get("mc_expected_dd"), pct=True),
                   "Worst DD": _f(s.get("mc_worst_dd"), pct=True), "P&L p5": _f(s.get("mc_pnl_p05"), pct=True),
                   "P&L p50": _f(s.get("mc_pnl_p50"), pct=True), "P&L p95": _f(s.get("mc_pnl_p95"), pct=True)})
    L += ["## Robustness Results (top candidates)", "", _table(pd.DataFrame(rows)), ""]
    L += ["## Monte Carlo Results (5,000 runs: missed trades, cost / slippage / return noise, reshuffled order)", "",
          _table(pd.DataFrame(mc)), ""]
    # ---------------- leakage
    line, ok = leakage_tests()
    L += ["## Leakage Audit", "", f"Automated suite `tests/tournament` (run at report time): **{line}** "
          f"({'all passed' if ok else 'FAILURES — see pytest output'}).", "",
          "| Leak type | How it is prevented | Test |", "|---|---|---|",
          "| Future candles / look-ahead | causal rolling features; every value at t equal with data truncated "
          "after t | `test_feature_group_is_causal` (13 groups) |",
          "| Target leakage | label of t = open[t+1] → open[t+1+h]; no feature correlates like a label "
          "| `test_target_is_future_and_aligned`, `test_no_feature_is_a_disguised_label` |",
          "| Overlapping labels | purge 18 bars + 16-bar embargo between train / val / test, purged K-fold "
          "| `test_splits_are_purged_and_embargoed`, `test_purged_kfold_gap` |",
          "| Normalization leakage | scalers / winsor bounds fit on training rows only "
          "| `test_preprocessor_fits_on_train_only` |",
          "| Regime leakage | forward filtering only (no smoothing); HSMM / IMM / Hamilton filters "
          "| `test_model_predictions_are_causal` (HMM, HSMM, SLDS, hybrid) |",
          "| Sequence / batch leakage | windows end at t; TimesNet periods frozen after training "
          "| deep smoke check (28 nets causal) |",
          "| Order-book future leakage | depth joined by snap_time ≤ bar close; metrics with +5 min delay "
          "| `test_feature_group_is_causal[ORDER_BOOK]` |",
          "| Hyper-parameter / model selection | per fold on validation only; held-out guard; stacking / meta "
          "on earlier folds only | `test_guard_*`, ensembles module |",
          "| Test-set contamination | VALID-A/B never loaded; DEV_END guard; forward shadow is the only true OOS "
          "| `test_guard_blocks_held_out_data` |",
          "| Pipeline false positives | random-walk null finds nothing; planted edge found; shuffled labels "
          "lose the edge | `test_null_has_no_significant_edge`, `test_planted_edge_*`, `test_label_shuffle_*` |", "",
          "Bug found and fixed during the build: the classical models first winsorized returns with a std computed "
          "over the whole frame (including test rows). The bounds now come from training rows only.", ""]
    # ---------------- significance
    L += ["## Statistical Significance", "",
          f"- N (experiments) = {n}; PBO (CSCV, 10 blocks, daily DEV-OOS P&L of all experiments) = {_f(an.get('pbo'))}.",
          f"- Hansen SPA p = {_f((an.get('spa') or {}).get('spa_p'), 3)}, White Reality Check p = "
          f"{_f((an.get('spa') or {}).get('rc_p'), 3)} (H0: no experiment beats FLAT).", ""]
    srows = []
    for e in top:
        s = res.get(e, {})
        r = b.set_index("experiment_id").loc[e] if e in set(b["experiment_id"]) else None
        if r is None:
            continue
        srows.append({"Model": r["entry"], "t": _f(r["t_stat"]), "DSR": _f(s.get("dsr")),
                      "Net 95% CI": f"{_f(s.get('net_ci_lo'), pct=True)} … {_f(s.get('net_ci_hi'), pct=True)}",
                      "Sharpe 95% CI": f"{_f(s.get('sharpe_ci_lo'))} … {_f(s.get('sharpe_ci_hi'))}",
                      "P(net ≤ 0)": _f(s.get("p_net_le_0"), pct=True), "Shuffled t": _f(s.get("shuffled_t"))})
    L += [_table(pd.DataFrame(srows)), ""]
    # ---------------- recommendation
    passed = an.get("passed", [])
    L += ["## Final Recommendation", ""]
    if not passed:
        L += ["**NO ROBUST PROFITABLE MODEL FOUND.** No experiment passed every pre-registered gate "
              "(data/experiments/sol15_preregistration.json). No finalist is frozen, nothing goes to forward shadow, "
              "and nothing goes near production. The gates were not relaxed after the results were seen.", ""]
    else:
        L += [f"{len(passed)} candidate(s) passed every DEV gate: "
              + ", ".join(res[e]["entry"] for e in passed)
              + ". Under the one-SE rule the simplest is preferred. Next step: freeze ≤ 3 and run them once in "
                "forward shadow (`alpha.tournament shadow`, lockbox SOL15-FORWARD) for ≥ 150 trades or 8 weeks. "
                "DEV results alone are not enough to trade.", ""]
    # ---------------- rejected
    rej = [x for x in an.get("results", []) if x["status"] == "REJECTED"]
    cnt = pd.Series([g for x in rej for g in x["failed"]]).value_counts()
    L += ["## Models Rejected", "", f"{len(rej)} of {n} experiments were rejected; gated-out / not-run entries are "
          "listed under Models Tested.", "", "## Why Models Were Rejected", "",
          "How often each gate failed:", "", _table(cnt.rename_axis("gate").reset_index(name="experiments")), "",
          "<details><summary>per experiment</summary>", "",
          _table(pd.DataFrame([{"Entry": x["entry"], "Family": x["family"], "Failed gates": "; ".join(x["failed"][:6])
                                + (" …" if len(x["failed"]) > 6 else "")} for x in rej])), "</details>", ""]
    L += ["## Next Research Direction", ""] + _next(b, an) + [""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L))
    return REPORT


def _side(repo: Repo, eid: str) -> str:
    d = repo.df("SELECT bucket, value FROM tournament.performance_metrics WHERE experiment_id = %s AND variant = "
                "'policy' AND scope = 'side' AND metric = 'net'", (eid,))
    if d.empty:
        return "–"
    return ", ".join(f"{r['bucket']} {_f(r['value'], pct=True)}" for _, r in d.iterrows())


def _next(b: pd.DataFrame, an: dict) -> list[str]:
    out = ["- The edge-to-cost ratio is the binding constraint at 15m: round trips cost ~14 bps, so a signal needs "
           "a mean gross move well above that per trade. Longer holds (4h-3d, as in the project's P6 momentum book) "
           "amortize costs; the tournament's evidence argues against pushing SOL-only intraday models further.",
           "- Honest forward data is the only clean test left: start collecting the L2 book / microprice features "
           "(already logged since 2026-10-04) and re-test order-book models once ≥ 6 months exist.",
           "- Maker execution changes the cost side the most; measure real passive fill rates and adverse selection "
           "in paper trading before modelling any strategy that depends on it."]
    if an.get("g1_open"):
        out.insert(0, "- Gate G1 opened: run the full deep / transformer / SSM sweeps (configs stage_4 / 5 / 6).")
    return out
