"""Tournament analysis: significance across ALL experiments, robustness of the top-K, the pre-registered gates
(data/experiments/sol15_preregistration.json), gate G1, and registry statuses.

    uv run python -m alpha.tournament analyze [--top 10]

Nothing here changes a model or its OOS series; it only reads stored output (plus shuffled-label control runs
of the top candidates, which are recorded as separate experiments with shuffled_control = true).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from alpha.tournament.analytics import montecarlo, robustness, significance
from alpha.tournament.database.repo import Repo, clean_json

OUT = Path("data/experiments/sol15_analysis.json")
CHEAP_FAMILIES = ("rules", "linear", "classical", "tree", "regime", "hybrid")
EXCLUDE_STAGES = ("smoke", "robust", "compare")


def load_board(repo: Repo) -> pd.DataFrame:
    d = repo.df("""SELECT e.experiment_id, e.stage, e.entry, e.model_id, m.family, m.complexity, m.experimental,
                          e.n_configs, e.feature_set_id, o.variant, o.summary
                   FROM tournament.experiments e JOIN tournament.models m USING (model_id)
                   JOIN tournament.oos_runs o USING (experiment_id)
                   WHERE e.status = 'done' AND NOT e.shuffled_control""")
    d = d[~d["stage"].str.startswith(EXCLUDE_STAGES)]
    if d.empty:
        return d
    rows = []
    for eid, g in d.groupby("experiment_id"):
        r = g.iloc[0][["experiment_id", "stage", "entry", "model_id", "family", "complexity", "experimental",
                       "n_configs", "feature_set_id"]].to_dict()
        for _, v in g.iterrows():
            s = v["summary"]
            pre = "" if v["variant"] == "policy" else f"{v['variant']}:"
            for k, val in s.items():
                r[pre + k] = val
        rows.append(r)
    b = pd.DataFrame(rows)
    for c in b.columns:
        if c not in ("experiment_id", "stage", "entry", "model_id", "family", "feature_set_id", "experimental"):
            b[c] = pd.to_numeric(b[c], errors="coerce")
    return b.sort_values("net", ascending=False).reset_index(drop=True)


def daily_matrix(repo: Repo, eids: list[str], variant: str = "policy") -> pd.DataFrame:
    d = repo.df("SELECT experiment_id, bucket, value FROM tournament.performance_metrics WHERE scope = 'daily' "
                "AND variant = %s AND metric = 'net' AND experiment_id = ANY(%s)", (variant, eids))
    if d.empty:
        return pd.DataFrame()
    m = d.pivot(index="bucket", columns="experiment_id", values="value")
    m.index = pd.to_datetime(m.index)
    return m.sort_index().fillna(0.0)


def gates(r: pd.Series, n_trials: int, extra: dict) -> list[str]:
    """Failed pre-registered gates (empty = passes). NaN inputs count as failures."""
    f = []
    g = lambda k: r.get(k, np.nan)
    cost_all = g("cost_bps") + max(g("funding_bps") if np.isfinite(g("funding_bps")) else 0.0, 0.0)
    checks = {
        "trades >= 300": g("trades") >= 300,
        "net > 0": g("net") > 0, "expectancy > 0": g("expectancy") > 0, "PF > 1": g("pf") > 1,
        "gross >= 1.5x cost": g("gross_bps") >= 1.5 * cost_all,
        "t >= 2": g("t_stat") >= 2,
        "net > 0 at cost x1.5": g("cost_x1.5:net") > 0, "net > 0 at cost x2": g("cost_x2:net") > 0,
        "net > 0 at slippage x2": g("slip_x2:net") > 0,
        ">= 60% folds positive": g("folds_positive") >= 0.6,
        "no month > 25% of profit": g("max_month_share") <= 0.25,
        ">= 60% regime buckets positive": g("regimes_positive") >= 0.6,
        "max DD <= 30%": g("max_dd") <= 0.30, "Calmar >= 0.5": g("calmar") >= 0.5,
        f"DSR >= 0.95 (N={n_trials})": extra.get("dsr", np.nan) >= 0.95,
        "PBO <= 0.2": extra.get("pbo", np.nan) <= 0.2,
        "SPA p < 0.05": extra.get("spa_p", np.nan) < 0.05,
        "shuffled control |t| < 2 and real beats it by 2": (abs(extra.get("shuffled_t", np.nan)) < 2
                                                            and g("t_stat") - extra.get("shuffled_t", np.nan) >= 2),
        "parameter stable": extra.get("param_stable") is True,
    }
    for k, ok in checks.items():
        if not bool(ok):
            f.append(k)
    return f


def shuffled_control(conn, eid: str) -> float:
    """Run (or reuse) the shuffled-label control of an experiment; returns its policy t-stat."""
    from alpha.tournament.experiments.matrix import context
    from alpha.tournament.training import runner

    spec_p = runner.ART / eid / "spec.json"
    if not spec_p.exists():
        return np.nan
    d = json.loads(spec_p.read_text())
    sd = d["spec"]
    from alpha.tournament.models import registry

    if registry.get(sd["model"]).input_kind == "members":
        return np.nan  # ensembles / meta have no single training set to shuffle
    sd.update(stage="control", shuffled=True, parent_id=eid)
    spec = runner.Spec(**sd)
    repo = Repo(conn)
    ctx = context(d["dataset_kind"], conn)
    ceid, _ = runner.experiment_id(spec, ctx)
    r = repo.conn.execute("SELECT o.summary FROM tournament.oos_runs o JOIN tournament.experiments e "
                          "USING (experiment_id) WHERE experiment_id = %s AND variant = 'policy' "
                          "AND e.status = 'done'", (ceid,)).fetchone()
    if r is None:
        logger.info("shuffled control for {}", eid)
        out = runner.run_experiment(ctx, spec, repo)
        t = out.summary.get("policy", {}).get("t_stat", np.nan)
        ft = out.summary.get("forced", {}).get("t_stat", np.nan)
    else:
        t = r[0].get("t_stat")
        ft = np.nan
    t = np.nan if t is None else float(t)
    # a control that abstains everywhere has no t; its forced variant is the meaningful comparison
    if not np.isfinite(t):
        r2 = repo.conn.execute("SELECT summary FROM tournament.oos_runs WHERE experiment_id = %s AND variant = 'forced'",
                               (ceid,)).fetchone()
        t = float(r2[0].get("t_stat") or 0.0) if r2 else 0.0
    return t


def robustness_for(repo: Repo, ctx, eid: str) -> dict:
    pred = repo.oos_predictions(eid)
    if pred.empty:
        return {}
    wf = repo.df("SELECT fold, threshold, abstained FROM tournament.walk_forward_runs WHERE experiment_id = %s", (eid,))
    pred = pred.join(wf.set_index("fold")["threshold"], on="fold").dropna(subset=["threshold"])
    abst = dict(zip(wf["fold"].astype(int), wf["abstained"]))
    idx = ctx.feats.index
    oos = np.zeros(len(idx), bool)
    for f in ctx.folds:
        oos |= f.masks(idx)[2]
    return robustness.perturb(pred, ctx.market.slice(oos), abst)


def analyze(conn, top: int = 10) -> dict:
    from alpha.tournament.experiments.matrix import context

    repo = Repo(conn)
    b = load_board(repo)
    if b.empty:
        logger.warning("no finished experiments")
        return {}
    n = len(b)
    M = daily_matrix(repo, b["experiment_id"].tolist())
    pbo_all = significance.pbo(M) if M.shape[1] >= 2 else np.nan
    spa_all = significance.spa(M) if M.shape[1] >= 1 else {}
    logger.info("{} experiments; PBO {:.2f}; SPA p {}", n, pbo_all, spa_all.get("spa_p"))
    stats = {}
    for _, r in b.iterrows():
        e = r["experiment_id"]
        d = M[e] if e in M else pd.Series(dtype=float)
        nz = d[d.index >= d[d != 0].index.min()] if (d != 0).any() else d
        stats[e] = {"dsr": significance.dsr(nz, n) if len(nz) else np.nan, **significance.bootstrap_ci(nz)}
    ctx = context("real", conn)
    top_ids = b.head(top)["experiment_id"].tolist()
    # G1 candidates: cheap families ranked by forced-variant gross edge / cost
    b["edge_to_cost_forced"] = b["forced:gross_bps"] / b["forced:cost_bps"]
    cheap = b[b["family"].isin(CHEAP_FAMILIES)].sort_values("edge_to_cost_forced", ascending=False)
    g1_ids = cheap.head(3)["experiment_id"].tolist()
    for e in dict.fromkeys(top_ids + g1_ids):
        repo.registry(e, "VALIDATING", "top candidate: robustness / controls / Monte Carlo")
        st = stats[e]
        st["shuffled_t"] = shuffled_control(conn, e)
        st.update(robustness_for(repo, ctx, e))
        st.update(montecarlo.simulate(repo.experiment_trades(e)))
        repo.metrics(e, "policy", "stat", {k: v for k, v in st.items() if not isinstance(v, (dict, list, str))})
        repo.commit()
    for e in stats:
        stats[e].update(pbo=pbo_all, spa_p=spa_all.get("spa_p"), rc_p=spa_all.get("rc_p"))
    g1 = []
    for e in g1_ids:
        r = b.set_index("experiment_id").loc[e]
        st = stats[e]
        ok = (r["edge_to_cost_forced"] >= 1.0 and abs(st.get("shuffled_t", np.nan)) < 2
              and r.get("forced:t_stat", np.nan) - st.get("shuffled_t", np.nan) >= 2)
        g1.append({"experiment_id": e, "entry": r["entry"], "edge_to_cost_forced": r["edge_to_cost_forced"],
                   "forced_t": r.get("forced:t_stat"), "shuffled_t": st.get("shuffled_t"), "opens_g1": bool(ok)})
    g1_open = any(x["opens_g1"] for x in g1)
    results = []
    for _, r in b.iterrows():
        e = r["experiment_id"]
        failed = gates(r, n, stats[e])
        if e not in top_ids and e not in g1_ids:
            failed = failed + ["not in top-K robustness review"] if failed else ["not in top-K robustness review"]
        status = "PASSED" if not failed else "REJECTED"
        repo.registry(e, status, "; ".join(failed)[:1000])
        results.append({"experiment_id": e, "entry": r["entry"], "family": r["family"], "status": status,
                        "failed": failed, **{k: v for k, v in stats[e].items() if not isinstance(v, (dict, list))}})
    repo.commit()
    passed = [x for x in results if x["status"] == "PASSED"]
    verdict = "NO ROBUST PROFITABLE MODEL FOUND" if not passed else "CANDIDATES PASSED DEV GATES"
    res = {"n_experiments": n, "pbo": pbo_all, "spa": spa_all, "g1": g1, "g1_open": g1_open, "verdict": verdict,
           "passed": [x["experiment_id"] for x in passed], "top": top_ids, "results": results}
    OUT.write_text(json.dumps(clean_json(res), indent=1, default=str))
    logger.info("verdict: {} | G1 open: {}", verdict, g1_open)
    return res
