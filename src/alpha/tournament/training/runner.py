"""Walk-forward experiment runner: the same data, features, folds, costs, policy and sizing for every entry.

For each fold:
    1. fit every config of the entry's grid (target x hyper-parameters) on the fold's training rows
    2. score validation; for each config x threshold candidate, backtest validation with costs
    3. pick the (config, threshold) with the highest validation net P&L; if none is positive the model abstains
       (FLAT) for the fold. The 'forced' variant keeps the best config anyway (diagnostic: is there gross edge?)
    4. trade the test fold with the choice
Test folds are concatenated into one DEV-OOS series (2021-07 -> 2024-07); that is what gets ranked.
"""

import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from alpha.tournament.analytics import metrics as mt
from alpha.tournament.analytics.regimes import attach as attach_regimes
from alpha.tournament.analytics.regimes import label as regime_labels
from alpha.tournament.backtesting import engine, policy
from alpha.tournament.data import dataset as dsm
from alpha.tournament.execution.fills import CostConfig, side_costs
from alpha.tournament.features import groups as fg
from alpha.tournament.models import registry
from alpha.tournament.targets.targets import Target, make as make_target, score_scale
from alpha.tournament.training.resources import model_bytes, track
from alpha.tournament.validation.guard import assert_dev
from alpha.tournament.validation.splits import Fold, walk_forward

ART = Path("models/tournament")
MIN_VAL_TRADES = 10
STRESS_COST = (1.25, 1.5, 2.0, 3.0)
STRESS_SLIP = (1.5, 2.0, 3.0)
MIN_TRAIN_DAYS_WITH_GROUP = 180


@dataclass
class Context:
    dataset_id: str
    raw: pd.DataFrame
    feats: pd.DataFrame
    group_cols: dict[str, list[str]]
    cost: CostConfig
    market: engine.Market
    rt_cost: pd.Series
    funding: pd.Series
    folds: list[Fold]
    labels: pd.DataFrame
    kind: str = "real"
    _targets: dict = field(default_factory=dict)

    def target(self, t: Target) -> pd.Series:
        if t.name not in self._targets:
            self._targets[t.name] = make_target(t, self.raw["sol_open"], self.raw["sol_close"], self.rt_cost)
        return self._targets[t.name]

    def group_cols_for(self, feature_set) -> list[str]:
        return fg.columns_for(feature_set, self.group_cols)

    def market_for(self, cost: CostConfig) -> engine.Market:
        return engine.make_market(self.raw["sol_open"], side_costs(self.raw, cost),
                                  self.funding if cost.funding else None)


def build_context(kind: str = "real", conn=None, cost: CostConfig = CostConfig(), n_synth: int = 140_000,
                  seed: int = 0, planted_bps: float = 0.0, symbol: str = dsm.SYMBOL,
                  forward: tuple[pd.Timestamp, pd.Timestamp] | None = None) -> Context:
    """forward = (oos_start, end): the recorded COMPARE window past DEV_END (alpha.tournament.experiments.compare
    is the only caller; it opens the lockbox first). Otherwise DEV only, guarded."""
    if forward is not None:
        oos_start, end = forward
        raw = dsm.load_raw(conn, end=end, allow_forward=True, symbol=symbol)
        funding = dsm.funding_events(conn, end=end, allow_forward=True, symbol=symbol)
        dataset_id = f"{symbol.lower()}15_fwd_{dsm.content_hash(raw)[:12]}"
        conn.execute("""INSERT INTO tournament.datasets (dataset_id, symbol, tf, start_ts, end_ts, n_rows, content_hash,
                        path, coverage) VALUES (%s,%s,'15m',%s,%s,%s,%s,'(not cached)','{}')
                        ON CONFLICT (dataset_id) DO NOTHING""",
                     (dataset_id, symbol, raw.index.min(), end, len(raw), dsm.content_hash(raw)))
        conn.commit()
        feats, group_cols = fg.build_features(raw)
        sc = side_costs(raw, cost)
        market = engine.make_market(raw["sol_open"], sc, funding if cost.funding else None)
        folds = walk_forward(raw.index.min(), oos_start=oos_start, end=end)
        return Context(dataset_id, raw, feats, group_cols, cost, market, pd.Series(2 * sc.total, index=raw.index),
                       funding, folds, regime_labels(raw), "forward")
    if kind == "real":
        ds = dsm.build(conn)
        raw, dataset_id = ds.frame, ds.dataset_id
        funding = dsm.funding_events(conn)
        assert_dev(raw.index, "raw dataset")
    else:
        raw = dsm.synthetic(n_synth, seed=seed, planted_bps=planted_bps)
        funding = dsm.synthetic_funding(raw.index)
        dataset_id = f"synthetic_{kind}_{seed}_{planted_bps:g}_{n_synth}"
    feats, group_cols = fg.build_features(raw)
    sc = side_costs(raw, cost)
    market = engine.make_market(raw["sol_open"], sc, funding if cost.funding else None)
    rt = pd.Series(2 * sc.total, index=raw.index)
    start = raw.index.min()
    if kind == "real":
        folds = walk_forward(start)
    else:  # synthetic: same structure on its own calendar (quarterly folds after 9 months of history)
        oos = (start + pd.Timedelta(days=300)).to_period("Q").start_time.tz_localize("UTC")
        folds = walk_forward(start, oos_start=oos, end=raw.index.max() - pd.Timedelta(days=1))
    return Context(dataset_id, raw, feats, group_cols, cost, market, rt, funding, folds, regime_labels(raw), kind)


@dataclass
class Spec:
    stage: str
    entry: str
    model: str
    feature_set: str | list[str] = "core"
    targets: list[str] = field(default_factory=lambda: ["reg_4"])
    params: list[dict] = field(default_factory=lambda: [{}])
    seed: int = 0
    shuffled: bool = False
    columns: list[str] | None = None
    parent_id: str | None = None
    eval_start: str | None = None          # restrict the OOS window (e.g. order-book A/B on 2023-01+)
    extra: dict = field(default_factory=dict)

    def groups(self) -> list[str]:
        return fg.FEATURE_SETS[self.feature_set] if isinstance(self.feature_set, str) else list(self.feature_set)

    def fs_name(self) -> str:
        return self.feature_set if isinstance(self.feature_set, str) else "+".join(self.feature_set)


@dataclass
class Outcome:
    experiment_id: str
    spec: Spec
    status: str
    reason: str = ""
    oos_index: pd.DatetimeIndex | None = None
    pos: dict[str, np.ndarray] = field(default_factory=dict)      # variant -> positions on the OOS index
    scores: pd.DataFrame | None = None                             # chosen-config test scores (+ proba, fold, h)
    trades: dict[str, pd.DataFrame] = field(default_factory=dict)
    summary: dict[str, dict] = field(default_factory=dict)
    bar_pnl: dict[str, pd.Series] = field(default_factory=dict)
    folds: list[dict] = field(default_factory=list)
    n_configs: int = 0
    seconds: float = 0.0


def git_info() -> tuple[str, str | None]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        diff = subprocess.run(["git", "diff", "HEAD"], capture_output=True, text=True).stdout
        untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "src/alpha/tournament",
                                    "configs/tournament", "sql"], capture_output=True, text=True).stdout.split()
        h = hashlib.sha256(diff.encode())
        for f in sorted(untracked):
            p = Path(f)
            if p.is_file():
                h.update(f.encode() + p.read_bytes())
        dirty = h.hexdigest()[:16] if (diff or untracked) else None
        return commit, dirty
    except Exception:
        return "unknown", None


def code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(Path(__file__).resolve().parents[1].rglob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def grid(spec: Spec) -> list[tuple[Target, dict]]:
    cls = registry.get(spec.model)
    g = []
    for t in spec.targets:
        tt = Target.parse(t)
        if tt.kind not in cls.target_kinds:
            continue
        for p in spec.params:
            g.append((tt, p))
    return g


def experiment_id(spec: Spec, ctx: Context) -> tuple[str, str]:
    d = asdict(spec)
    d.pop("stage")
    payload = json.dumps({"spec": d, "dataset": ctx.dataset_id, "cost": ctx.cost.to_json(),
                          "folds": [(str(f.test_start), str(f.test_end)) for f in ctx.folds]}, sort_keys=True,
                         default=str)
    h = hashlib.sha256(payload.encode()).hexdigest()
    return f"{spec.stage}.{spec.entry}.{h[:10]}", h


def _columns(spec: Spec, ctx: Context) -> list[str]:
    if spec.columns:
        return spec.columns
    return fg.columns_for(spec.groups(), ctx.group_cols)


def _val_net(pos: np.ndarray, m: engine.Market) -> tuple[float, int]:
    r = engine.run(pos, m)
    return float(r.bar_pnl.sum() * 1e4), int(len(r.trades))


def run_experiment(ctx: Context, spec: Spec, repo=None, save_models: bool = False) -> Outcome:
    t_start = time.time()
    eid, chash = experiment_id(spec, ctx)
    cls = registry.get(spec.model)
    g = grid(spec)
    out = Outcome(eid, spec, "running", n_configs=len(g))
    cols = _columns(spec, ctx)
    X = ctx.feats[cols] if cls.input_kind != "members" else ctx.feats
    if cls.input_kind in ("series", "sequence"):  # these read raw-ish columns (returns, closes) too
        X = ctx.feats
    idx = ctx.feats.index
    cov_start = fg.group_coverage_start(ctx.feats, ctx.group_cols, spec.groups())
    eval_start = max(cov_start, pd.Timestamp(spec.eval_start, tz="UTC")) if spec.eval_start else cov_start
    folds = [f for f in ctx.folds
             if f.test_start >= eval_start and f.train_end - max(cov_start, f.train_start)
             >= pd.Timedelta(days=MIN_TRAIN_DAYS_WITH_GROUP)]
    if repo is not None:
        commit, dirty = git_info()
        fs_id = fg.feature_set_id(spec.fs_name(), cols)
        repo.upsert_model(spec.model, cls.family, cls.complexity, cls.input_kind, cls.experimental,
                          cls.description or (cls.__doc__ or "").strip().split("\n")[0])
        mv = f"{spec.model}@{code_hash()[:8]}"
        repo.upsert_model_version(mv, spec.model, code_hash())
        repo.upsert_feature_set(fs_id, spec.fs_name(), spec.groups(), cols, eval_start)
        for _, p in g:
            repo.upsert_hp(hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()[:16], spec.model, p)
        repo.start_experiment(dict(
            experiment_id=eid, stage=spec.stage, entry=spec.entry, model_id=spec.model, model_version_id=mv,
            dataset_id=ctx.dataset_id, feature_set_id=fs_id,
            grid=[{"target": t.name, "params": p} for t, p in g], n_configs=len(g), cost_config=ctx.cost.to_json(),
            seed=spec.seed, git_commit=commit, git_dirty_hash=dirty, config_hash=chash, status="running",
            shuffled_control=spec.shuffled, parent_id=spec.parent_id))
    if not g or not folds:
        out.status, out.reason = "not_run", ("no config matches the model's target kinds" if not g
                                             else f"no fold with enough history for {spec.groups()}")
        if repo is not None:
            repo.finish_experiment(eid, out.status, out.reason)
        return out
    tr_runs, va_runs, wf_runs = [], [], []
    oos_mask = np.zeros(len(idx), bool)
    pos_pol = np.zeros(len(idx))
    pos_forced = np.zeros(len(idx))
    score_rows = []
    fold_of = np.full(len(idx), -1)
    rng = np.random.default_rng(spec.seed)
    failures: list[str] = []
    for f in folds:
        tr, va, te = f.masks(idx)
        tr &= idx >= cov_start
        n_end = int(np.searchsorted(idx, f.test_end))
        Xf = X.iloc[:n_end]
        trf, vaf, tef = tr[:n_end], va[:n_end], te[:n_end]
        m_val = ctx.market.slice(va)
        best = None  # (net, ci, ti, threshold, hold, val_score, model)
        results = []
        for ci, (tgt, params) in enumerate(g):
            y = ctx.target(tgt).iloc[:n_end]
            if spec.shuffled:
                y = y.copy()
                ti_ = np.flatnonzero(trf & y.notna().to_numpy())
                y.iloc[ti_] = rng.permutation(y.iloc[ti_].to_numpy())
            model = cls(tgt, params, seed=spec.seed, columns=None)
            model.context = ctx  # series / sequence models may need raw columns
            try:
                with track() as tf:
                    model.fit(Xf, y, trf, vaf)
                with track() as tp:
                    raw_va = model.predict(Xf, vaf)
                    raw_te = model.predict(Xf, tef)
            except Exception as e:  # one bad config must not kill the experiment; it is recorded
                logger.warning("{} fold {} config {} failed: {!r}", eid, f.k, ci, e)
                failures.append(f"fold {f.k} config {ci}: {e!r}"[:300])
                results.append(None)
                continue
            scale = (np.ones(n_end) if getattr(model, "output_units", "target") == "bps"
                     else score_scale(tgt, ctx.raw["sol_close"]).to_numpy()[:n_end])
            s_va = (raw_va * scale)[vaf]
            s_te = (raw_te * scale)[tef]
            tr_runs.append((eid, f.k, ci, f.train_start, f.train_end, int(trf.sum()), tf["seconds"], tp["seconds"],
                            tf["cpu_percent"], tf["rss_mb"], tf["gpu_mb"], model_bytes(model) if ci == 0 else None))
            ths = policy.candidate_thresholds(s_va, float(np.nanmedian(ctx.rt_cost.to_numpy()[va]))
                                              if tgt.kind == "reg" and model.family not in ("rules",) else None)
            if model.family == "rules" and spec.model in ("buy_hold",):
                ths = [0.5]
            for ti, th in enumerate(ths):
                p = policy.positions(s_va, th, tgt.h)
                net, ntr = _val_net(p, m_val)
                va_runs.append([eid, f.k, ci, ti, f.val_start, f.val_end, net, ntr, False])
                if ntr >= getattr(model, "min_val_trades", MIN_VAL_TRADES) and (best is None or net > best[0]):
                    best = (net, ci, ti, th, tgt.h, len(va_runs) - 1)
            results.append((model, s_te, tgt))
        # choose
        te_pos = np.flatnonzero(te)
        oos_mask |= te
        fold_of[te] = f.k
        if best is None:
            wf_runs.append((eid, f.k, f.test_start, f.test_end, None, None, True, 0, None, None, 0, None, None))
            continue
        net, ci, ti, th, h, vrow = best
        va_runs[vrow][-1] = True
        model, s_te, tgt = results[ci]
        forced = policy.positions(s_te, th, h)
        pos_forced[te_pos] = forced
        abstain = net <= 0
        pos_pol[te_pos] = 0 if abstain else forced
        proba = model.predict_proba(Xf, tef)[tef] if model.target.kind == "cls" else None
        score_rows.append(pd.DataFrame({"fold": f.k, "score": s_te, "target_h": h, "threshold": th,
                                        "p_short": proba[:, 0] if proba is not None else np.nan,
                                        "p_flat": proba[:, 1] if proba is not None else np.nan,
                                        "p_long": proba[:, 2] if proba is not None else np.nan},
                                       index=idx[te_pos]))
        imp = model.feature_importance()
        if imp and repo is not None:
            repo.importance(eid, f.k, imp, "gain" if model.family == "tree" else "coef")
        if save_models:
            pth = model.save(ART / eid / f"fold{f.k:02d}.joblib")
            if repo is not None:
                repo.artifact(eid, "model", pth)
        mf = ctx.market.slice(te)
        rf = engine.run(forced, mf)
        rp = engine.run(pos_pol[te_pos], mf)
        wf_runs.append((eid, f.k, f.test_start, f.test_end, ci, th, bool(abstain), len(rp.trades),
                        float(rp.trades["gross_bps"].sum()) if len(rp.trades) else 0.0,
                        float(rp.bar_pnl.sum() * 1e4), len(rf.trades),
                        float(rf.trades["gross_bps"].sum()) if len(rf.trades) else 0.0, float(rf.bar_pnl.sum() * 1e4)))
        out.folds.append({"fold": f.k, "config": ci, "target": tgt.name, "params": g[ci][1], "threshold": th,
                          "val_net_bps": net, "abstained": bool(abstain)})
    # ---- OOS evaluation ----
    oos_idx = evaluate(ctx, out, pos_pol, pos_forced, oos_mask, fold_of, cls.complexity,
                       pd.concat(score_rows) if score_rows else None)
    n_fits = len(folds) * len(g)
    out.status = "failed" if len(failures) == n_fits else "done"
    if failures and all("NotRunnable" in f for f in failures):
        out.status = "not_run"
    out.reason = (f"{len(failures)}/{n_fits} fits failed; first: {failures[0]}" if failures else "")
    out.seconds = time.time() - t_start
    if repo is not None:
        _persist(repo, out, va_runs, tr_runs, wf_runs, oos_idx)
    return out


def evaluate(ctx: Context, out: Outcome, pos_pol: np.ndarray, pos_forced: np.ndarray, oos_mask: np.ndarray,
             fold_of: np.ndarray, complexity: int, scores: pd.DataFrame | None) -> pd.DatetimeIndex:
    """Backtest the concatenated OOS positions under every cost variant and fill out.summary / trades."""
    eid = out.experiment_id
    idx = ctx.feats.index
    oos_idx = idx[oos_mask]
    out.oos_index = oos_idx
    m_oos = ctx.market.slice(oos_mask)
    out.pos = {"policy": pos_pol[oos_mask], "forced": pos_forced[oos_mask]}
    out.scores = scores
    fold_series = pd.Series(fold_of[oos_mask], index=oos_idx)
    variants = {"policy": (out.pos["policy"], m_oos), "forced": (out.pos["forced"], m_oos)}
    for k in STRESS_COST:
        variants[f"cost_x{k:g}"] = (out.pos["policy"], ctx.market_for(ctx.cost.with_(fee_mult=k, slip_mult=k))
                                    .slice(oos_mask))
    for k in STRESS_SLIP:
        variants[f"slip_x{k:g}"] = (out.pos["policy"], ctx.market_for(ctx.cost.with_(slip_mult=k)).slice(oos_mask))
    variants["mixed_maker"] = (out.pos["policy"], ctx.market_for(ctx.cost.with_(execution="mixed_maker"))
                               .slice(oos_mask))
    oos_days = max(1, (oos_idx.max() - oos_idx.min()).days) if len(oos_idx) else 1
    for v, (p, m) in variants.items():
        r = engine.run(p, m)
        t = r.trades
        if len(t):
            t = attach_regimes(t, ctx.labels)
            t["fold"] = fold_series.reindex(pd.DatetimeIndex(t["entry_ts"]) - pd.Timedelta(minutes=15)).to_numpy()
        c = mt.card(f"{eid}:{v}", t, complexity)
        out.summary[v] = mt.summary_row(c, t, oos_days)
        if v in ("policy", "forced"):
            out.trades[v] = t
            out.bar_pnl[v] = r.bar_pnl
    return oos_idx


def _persist(repo, out: Outcome, va_runs, tr_runs, wf_runs, oos_idx):
    eid = out.experiment_id
    repo.training_runs(tr_runs)
    repo.validation_runs([tuple(r) for r in va_runs])
    repo.walk_forward_runs(wf_runs)
    s0, s1 = (oos_idx.min(), oos_idx.max()) if len(oos_idx) else (None, None)
    for v, s in out.summary.items():
        repo.oos_run(eid, v, s0, s1, s)
        repo.metrics(eid, v, "all", s)
    for v in ("policy", "forced"):
        t = out.trades.get(v)
        if t is None or t.empty:
            continue
        repo.trades(eid, v, t)
        for scope, tab in mt.all_breakdowns(t).items():
            repo.metrics(eid, v, scope, tab)
        repo.signals(eid, v, oos_idx, out.pos[v])
        daily = out.bar_pnl[v].groupby(out.bar_pnl[v].index.floor("D")).sum()
        repo.metrics(eid, v, "daily", pd.DataFrame({"net": daily}, index=daily.index.strftime("%Y-%m-%d")))
    if out.scores is not None:
        sc = out.scores
        for k, g in sc.groupby("fold"):
            repo.predictions(eid, int(k), g.index, g["score"].to_numpy(),
                             g[["p_short", "p_flat", "p_long"]].to_numpy(), int(g["target_h"].iloc[0]))
    if out.folds:
        ch = pd.DataFrame(out.folds).set_index("fold")
        ch = ch[[c for c in ch.columns if c not in ("params", "target")]].apply(pd.to_numeric, errors="coerce")
        repo.metrics(eid, "policy", "choice", ch.astype(float))
    repo.finish_experiment(eid, out.status, out.reason, s0, s1)
    repo.registry(eid, "EXPERIMENTAL", "walk-forward run complete")
    repo.commit()
