"""Configuration-driven tournament: a YAML file lists entries; each becomes one walk-forward experiment.

    stage: stage_2
    dataset: real                      # real | synthetic_null | synthetic_planted
    defaults: {feature_set: core, targets: [reg_1, reg_4, reg_16], seed: 0}
    entries:
      - {entry: ridge, model: ridge, params: [{alpha: 10}, {alpha: 1000}]}

Runs are resumable (an experiment that is already 'done' is skipped unless --force) and can run in parallel
worker processes (each builds the shared context once from the cached dataset).
"""

import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import psycopg
import yaml
from loguru import logger

from alpha.config import get_settings
from alpha.tournament.database.repo import Repo
from alpha.tournament.training import runner
from alpha.tournament.training.resources import wait_for_ram

_CTX: dict = {}


def load_specs(path: str) -> tuple[dict, list[runner.Spec]]:
    cfg = yaml.safe_load(Path(path).read_text())
    d = cfg.get("defaults", {})
    specs = []
    for e in cfg["entries"]:
        e = {**d, **e}
        specs.append(runner.Spec(stage=cfg["stage"], entry=e.pop("entry"), model=e.pop("model"),
                                 feature_set=e.pop("feature_set", "core"), targets=e.pop("targets", ["reg_4"]),
                                 params=e.pop("params", [{}]), seed=e.pop("seed", 0),
                                 shuffled=e.pop("shuffled", False), columns=e.pop("columns", None),
                                 parent_id=e.pop("parent_id", None), eval_start=e.pop("eval_start", None),
                                 extra=e))
    return cfg, specs


def context(kind: str, conn=None) -> runner.Context:
    if kind not in _CTX:
        if kind == "real":
            _CTX[kind] = runner.build_context("real", conn)
        elif kind == "synthetic_null":
            _CTX[kind] = runner.build_context("synthetic", planted_bps=0.0)
        elif kind == "synthetic_planted":
            _CTX[kind] = runner.build_context("synthetic", planted_bps=12.0)
        else:
            raise ValueError(kind)
    return _CTX[kind]


def _worker(kind: str, spec: runner.Spec, force: bool) -> tuple[str, str, float, dict]:
    import warnings

    warnings.filterwarnings("ignore")
    wait_for_ram()
    with psycopg.connect(get_settings().database_url) as conn:
        ctx = context(kind, conn)
        repo = Repo(conn)
        eid, _ = runner.experiment_id(spec, ctx)
        if not force and repo.status(eid) == "done":
            return eid, "skipped", 0.0, {}
        try:
            out = run_spec(ctx, spec, repo)
        except Exception as e:  # recorded, the queue goes on
            logger.exception("{} failed", eid)
            conn.rollback()
            repo.finish_experiment(eid, "failed", repr(e))
            return eid, "failed", 0.0, {}
        s = out.summary.get("policy", {})
        return eid, out.status, out.seconds, {k: s.get(k) for k in ("trades", "net", "pf", "sharpe", "t_stat")}


def run_spec(ctx, spec: runner.Spec, repo: Repo | None):
    """Dispatch: ordinary models go through the walk-forward runner; ensembles / meta-labels are built from
    their members' stored out-of-fold output."""
    from alpha.tournament.models import registry

    cls = registry.get(spec.model)
    if spec.extra.get("requires_g1") and not g1_open():
        return gated_out(ctx, spec, repo, "gate G1 closed: no stage 2-3 model showed gross edge >= cost that also "
                                          "beat its shuffled control (data/experiments/sol15_analysis.json)")
    if cls.input_kind == "members":
        from alpha.tournament.models.ensemble.ensembles import run_members_entry

        out = run_members_entry(ctx, spec, repo)
    else:
        out = runner.run_experiment(ctx, spec, repo, save_models=spec.extra.get("save_models", False))
    if repo is not None:
        p = runner.ART / out.experiment_id / "spec.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"spec": asdict(spec), "dataset_kind": ctx.kind}, default=str, indent=1))
        repo.artifact(out.experiment_id, "config", p)
        repo.commit()
    return out


def g1_open() -> bool:
    p = Path("data/experiments/sol15_analysis.json")
    if not p.exists():
        raise RuntimeError("run `alpha.tournament analyze` after stage 3 first: gate G1 decides the full sweeps")
    return bool(json.loads(p.read_text()).get("g1_open"))


def gated_out(ctx, spec, repo, reason: str):
    from alpha.tournament.models import registry

    eid, chash = runner.experiment_id(spec, ctx)
    out = runner.Outcome(eid, spec, "gated_out", reason)
    if repo is not None:
        cls = registry.get(spec.model)
        commit, dirty = runner.git_info()
        repo.upsert_model(spec.model, cls.family, cls.complexity, cls.input_kind, cls.experimental,
                          cls.description or (cls.__doc__ or "").strip().split("\n")[0])
        repo.start_experiment(dict(experiment_id=eid, stage=spec.stage, entry=spec.entry, model_id=spec.model,
                                   dataset_id=ctx.dataset_id, grid=[{"targets": spec.targets, "params": spec.params}],
                                   n_configs=0, cost_config=ctx.cost.to_json(), seed=spec.seed, git_commit=commit,
                                   git_dirty_hash=dirty, config_hash=chash, status="gated_out", status_reason=reason,
                                   shuffled_control=False))
    return out


def run_config(conn, path: str, workers: int = 1, force: bool = False, only: list[str] | None = None) -> None:
    cfg, specs = load_specs(path)
    if only:
        specs = [s for s in specs if s.entry in only]
    kind = cfg.get("dataset", "real")
    logger.info("{}: {} entries on {} ({} workers)", cfg["stage"], len(specs), kind, workers)
    if workers <= 1:
        for s in specs:
            eid, st, sec, m = _worker(kind, s, force)
            logger.info("{} {} {:.0f}s {}", eid, st, sec, _fmt(m))
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_worker, kind, s, force): s for s in specs}
        for f in as_completed(futs):
            eid, st, sec, m = f.result()
            logger.info("{} {} {:.0f}s {}", eid, st, sec, _fmt(m))


def _fmt(m: dict) -> str:
    return " ".join(f"{k}={v:.3f}" if isinstance(v, float) and math.isfinite(v) else f"{k}={v}" for k, v in m.items())


def reproduce(conn, experiment_id: str) -> bool:
    """Re-run an experiment from its stored spec and compare the stored OOS policy metrics."""
    repo = Repo(conn)
    p = runner.ART / experiment_id / "spec.json"
    d = json.loads(p.read_text())
    spec = runner.Spec(**d["spec"])
    ctx = context(d["dataset_kind"], conn)
    stored = repo.df("SELECT variant, summary FROM tournament.oos_runs WHERE experiment_id = %s", (experiment_id,))
    out = run_spec(ctx, spec, None)
    ok = out.experiment_id == experiment_id
    for _, r in stored.iterrows():
        a, b = r["summary"], out.summary.get(r["variant"], {})
        for k in ("trades", "net", "gross", "pf"):
            x, y = a.get(k), b.get(k)
            same = (x is None and (y is None or not np.isfinite(y))) or (
                x is not None and y is not None and np.isclose(float(x), float(y), rtol=1e-6, atol=1e-9))
            if not same:
                ok = False
                print(f"MISMATCH {r['variant']} {k}: stored {x} vs rerun {y}")
    print(f"{experiment_id}: {'REPRODUCED' if ok else 'NOT reproduced'}")
    return ok
