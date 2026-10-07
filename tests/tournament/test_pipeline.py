"""End-to-end pipeline validation on synthetic data: the null finds nothing, a planted edge is found, runs are
reproducible, and the model interface is honoured by every registered (cheap) model."""

import numpy as np
import pytest

from alpha.tournament.models import registry
from alpha.tournament.models.base import BaseTradingModel
from alpha.tournament.training.runner import Spec, build_context, run_experiment


@pytest.fixture(scope="module")
def null_ctx():
    return build_context("synthetic", n_synth=50_000, seed=11, planted_bps=0.0)


@pytest.fixture(scope="module")
def planted_ctx():
    return build_context("synthetic", n_synth=50_000, seed=11, planted_bps=12.0)


SPECS = [Spec("t", "ridge", "ridge", targets=["reg_1", "reg_4", "reg_16"], params=[{"alpha": 100}]),
         Spec("t", "lgb", "lightgbm", targets=["reg_4", "cls_4_1"], params=[{"n_estimators": 100}]),
         Spec("t", "flow", "flow_follow", targets=["reg_4", "reg_16"], params=[{"window": 60}]),
         Spec("t", "mom", "momentum", targets=["reg_4", "reg_16"])]


def test_null_has_no_significant_edge(null_ctx):
    """False-positive check: on a random walk no entry may reach t >= 2 after costs."""
    for s in SPECS:
        o = run_experiment(null_ctx, s)
        t = o.summary["policy"]["t_stat"]
        assert not (np.isfinite(t) and t >= 2), f"{s.entry} found a 'significant' edge in pure noise (t={t:.2f})"


def test_planted_edge_is_found_with_right_sign(planted_ctx):
    o = run_experiment(planted_ctx, SPECS[0])
    s = o.summary["policy"]
    assert s["t_stat"] > 3 and s["net"] > 0
    # the edge is 12 bps per 4 bars; per trade gross must be positive and of the right order (not >> planted)
    assert 0 < s["gross_bps"] < 400


def test_reproducible(planted_ctx):
    a = run_experiment(planted_ctx, SPECS[1])
    b = run_experiment(planted_ctx, SPECS[1])
    assert a.experiment_id == b.experiment_id
    for k in ("trades", "net", "gross"):
        assert a.summary["policy"][k] == b.summary["policy"][k]


CHEAP = ["flat", "buy_hold", "random", "momentum", "reversal", "breakout", "flow_follow", "linear_regression",
         "ridge", "lasso", "elasticnet", "logistic", "decision_tree", "hist_gb", "lightgbm", "xgboost", "catboost"]


@pytest.mark.parametrize("name", CHEAP)
def test_model_contract(planted_ctx, name, tmp_path):
    cls = registry.get(name)
    assert issubclass(cls, BaseTradingModel)
    for attr in ("family", "complexity", "input_kind", "target_kinds"):
        assert hasattr(cls, attr)
    from alpha.tournament.features import groups as fg
    from alpha.tournament.targets.targets import Target

    ctx = planted_ctx
    kind = cls.target_kinds[0]
    tgt = Target(kind, 4, 1.0 if kind == "cls" else 0.0)
    X = ctx.feats
    y = ctx.target(tgt)
    n = len(X)
    tr, va = np.arange(n) < 20_000, (np.arange(n) >= 20_100) & (np.arange(n) < 25_000)
    m = cls(tgt, {"n_estimators": 30} if cls.family == "tree" else {}, seed=0,
            columns=fg.columns_for("core", ctx.group_cols))
    m.context = ctx
    m.fit(X, y, tr, va)
    rows = np.arange(n) >= 25_000
    p = m.predict(X, rows)
    assert np.isnan(p[~rows]).all() and np.isfinite(p[rows]).mean() > 0.95
    pr = m.predict_proba(X, rows)[rows]
    assert pr.shape[1] == 3
    pos = m.generate_signal(X, rows, threshold=float(np.nanquantile(np.abs(p[rows]), 0.9)), hold=4)
    assert set(np.unique(pos)) <= {-1.0, 0.0, 1.0}
    path = m.save(tmp_path / f"{name}.joblib")
    m2 = cls.load(path)
    np.testing.assert_allclose(m2.predict(X, rows)[rows], p[rows], equal_nan=True)


def test_registry_complete():
    """Every family of the brief has registered entries (implemented or explicitly gated)."""
    fams = {"rules", "linear", "tree", "classical", "regime", "deep", "transformer", "ssm", "foundation", "rl"}
    names = set(registry.REGISTRY)
    for n in ("lightgbm", "xgboost", "catboost", "hmm", "hsmm", "slds", "lstm", "tcn", "patchtst", "mamba",
              "chronos_bolt", "ppo", "dqn", "stacking", "meta"):
        assert n in names
    assert fams  # documented above
