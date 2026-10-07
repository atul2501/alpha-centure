"""Automated leakage tests for the SOL 15m tournament pipeline (synthetic data; no database needed)."""

import numpy as np
import pandas as pd
import pytest

from alpha.research.splits import DEV_END
from alpha.tournament.backtesting import engine, policy
from alpha.tournament.data.dataset import synthetic, synthetic_funding
from alpha.tournament.execution.fills import CostConfig, side_costs
from alpha.tournament.features import groups as fg
from alpha.tournament.models import registry
from alpha.tournament.models.base import Preprocessor
from alpha.tournament.targets.targets import Target, fwd_bps, make
from alpha.tournament.validation.guard import HeldOutAccess, assert_dev, check_end
from alpha.tournament.validation.splits import GAP, purged_kfold, walk_forward

N = 12_000


@pytest.fixture(scope="module")
def raw():
    return synthetic(N, seed=3, planted_bps=10.0)


@pytest.fixture(scope="module")
def feats(raw):
    return fg.build_features(raw)


@pytest.mark.parametrize("group", list(fg.GROUPS))
def test_feature_group_is_causal(raw, group):
    """Every feature at bar t is unchanged when all bars after t are removed (future-candle / look-ahead leak)."""
    full = fg.GROUPS[group](raw)
    for t in (3000, 7777, N - 5):
        cut = fg.GROUPS[group](raw.iloc[: t + 1])
        a, b = full.iloc[t], cut.iloc[t]
        pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-7, atol=1e-10, obj=f"{group} @ {t}")


def test_target_is_future_and_aligned(raw):
    o = raw["sol_open"]
    f = fwd_bps(o, 4)
    t = 500
    assert np.isclose(f.iloc[t], 1e4 * np.log(o.iloc[t + 5] / o.iloc[t + 1]))
    assert f.iloc[-5:].isna().all(), "labels past the data end must be missing, not filled"


def test_no_feature_is_a_disguised_label(raw, feats):
    """Alignment check: no feature may correlate with the forward return more than with anything plausible.
    The planted edge is real (|corr| ~ 0.1); a leaked label would be ~1."""
    X, _ = feats
    f = fwd_bps(raw["sol_open"], 1)
    c = X.corrwith(f).abs().max()
    assert c < 0.5, f"a feature correlates {c:.2f} with the next-bar return: leak"
    leaked = X.assign(leak=f)
    assert leaked.corrwith(f).abs().max() > 0.99  # the check itself works


def test_splits_are_purged_and_embargoed():
    folds = walk_forward(pd.Timestamp("2020-09-14", tz="UTC"))
    assert len(folds) == 12 and folds[-1].test_end == DEV_END
    for f in folds:
        assert f.train_end + GAP <= f.val_start + pd.Timedelta(seconds=1)
        assert f.val_end + GAP <= f.test_start + pd.Timedelta(seconds=1)
        # the label of the last training row (t + 1 + 16 bars) ends before validation starts
        assert f.train_end + 17 * pd.Timedelta(minutes=15) < f.val_start
    for a, b in zip(folds, folds[1:]):
        assert a.test_end == b.test_start


def test_purged_kfold_gap():
    idx = pd.date_range("2022-01-01", periods=5000, freq="15min", tz="UTC")
    for tr, va in purged_kfold(idx, 5):
        lo, hi = idx[va].min(), idx[va].max()
        near = (idx[tr] > lo - GAP) & (idx[tr] < hi + GAP)
        assert not near.any()


def test_guard_blocks_held_out_data():
    with pytest.raises(HeldOutAccess):
        check_end(pd.Timestamp("2025-01-01", tz="UTC"))
    with pytest.raises(HeldOutAccess):
        assert_dev(pd.date_range(DEV_END - pd.Timedelta(hours=1), periods=8, freq="15min", tz="UTC"))
    assert check_end(None) == DEV_END


def test_guard_loader_never_returns_held_out_rows(monkeypatch):
    from alpha.tournament.data import dataset as dsm

    with pytest.raises(HeldOutAccess):
        dsm.load_raw(conn=None, end=pd.Timestamp("2025-10-02", tz="UTC"))


def test_preprocessor_fits_on_train_only(feats):
    X, gc = feats
    cols = fg.columns_for("core", gc)
    tr = X.index < X.index[6000]
    p = Preprocessor().fit(X.loc[tr, cols])
    assert p.fit_range[1] < X.index[6000]
    # changing held-out rows must not change the fitted statistics
    X2 = X.copy()
    X2.loc[~tr, cols] = 1e6
    p2 = Preprocessor().fit(X2.loc[tr, cols])
    np.testing.assert_allclose(p.mu, p2.mu)


CAUSAL_MODELS = [("ridge", {}), ("lightgbm", {"n_estimators": 50}), ("momentum", {}), ("ar", {}), ("kalman", {}),
                 ("garch", {}), ("gaussian_hmm", {"K": 2}), ("hsmm", {"n_fit": 3000, "D": 48}), ("slds", {}),
                 ("hybrid", {"regime": "hmm", "predictor": "ridge", "feature_set": "core"})]


@pytest.mark.parametrize("name,params", CAUSAL_MODELS)
def test_model_predictions_are_causal(raw, feats, name, params):
    """A prediction at bar t (from a model fit on earlier rows) is identical whether or not rows after t are
    present: filters, sequence windows and regime probabilities never look ahead."""
    from alpha.tournament.training.runner import build_context

    ctx = build_context("synthetic", n_synth=N, seed=3, planted_bps=10.0)
    X = ctx.feats
    tgt = Target("reg", 4)
    y = ctx.target(tgt)
    n = len(X)
    train = np.arange(n) < 6000
    val = (np.arange(n) >= 6100) & (np.arange(n) < 7000)
    cls = registry.get(name)
    m = cls(tgt, params, seed=0, columns=None if cls.input_kind != "tabular" else fg.columns_for("core", ctx.group_cols))
    m.context = ctx
    m.fit(X, y, train, val)
    full = m.predict(X, np.arange(n) >= 7000)
    for t in (7200, 9000):
        Xc = X.iloc[: t + 1]
        rows = np.arange(t + 1) >= 7000
        cut = m.predict(Xc, rows)
        assert np.isclose(full[t], cut[t], rtol=1e-4, atol=1e-6, equal_nan=True), f"{name} looks ahead at {t}"


def test_label_shuffle_breaks_edge():
    """The shuffled-label control on a planted edge must lose the edge (the control is meaningful)."""
    from alpha.tournament.training.runner import Spec, build_context, run_experiment

    ctx = build_context("synthetic", n_synth=40_000, seed=1, planted_bps=12.0)
    real = run_experiment(ctx, Spec("t", "r", "ridge", targets=["reg_4"], params=[{"alpha": 100}]))
    shuf = run_experiment(ctx, Spec("t", "s", "ridge", targets=["reg_4"], params=[{"alpha": 100}], shuffled=True))
    assert real.summary["policy"]["t_stat"] > 5
    assert shuf.summary["forced"]["net"] < real.summary["forced"]["net"] / 3


def test_engine_costs_and_flip():
    idx = pd.date_range("2022-01-01", periods=10, freq="15min", tz="UTC")
    o = pd.Series(100 * np.exp(np.arange(10) * 1e-3), index=idx)
    raw = pd.DataFrame({"sol_open": o, "sol_close": o, "bid_1": 1e9, "ask_1": 1e9})
    sc = side_costs(raw, CostConfig())
    m = engine.make_market(o, sc, None)
    pos = np.array([1, 1, -1, -1, 0, 0, 0, 0, 0, 0], float)
    r = engine.run(pos, m)
    assert len(r.trades) == 2
    t0, t1 = r.trades.iloc[0], r.trades.iloc[1]
    assert np.isclose(t0["gross_bps"], 20.0, atol=1e-6)  # two bars of +10 bps
    assert np.isclose(t1["gross_bps"], -20.0, atol=1e-6)
    assert np.isclose(t0["fee_bps"], 2 * 5.0) and np.isclose(t1["fee_bps"], 2 * 5.0)
    assert np.isclose(r.bar_pnl.sum() * 1e4, t0["net_bps"] + t1["net_bps"])


def test_engine_funding_sign():
    idx = pd.date_range("2022-01-01", periods=40, freq="15min", tz="UTC")
    o = pd.Series(100.0, index=idx)
    raw = pd.DataFrame({"sol_open": o, "sol_close": o, "bid_1": 1e9, "ask_1": 1e9})
    f = pd.Series([0.001], index=[idx[0] + pd.Timedelta(hours=4)])  # 10 bps, settled at 04:00
    m = engine.make_market(o, side_costs(raw, CostConfig()), f)
    long = engine.run(np.ones(40), m).trades
    short = engine.run(-np.ones(40), m).trades
    assert np.isclose(long["funding_bps"].sum(), 10.0) and np.isclose(short["funding_bps"].sum(), -10.0)
    flat_before = np.zeros(40)
    flat_before[16:] = 1  # decided at 04:00 close -> enters after the settlement
    assert np.isclose(engine.run(flat_before, m).trades["funding_bps"].sum(), 0.0)


def test_policy_hold_and_flip():
    s = np.array([0, 2, 0, 0, 0, -2, 0, 0, 0, 0], float)
    p = policy.positions(s, 1.0, 3)
    np.testing.assert_array_equal(p, [0, 1, 1, 1, 0, -1, -1, -1, 0, 0])


def test_stress_costs_scale():
    idx = pd.date_range("2022-01-01", periods=200, freq="15min", tz="UTC")
    o = pd.Series(100 * np.exp(np.cumsum(np.full(200, 1e-4))), index=idx)
    raw = pd.DataFrame({"sol_open": o, "sol_close": o, "bid_1": 2e6, "ask_1": 2e6})
    a = side_costs(raw, CostConfig())
    b = side_costs(raw, CostConfig().with_(fee_mult=2, slip_mult=2))
    np.testing.assert_allclose(b.total, 2 * a.total)


def test_bad_depth_snapshot_does_not_explode_cost():
    idx = pd.date_range("2022-01-01", periods=1000, freq="15min", tz="UTC")
    o = pd.Series(100.0, index=idx)
    d = np.full(1000, 2e6)
    d[500] = 5.0  # a corrupt snapshot
    raw = pd.DataFrame({"sol_open": o, "sol_close": o, "bid_1": d, "ask_1": d})
    c = side_costs(raw, CostConfig())
    assert c.total.max() < 2 * np.median(c.total)
