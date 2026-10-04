import numpy as np
import pandas as pd
import pytest

from alpha.regime.hmm import FEATURES, RegimeModel, _name_states, regime_features
from tests.conftest import make_ohlcv


def regime_path(seed=0):
    """Price path cycling through calm, trending and violent phases."""
    rng = np.random.default_rng(seed)
    pieces = []
    for drift, vol, n in [(0, 0.004, 300), (0.004, 0.01, 250), (0, 0.003, 300), (-0.005, 0.012, 250),
                          (0, 0.04, 120), (0, 0.004, 300), (0.003, 0.01, 250)]:
        pieces.append(rng.normal(drift, vol, n))
    r = np.concatenate(pieces)
    df = make_ohlcv(len(r), seed=seed, freq="4h")
    close = 100 * np.exp(np.cumsum(r))
    df["close"] = close
    df["open"] = np.r_[close[0], close[:-1]]
    df["high"] = np.maximum(df["open"], close) * (1 + np.abs(r) / 2)
    df["low"] = np.minimum(df["open"], close) * (1 - np.abs(r) / 2)
    return df


@pytest.fixture(scope="module")
def fitted():
    feats = {"A": regime_features(regime_path(0)), "B": regime_features(regime_path(1))}
    return RegimeModel.fit(feats, n_states=5, n_init=2), feats


def test_filter_is_causal(fitted):
    model, feats = fitted
    f = feats["A"]
    full = model.filter(f)
    for t in (100, 700, len(f) - 1):
        cut = model.filter(f.iloc[: t + 1])
        pd.testing.assert_series_equal(full.iloc[t].drop("regime"), cut.iloc[-1].drop("regime"),
                                       check_names=False, rtol=1e-9)


def test_filter_matches_hmmlearn_on_last_bar(fitted):
    """At the final bar, smoothing == filtering, so hmmlearn's posterior must equal ours there."""
    model, feats = fitted
    f = feats["B"]
    X = (f[FEATURES].to_numpy() - model.center) / model.scale
    ref = model.hmm.predict_proba(X)[-1]
    ours = model.filter(f).iloc[-1]
    for k, name in enumerate(model.names):
        assert ours[f"p_{name}"] >= ref[k] - 1e-6  # >= because duplicate names are summed
    assert np.isclose(sum(ref), 1)


def test_probabilities_sum_to_one_and_all_regimes_named(fitted):
    model, feats = fitted
    out = model.filter(feats["A"])
    p = out[[c for c in out.columns if c.startswith("p_") and c != "p_switch"]]
    assert np.allclose(p.sum(axis=1), 1)
    assert sorted(model.names) == sorted(["trend_up", "trend_down", "range", "squeeze", "extreme"])
    # each synthetic phase should be recognized (feature rows start ~30 bars in, after indicator warmup)
    phase = lambda a, b: out["regime"].iloc[a - 30:b - 30]
    assert (phase(1070, 1190) == "extreme").mean() > 0.7
    assert (phase(270, 520) == "trend_up").mean() > 0.7
    assert (phase(820, 1070) == "trend_down").mean() > 0.7


def test_state_naming_is_deterministic():
    # columns: ret_6, rvol, slope, bb_width
    means = np.array([[0.05, -4.0, 0.01, -3.0],    # trend up
                      [-0.05, -4.0, -0.01, -3.0],  # trend down
                      [0.0, -5.0, 0.0, -4.5],      # squeeze (narrowest bands)
                      [0.0, -2.0, 0.0, -2.0],      # extreme (highest vol)
                      [0.0, -4.5, 0.0, -3.5]])     # range
    assert _name_states(means, 5) == ["trend_up", "trend_down", "squeeze", "extreme", "range"]
    assert _name_states(means[[4, 3, 2, 1, 0]], 5) == ["range", "extreme", "squeeze", "trend_down", "trend_up"]


def test_fit_skips_symbols_without_history(fitted):
    _, feats = fitted
    m = RegimeModel.fit({"A": feats["A"], "NEW": feats["B"].iloc[:0]}, n_states=5, n_init=1)
    assert len(m.names) == 5
