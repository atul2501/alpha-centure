import numpy as np
import pandas as pd
import pytest

from alpha.research.portfolio_sim import newey_west_t, simulate, to_weights


def _frames(n=48):
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    cols = ["A", "B"]
    score = pd.DataFrame(np.tile([1.0, -1.0], (n, 1)), index=idx, columns=cols)
    score.iloc[::5] *= -1  # changes often
    vol = pd.DataFrame(0.01, index=idx, columns=cols)
    el = pd.DataFrame(True, index=idx, columns=cols)
    return idx, score, vol, el


def test_rebalance_every_h_holds_weights_between_rebalances():
    idx, score, vol, el = _frames()
    w = to_weights(score, vol, el, "ts", every=12)
    changes = w.diff().abs().sum(axis=1)
    assert (changes[changes > 0].index.hour % 12 == 0).all()
    assert w.abs().sum(axis=1).max() == pytest.approx(1.0)


def test_simulate_accounting():
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    w = pd.DataFrame({"A": [0.0, 1.0, 1.0]}, index=idx)
    ret = pd.DataFrame({"A": [0.0, 0.0, np.log(1.01)]}, index=idx)
    fund = pd.DataFrame({"A": [0.0, 0.0, 0.0001]}, index=idx)
    r = simulate(w, ret, fund, pd.Series({"A": 5.0}))
    h = r.hourly
    assert h["cost"].iloc[1] == pytest.approx(5e-4)          # bought 1.0 at 5 bps
    assert h["gross"].iloc[2] == pytest.approx(0.01)          # held over the +1% bar
    assert h["funding"].iloc[2] == pytest.approx(0.0001)      # long pays positive funding
    assert h["net"].sum() == pytest.approx(0.01 - 0.0001 - 5e-4)


def test_ineligible_gets_zero_weight_and_xs_is_dollar_neutral():
    idx, score, vol, el = _frames()
    el["B"] = False
    assert (to_weights(score, vol, el, "ts")["B"] == 0).all()
    el["B"] = True
    w = to_weights(score, vol, el, "xs")
    assert w.sum(axis=1).abs().max() == pytest.approx(0.0)


def test_newey_west_wider_for_autocorrelated_series():
    rng = np.random.default_rng(0)
    e = rng.normal(0.1, 1, 2000)
    x = pd.Series(np.convolve(e, np.ones(10) / 10, mode="same"))
    assert abs(newey_west_t(x, 20)) < abs(x.mean() / (x.std() / np.sqrt(len(x))))
