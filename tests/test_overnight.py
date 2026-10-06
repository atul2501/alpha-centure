import numpy as np
import pandas as pd
import pytest

from alpha.research.flow1m import features_15m, to_weights_bars
from alpha.research.overnight import guard_dispersion, guard_reversal, residual_momentum


def _m1(n=60 * 24 * 40, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    vol = rng.uniform(5, 15, n)
    return pd.DataFrame({"close": close, "volume": vol, "trades": rng.integers(10, 50, n),
                         "buy": vol * rng.uniform(0.3, 0.7, n)}, index=idx)


def test_flow_features_use_only_minutes_closed_before_the_decision():
    m1 = _m1()
    full = features_15m(m1)
    cut_at = full.index[-200]                       # a bar; its close is cut_at + 15 min
    part = features_15m(m1[m1.index < cut_at + pd.Timedelta(minutes=15)])
    cols = ["flow_5", "flow_15", "flow_60", "flow_240", "cvd_slope_60", "buy_surprise_60", "vol_burst_15", "close"]
    pd.testing.assert_series_equal(full.loc[cut_at, cols], part.loc[cut_at, cols], check_names=False)
    assert np.isnan(part.loc[cut_at, "fwd_1"])     # the target needs the future: unknown at decision time


def test_weights_zero_on_ineligible_and_hold_between_rebalances():
    idx = pd.date_range("2024-01-01", periods=40, freq="15min", tz="UTC")
    sc = pd.DataFrame(np.random.default_rng(0).normal(size=(40, 3)), index=idx, columns=list("ABC"))
    vol = pd.DataFrame(0.01, index=idx, columns=list("ABC"))
    el = pd.DataFrame(True, index=idx, columns=list("ABC"))
    el["C"] = False
    w = to_weights_bars(sc, vol, el, "ts", 4)
    assert (w["C"] == 0).all()
    changes = w.diff().abs().sum(axis=1)
    assert all(i % 4 == 0 for i, c in enumerate(changes) if c > 0 and i > 0)
    assert w.abs().sum(axis=1).max() == pytest.approx(1.0)


def _panel(n=24 * 200, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rows = []
    for s in ["BTCUSDT", "AUSDT", "BUSDT", "CUSDT"]:
        r = rng.normal(0, 0.01, n)
        rows.append(pd.DataFrame({"time": idx, "symbol": s, "ret": r, "eligible": True}))
    return pd.concat(rows).set_index(["time", "symbol"]).sort_index()


def test_residual_momentum_and_guards_do_not_look_ahead():
    p = _panel()
    t = p.index.get_level_values("time").unique()[-500]
    q = p.copy()
    future = q.index.get_level_values("time") > t
    q.loc[future, "ret"] = q.loc[future, "ret"] * 50          # change only the future
    a, b = residual_momentum(p)["rmom_168"].loc[t], residual_momentum(q)["rmom_168"].loc[t]
    pd.testing.assert_series_equal(a, b)
    assert guard_reversal(p).loc[t] == guard_reversal(q).loc[t]
    assert guard_dispersion(p).loc[t] == guard_dispersion(q).loc[t]
    g = guard_reversal(p)
    assert g.between(0.5, 1.0).all() and guard_dispersion(p).between(0, 1.0).all()
