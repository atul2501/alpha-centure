import numpy as np
import pandas as pd

from alpha.features.indicators import add_all
from alpha.strategies.playbook import PLAYBOOK, Breakout, MeanReversion, all_setups
from tests.conftest import make_ohlcv


def with_ctx(df):
    f = add_all(df)
    for p in ("h4_", "d1_"):
        f[p + "trend"] = 0.0
    f["funding_z"] = 0.0
    return f


def test_breakout_fires_after_squeeze_on_volume():
    n = 300
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(1)
    close = np.r_[100 + np.cumsum(rng.normal(0, 0.5, 200)), 100 + rng.normal(0, 0.05, 99), [0]]
    close[-1] = close[-2] + 3  # breakout bar
    open_ = np.r_[close[0], close[:-1]]
    hi, lo = np.maximum(open_, close) + 0.05, np.minimum(open_, close) - 0.05
    vol = np.r_[np.full(n - 1, 100.0), [1000.0]]
    df = pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close, "volume": vol,
                       "taker_buy_base": vol / 2}, index=idx)
    s = Breakout("breakout", ("15m",)).setups(with_ctx(df))
    assert len(s) >= 1 and s["side"].iloc[-1] == 1 and s.index[-1] == idx[-1]
    assert s["stop"].iloc[-1] < close[-1] < s["target"].iloc[-1]


def test_mean_reversion_fires_on_stretch_and_targets_mean():
    df = make_ohlcv(400, seed=3)
    df.iloc[-3:, df.columns.get_indexer(["open", "high", "low", "close"])] *= np.array([[0.97], [0.94], [0.91]])
    f = with_ctx(df)
    s = MeanReversion("mean_reversion", ("15m",)).setups(f)
    assert (s["side"] == 1).any()
    last = s[s["side"] == 1].iloc[-1]
    assert last["target"] > f.loc[s[s["side"] == 1].index[-1], "close"] > last["stop"]


def test_flat_market_produces_no_setups():
    idx = pd.date_range("2026-01-01", periods=500, freq="15min", tz="UTC")
    df = pd.DataFrame({"open": 100.0, "high": 100.01, "low": 99.99, "close": 100.0, "volume": 10.0,
                       "taker_buy_base": 5.0}, index=idx)
    assert all_setups(with_ctx(df), "15m").empty


def test_every_setup_has_valid_geometry():
    f = with_ctx(make_ohlcv(3000, seed=11))
    s = all_setups(f, "15m")
    assert not s.empty and set(s["strategy"]) <= {x.name for x in PLAYBOOK}
    close = f.loc[s.index, "close"]
    assert ((close - s["stop"]) * s["side"] > 0).all(), "stop on the wrong side"
