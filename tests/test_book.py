"""V4_carry book: 0.5 ridge + 0.25 rules + 0.25 carry, re-scaled to the vol target; caps hold and ineligible coins get nothing."""

import numpy as np
import pandas as pd
import pytest

from alpha.research.panel import wide
from alpha.research.phase4 import MAX_COIN, MAX_GROSS, vol_target
from alpha.strategy import p6

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]


@pytest.fixture(scope="module")
def book():
    rng = np.random.default_rng(0)
    t = pd.date_range("2026-01-01", periods=150 * 24, freq="h", tz="UTC")
    parts = []
    for i, s in enumerate(SYMS):
        ret = rng.normal(0.0001 * (i - 1), 0.01, len(t))
        fr = np.where(t.hour % 8 == 0, rng.normal(1e-4, 1e-4, len(t)), 0.0)  # funding settles every 8h
        parts.append(pd.DataFrame({"symbol": s, "ret": ret, "close": 100 * np.exp(np.cumsum(ret)),
                                   "eligible": s != "XRPUSDT", "funding_rate": fr, "funding_last": fr,
                                   "premium": rng.normal(0, 1e-4, len(t)), "oi_value": 1e8, "ls_ratio": 1.0,
                                   "toptrader_ls": 1.0, "taker_buy_ratio": 0.5}, index=pd.Index(t, name="time")))
    panel = pd.concat(parts).set_index("symbol", append=True).sort_index()
    sc = pd.DataFrame(rng.normal(0, 1, (len(t), len(SYMS))), index=t, columns=SYMS)
    return panel, sc, p6.book_weights(panel, sc)


def test_caps_hold(book):
    w = book[2]
    assert (w.abs().sum(axis=1) <= MAX_GROSS + 1e-9).all()
    assert (w.abs() <= MAX_COIN + 1e-9).all().all()


def test_ineligible_coin_gets_nothing(book):
    assert (book[2]["XRPUSDT"] == 0).all()


def test_is_the_mix_rescaled(book):
    panel, sc, w = book
    base = p6.target_weights(panel, sc)
    blend = (0.5 * base + 0.25 * p6.rules_weights(panel).reindex_like(base).fillna(0.0)
             + 0.25 * p6.carry_weights(panel).reindex_like(base).fillna(0.0))
    np.testing.assert_allclose(w.to_numpy(), vol_target(blend, wide(panel, "ret")).to_numpy(), atol=1e-12)
    assert w.abs().sum(axis=1).iloc[-1000:].mean() > blend.abs().sum(axis=1).iloc[-1000:].mean()  # scaled up
