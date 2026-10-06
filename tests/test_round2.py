import numpy as np
import pandas as pd

from alpha.research.round2 import beta_neutral, btc_beta, cost_band


def _panel(n=24 * 200, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    btc = rng.normal(0, 0.01, n)
    rows = []
    for s, b in (("BTCUSDT", 1.0), ("AUSDT", 1.5), ("BUSDT", 0.5), ("CUSDT", 1.2)):
        r = b * btc + rng.normal(0, 0.005, n) if s != "BTCUSDT" else btc
        rows.append(pd.DataFrame({"time": idx, "symbol": s, "ret": r, "eligible": True}))
    return pd.concat(rows).set_index(["time", "symbol"]).sort_index()


def test_beta_neutral_removes_btc_exposure():
    p = _panel()
    t = p.index.get_level_values("time").unique()[-100:]
    w = pd.DataFrame(0.1, index=t, columns=["AUSDT", "BTCUSDT", "BUSDT", "CUSDT"])    # all long: full BTC beta
    out = beta_neutral(p)(w, t)
    b = btc_beta(p).reindex(t).reindex(columns=w.columns)
    assert ((out * b).sum(axis=1).abs() < 1e-9).all()


def test_cost_band_keeps_position_when_expected_gain_is_below_cost():
    idx = pd.date_range("2024-01-01", periods=144, freq="h", tz="UTC")
    w = pd.DataFrame({"A": 0.2, "B": 0.2}, index=idx)
    vol = pd.DataFrame(0.01, index=idx, columns=["A", "B"])
    score = pd.DataFrame({"A": 2.0, "B": 0.0001}, index=idx)           # A strong forecast, B ~zero
    costs = pd.DataFrame({"maker_bps": [2.0, 2.0], "taker_bps": [5.0, 5.0]}, index=["A", "B"])
    out = cost_band(score, costs)(w, None, vol, 72)
    assert (out["A"].iloc[-1] == 0.2) and (out["B"] == 0).all()        # B never opened: gain < 1.5 x cost
