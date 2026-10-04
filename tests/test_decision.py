import pandas as pd

from alpha.decision import LONG, PASS, SHORT, Policy, decide

T = pd.Timestamp("2026-05-01 10:00", tz="UTC")


def cands(rows):
    cols = ["symbol", "strategy", "tf", "side", "regime", "ev_r"]
    return pd.DataFrame(rows, columns=cols, index=pd.DatetimeIndex([T] * len(rows), name="signal_time"))


POL = Policy(enabled={("breakout", "1h", "trend_up"), ("mean_reversion", "1h", "range"),
                      ("trend_pullback", "1h", "trend_up")}, min_ev_r=0.1)


def test_gates_and_reasons():
    c = cands([
        ("BTC", "breakout", "1h", 1, "trend_up", 0.30),       # best for BTC -> LONG
        ("BTC", "trend_pullback", "1h", 1, "trend_up", 0.20),  # same symbol, lower EV
        ("ETH", "breakout", "1h", 1, "range", 0.50),           # regime not enabled
        ("SOL", "mean_reversion", "1h", -1, "range", 0.05),    # EV too low
        ("SUI", "breakout", "1h", -1, "trend_up", 0.4),        # symbol busy
    ])
    d = decide(c, POL, busy_symbols={"SUI"})
    assert list(d["action"]) == [LONG, PASS, PASS, PASS, PASS]
    assert list(d["reason"]) == ["", "not_best", "regime_off", "edge_below_min", "position_open"]
    assert d.index.name == "signal_time" and (d.index == T).all()


def test_long_and_short_on_same_symbol_is_conflict():
    c = cands([("BTC", "breakout", "1h", 1, "trend_up", 0.3), ("BTC", "trend_pullback", "1h", -1, "trend_up", 0.3)])
    d = decide(c, POL)
    assert (d["action"] == PASS).all() and (d["reason"] == "conflict").all()


def test_short_action_and_policy_roundtrip():
    c = cands([("BTC", "mean_reversion", "1h", -1, "range", 0.2)])
    assert decide(c, POL)["action"].iloc[0] == SHORT
    assert Policy.from_dict(POL.to_dict()) == POL


def test_policy_learn_requires_positive_lower_bound():
    rows = [("a", "1h", "x", 0.5)] * 40 + [("b", "1h", "x", -0.2)] * 40 + [("c", "1h", "x", 2.0)] * 5
    train = pd.DataFrame(rows, columns=["strategy", "tf", "regime", "r"])
    train.loc[train.index[:40], "r"] += [(-1) ** i * 0.3 for i in range(40)]
    p = Policy.learn(train, min_trades=30)
    assert p.enabled == {("a", "1h", "x")}


def test_cost_gate_and_exposure_cap():
    pol = Policy(enabled={("breakout", "1h", "trend_up")}, min_ev_r=0.0, max_cost_r=0.15, max_same_side=1)
    c = cands([("BTC", "breakout", "1h", 1, "trend_up", 0.3), ("ETH", "breakout", "1h", 1, "trend_up", 0.5),
               ("SOL", "breakout", "1h", 1, "trend_up", 0.9)])
    c["close_px"], c["stop"] = 100.0, [99.0, 98.0, 99.5]  # cost_r: 0.12, 0.06, 0.24
    d = decide(c, pol)
    assert list(d["reason"]) == ["exposure_cap", "", "cost_too_high"]
    assert decide(c, pol, open_sides={1: 1})["action"].eq(PASS).all()


def test_weighted_policy_learn_prefers_recent_behaviour():
    import numpy as np
    old = [("a", "1h", "x", -0.5)] * 60
    new = [("a", "1h", "x", 0.6)] * 40
    train = pd.DataFrame(old + new, columns=["strategy", "tf", "regime", "r"])
    train["r"] += np.tile([0.05, -0.05], 50)
    assert Policy.learn(train).enabled == set()
    w = np.r_[np.full(60, 0.05), np.ones(40)]
    assert Policy.learn(train, weights=w).enabled == {("a", "1h", "x")}
