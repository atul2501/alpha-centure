"""Strategy trace: JSON round trip, contributions add up to the blend, and a bare trace (final only) still works."""

import pytest

from alpha.strategy.trace import BLEND, BOOK, FINAL, SIGNAL, Layer, StrategyTrace, contributions, gross, scale_factor


def _trace() -> StrategyTrace:
    a = {"BTCUSDT": 0.4, "ETHUSDT": -0.2}
    b = {"BTCUSDT": -0.2, "ETHUSDT": 0.6}
    blend = {s: 0.5 * a[s] + 0.5 * b[s] for s in a}
    final = {s: 2 * v for s, v in blend.items()}
    return StrategyTrace("X", "2026-10-10T00:00:00+00:00", "every 72h", {"vol target": 0.2}, [
        Layer("sig", "Signal", SIGNAL, values={"BTCUSDT": 1.0, "ETHUSDT": -1.0}, features={"BTCUSDT": {"f": 0.5}}),
        Layer("a", "Book A", BOOK, values=a, blend_weight=0.5, inputs=["sig"]),
        Layer("b", "Book B", BOOK, values=b, blend_weight=0.5),
        Layer("blend", "Blend", BLEND, values=blend),
        Layer("final", "Final", FINAL, values=final)], final)


def test_json_round_trip():
    t = _trace()
    assert StrategyTrace.from_json(t.to_json()) == t


def test_contributions_add_up_to_blend_and_scale():
    t = _trace()
    c = contributions(t)
    for s in t.final:
        assert sum(v[s] for v in c.values()) == pytest.approx(t.of(BLEND)[0].values[s])
    assert scale_factor(t) == pytest.approx(2.0)
    assert gross(t.final) == pytest.approx(0.6)


def test_feeds_declared_or_previous_step():
    t = _trace()
    assert t.feeds(t.layers[1]) == ["sig"]                    # declared
    assert t.feeds(t.layers[2]) == ["sig"]                    # default: every signal
    assert t.feeds(t.layers[3]) == ["a", "b"]
    assert t.feeds(t.layers[4]) == ["blend"]
    assert t.symbols()[0] == "ETHUSDT"                        # biggest final position first


def test_bare_trace_has_no_breakdown_but_works():
    t = StrategyTrace("Y", "2026-10-10T00:00:00+00:00", final={"SOLUSDT": -0.3})
    assert contributions(t) == {} and scale_factor(t) is None and t.symbols() == ["SOLUSDT"]
