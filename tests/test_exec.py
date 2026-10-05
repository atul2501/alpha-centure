import pytest

from alpha.exec.account import Account
from alpha.exec.fillsim import BEYOND_BOOK_PENALTY_BPS, Book, MakerOrder, taker_fill
from alpha.exec.oms import LotFilter, RiskState, cap_targets, margin_ratio, plan_orders


def _book():
    b = Book("X")
    b.update([["100.0", "1"], ["99.9", "2"]], [["100.1", "1"], ["100.2", "3"]], 1, 2)
    return b


def test_taker_walks_levels_and_flags_beyond_book():
    f = taker_fill(_book(), +1, 2.0)
    assert f.price == pytest.approx((100.1 + 100.2) / 2) and f.liquidity == "taker" and not f.beyond_book
    f = taker_fill(_book(), -1, 4.0)  # bids hold 3
    expected = (100.0 + 2 * 99.9 + 99.9 * (1 - BEYOND_BOOK_PENALTY_BPS / 1e4)) / 4
    assert f.price == pytest.approx(expected) and f.beyond_book


def test_maker_queue_then_partial_then_sweep():
    o = MakerOrder(1, "X", +1, 100.0, qty=2.0, queue_ahead=1.5, placed_ms=0, expires_ms=10)
    assert o.on_trade(100.0, 1.0, buyer_is_maker=False) is None      # aggressive BUY never fills our bid
    assert o.on_trade(100.0, 1.0, buyer_is_maker=True) is None       # eats queue ahead (1.5 -> 0.5)
    f = o.on_trade(100.0, 1.0, buyer_is_maker=True)                  # 0.5 queue, 0.5 to us
    assert f.qty == pytest.approx(0.5) and f.liquidity == "maker" and f.price == 100.0
    assert o.on_trade(100.1, 5.0, buyer_is_maker=True) is None       # above our bid: not ours
    f = o.on_trade(99.9, 0.01, buyer_is_maker=True)                  # traded through: rest fills
    assert f.qty == pytest.approx(1.5) and o.remaining == 0


def test_account_realized_unrealized_fees_funding():
    a = Account(10_000)
    a.apply_fill("X", +1, 10, 100.0, "taker")      # fee 0.5
    a.apply_fill("X", -1, 4, 110.0, "maker")       # realize 40, fee 0.088
    assert a.realized == pytest.approx(40.0)
    assert a.pos("X").qty == pytest.approx(6) and a.pos("X").entry == pytest.approx(100.0)
    paid = a.apply_funding("X", 110.0, 0.0001)     # long pays
    assert paid == pytest.approx(6 * 110 * 0.0001)
    eq = a.equity({"X": 120.0})
    assert eq == pytest.approx(10_000 + 40 - 0.5 - 0.088 - paid + 6 * 20)
    a.apply_fill("X", -1, 10, 120.0, "taker")      # flip to -4 at 120
    assert a.pos("X").qty == pytest.approx(-4) and a.pos("X").entry == pytest.approx(120.0)


def test_plan_orders_band_lots_min_notional_and_reduce_only():
    marks = {"A": 100.0, "B": 10.0, "C": 1.0}
    f = {"A": LotFilter(0.001, 0.001, 5.0), "B": LotFilter(0.1, 0.1, 5.0), "C": LotFilter(1, 1, 5.0)}
    pos = {"A": 10.0, "B": 0.0, "C": 0.0}          # A at 0.1 of 10k equity
    orders = plan_orders({"A": 0.105, "B": 0.2, "C": 0.0004}, pos, marks, 10_000, f, band=0.01)
    assert [o.symbol for o in orders] == ["B"]     # A inside band, C below min notional / band
    assert orders[0].side == 1 and orders[0].qty == pytest.approx(200.0)
    ro = plan_orders({"A": 0.3, "B": -0.2}, pos, marks, 10_000, f, band=0.01, reduce_only=True)
    assert ro == []                                # no new risk allowed


def test_caps_and_risk_state():
    t = cap_targets({"A": 0.9, "B": -0.5, "C": 0.5, "D": 0.5, "E": 0.5, "F": 0.5, "G": 0.5})
    assert max(abs(v) for v in t.values()) <= 0.5 and sum(abs(v) for v in t.values()) == pytest.approx(3.0)
    r = RiskState(peak_equity=10_000)
    r.observe(10_000, "d1")
    r.observe(9_650, "d1")
    assert r.daily_stop and not r.halted
    r.observe(8_990, "d2")
    assert not r.halted                            # -10%: within the strategy's normal drawdowns
    r.observe(6_990, "d3")
    assert r.halted                                # -30%: catastrophe stop
    assert margin_ratio({"A": 30_000}, 10_000) == pytest.approx(0.045)
