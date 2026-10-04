import numpy as np
import pandas as pd
import pytest

from alpha.strategies.labels import EXIT_MENU, Costs, Entry, ExitPolicy, apply_exit, label_multi, label_setups
from tests.test_labels import NO_COST, frame, setup


def with_atr(f, atr=1.0):
    return f.assign(atr=atr)


def test_breakeven_moves_after_close_beyond_1r_and_applies_next_bar():
    # entry 100, stop 98 (risk 2). bar1 closes 102.5 (+1.25R) -> stop to 100 from bar2. bar2 dips to 99.5 -> breakeven exit
    f = with_atr(frame([(100, 100, 100, 100), (100, 103, 99.8, 102.5), (102.5, 102.6, 99.5, 100), (100, 100, 100, 100)]))
    r = label_setups(setup(f, 0, 1, 98, 110, max_bars=3), f, NO_COST, EXIT_MENU["breakeven"]).iloc[0]
    assert r["outcome"] == "breakeven" and r["exit"] == 100 and r["r"] == pytest.approx(0.0)
    fixed = label_setups(setup(f, 0, 1, 98, 110, max_bars=3), f, NO_COST).iloc[0]
    assert fixed["outcome"] == "timeout"  # without the rule the dip to 99.5 does not touch 98


def test_breakeven_not_triggered_by_intrabar_high_only():
    # high reaches +2R but the bar closes below +1R: stop must stay at 98
    f = with_atr(frame([(100, 100, 100, 100), (100, 104, 99.9, 101), (101, 101, 99, 99.5), (99.5, 99.5, 97, 97)]))
    r = label_setups(setup(f, 0, 1, 98, 110), f, NO_COST, EXIT_MENU["breakeven"]).iloc[0]
    assert r["outcome"] == "stop" and r["exit"] == 98


def test_trailing_stop_only_ratchets_up():
    ex = ExitPolicy("t", trail_atr=1.0, trail_start_r=1.0, keep_target=False)
    # entry 100, stop 98 (risk 2), ATR 1
    bars = [(100, 100, 100, 100),
            (100, 103, 100, 102.5),      # close +1.25R -> trailing on, best 103 -> stop 102
            (102.5, 106, 102.2, 105.5),  # best 106 -> stop 105
            (105.5, 105.6, 105.1, 105.3),  # ATR jumps to 3 -> 106-3=103, must NOT loosen below 105
            (105.3, 105.4, 104.5, 104.6),  # low 104.5 hits 105 -> exit at 105
            (104.6, 104.6, 104.6, 104.6)]
    f = frame(bars).assign(atr=[1.0, 1.0, 1.0, 3.0, 1.0, 1.0])
    r = label_setups(setup(f, 0, 1, 98, np.nan, max_bars=5), f, NO_COST, ex).iloc[0]
    assert r["outcome"] == "trail" and r["exit"] == 105 and r["r"] == pytest.approx(2.5)


def test_partial_then_runner_blends_legs():
    ex = ExitPolicy("p", partial_at_r=1.0, partial_frac=0.5, breakeven_at_r=1.0, keep_target=False)
    # risk 2: partial at 102 hit on bar1 (close 102.5 -> stop to BE), bar2 falls to 99 -> runner at BE 100
    f = with_atr(frame([(100, 100, 100, 100), (100, 102.6, 99.9, 102.5), (102, 102, 99, 99.5), (99.5, 99.5, 99.5, 99.5)]))
    r = label_setups(setup(f, 0, 1, 98, np.nan), f, NO_COST, ex).iloc[0]
    assert r["gross"] == pytest.approx(0.5 * 0.02 + 0.5 * 0.0)
    assert r["r"] == pytest.approx(0.5)


def test_stop_and_partial_same_bar_counts_as_stop():
    ex = ExitPolicy("p", partial_at_r=1.0, partial_frac=0.5, keep_target=False)
    f = with_atr(frame([(100, 100, 100, 100), (100, 103, 97, 100), (100, 100, 100, 100)]))
    r = label_setups(setup(f, 0, 1, 98, np.nan), f, NO_COST, ex).iloc[0]
    assert r["outcome"] == "stop" and r["r"] == pytest.approx(-1.0)


def test_limit_entry_fills_at_better_price_or_not_at_all():
    ent = Entry("limit", offset_atr=0.5, fill_bars=2)
    # signal close 100, ATR 2 -> long limit 99. bar1 low 99.5 (no fill), bar2 low 98.8 -> fill at 99
    f = with_atr(frame([(100, 100, 100, 100), (100, 101, 99.5, 100.5), (100.5, 101, 98.8, 100), (100, 104, 100, 103.5),
                        (103.5, 104, 103, 103.5)]), atr=2.0)
    r = label_setups(setup(f, 0, 1, 97, 103), f, NO_COST, entry=ent).iloc[0]
    assert r["entry"] == 99 and r["entry_time"] == f.index[2] and r["outcome"] == "target"
    f2 = with_atr(frame([(100, 100, 100, 100), (100, 101, 99.5, 100.5), (100.5, 102, 99.6, 101), (101, 104, 100, 103.5)]), 2.0)
    assert label_setups(setup(f2, 0, 1, 97, 103), f2, NO_COST, entry=ent).empty  # never filled -> no trade


def test_maker_fees_per_leg():
    costs = Costs(fee=0.0005, slippage=0.0001, fee_entry=0.0002, fee_target=0.0002)
    f = with_atr(frame([(100, 100, 100, 100), (100, 100.5, 99, 100), (100, 103.5, 99.9, 103), (103, 103, 103, 103)]), 2.0)
    r = label_setups(setup(f, 0, 1, 97, 103), f, costs, entry=Entry("limit", 0.0, 2)).iloc[0]
    assert r["gross"] == pytest.approx(0.03) and r["pnl"] == pytest.approx(0.03 - 0.0002 - 0.0002)
    m = label_setups(setup(f, 0, 1, 97, 103), f, costs).iloc[0]  # market entry: taker + slippage on entry
    assert m["pnl"] == pytest.approx(0.03 - 0.0006 - 0.0002)


def test_label_multi_and_apply_exit():
    f = with_atr(frame([(100, 100, 100, 100), (100, 103, 99.8, 102.5), (102.5, 102.6, 99.5, 100), (100, 100, 100, 100)]))
    s = pd.concat([setup(f, 0, 1, 98, 110, max_bars=3), setup(f, 0, -1, 103, 95, max_bars=3)])
    s["strategy"] = ["a", "b"]
    ds = label_multi(s, f, NO_COST, {k: EXIT_MENU[k] for k in ("fixed", "breakeven")})
    assert {"r__fixed", "r__breakeven", "outcome__breakeven"} <= set(ds.columns)
    out = apply_exit(ds, {"a": "breakeven"})
    assert list(out["exit_policy"]) == ["breakeven", "fixed"]
    assert out["outcome"].iloc[0] == "breakeven" and out["outcome"].iloc[1] == ds["outcome"].iloc[1]
