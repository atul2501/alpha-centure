"""Animated views: edge speed/colour rules, cycle progress, and the SVGs render for real and minimal inputs."""

import pandas as pd

from alpha.live.flowviz import COLOR, cycle_progress, edge_style, pipeline_svg, strategy_svg
from alpha.live.workflow import FAIL, IDLE, OK, WARN, Check, Stage
from alpha.strategy.trace import StrategyTrace
from tests.test_trace import _trace


def test_edge_style():
    assert not edge_style(1000, FAIL).moving and edge_style(1000, FAIL).color == COLOR[FAIL]
    assert not edge_style(0, OK).moving and edge_style(None, OK).color == COLOR[IDLE]
    slow, fast = edge_style(1, OK), edge_style(5000, OK)
    assert slow.moving and fast.moving and fast.dur_s < slow.dur_s
    assert edge_style(50, WARN).color == COLOR[WARN]


def test_cycle_progress():
    now = pd.Timestamp("2026-10-09 12:00", tz="UTC")
    assert cycle_progress("2026-10-10 00:00:00+00:00", 72, now) == 1 - 12 / 72
    assert cycle_progress(None, 72, now) is None
    assert cycle_progress("2026-10-09 11:00:00+00:00", 72, now) == 1.0


def test_pipeline_renders_with_and_without_data():
    keys = ["binance", "collector", "database", "strategy", "risk", "execution", "ledger"]
    ss = [Stage(k, k, "", [Check("x", OK, "1")]) for k in keys]
    svg = pipeline_svg(ss, {"msgs_min": 1400, "rows_min": 1500, "book_min": 1380, "next_decision": None})
    assert svg.count("<svg") == 1 and 'class="flow"' in svg
    idle = pipeline_svg([Stage(k, k, "") for k in keys], {})
    assert 'class="still"' in idle                          # no message rate = nothing moving into the collector


def test_strategy_svg_any_trace():
    full = strategy_svg(_trace())
    assert "Book A" in full and "BTC" in full and "1 · Signals" in full
    bare = strategy_svg(StrategyTrace("Y", "t", final={"SOLUSDT": -0.3}))
    assert "SOL" in bare and "<svg" in bare
