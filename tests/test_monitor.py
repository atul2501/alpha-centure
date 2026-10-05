import numpy as np
import pandas as pd

from alpha.strategy.monitor import MIN_DAYS, PAUSE_DRAWDOWN, MonitorState, evaluate


def _days(values, start="2026-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D", tz="UTC"))


def test_warming_up_stays_active():
    st = evaluate(_days([0.01] * (MIN_DAYS - 1)), MonitorState(), pd.Timestamp("2026-03-01", tz="UTC"))
    assert st.active and "warming up" in st.reason


def test_pauses_on_deep_shadow_drawdown_only():
    rng = np.random.default_rng(0)
    calm = list(rng.normal(0.001, 0.01, 200))
    today = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=199)
    assert evaluate(_days(calm), MonitorState(), today).active           # normal noise: stays on
    crash = calm[:150] + [-0.01] * 50                                       # ~-50% from peak
    st = evaluate(_days(crash), MonitorState(), today)
    assert not st.active and st.drawdown > PAUSE_DRAWDOWN and "paused" in st.reason


def test_resumes_only_when_recent_sharpe_recovers():
    today = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=199)
    paused = MonitorState(active=False, reason="paused", since="2025-12-01")
    still_bad = [0.0] * 140 + list(np.tile([0.01, -0.012], 30))            # flat-to-negative last 60 days
    assert not evaluate(_days(still_bad), paused, today).active
    good = [0.0] * 140 + list(np.tile([0.012, -0.006], 30))                # clearly positive last 60 days
    st = evaluate(_days(good), paused, today)
    assert st.active and "resumed" in st.reason


def test_state_round_trips_through_json():
    st = MonitorState(active=False, reason="x", since="2026-01-01", drawdown=0.25, sharpe=float("nan"))
    back = MonitorState.from_json(st.to_json())
    assert back.active is False and back.drawdown == 0.25 and np.isnan(back.sharpe)
