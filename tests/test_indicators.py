import numpy as np
import pandas as pd
import pytest

from alpha.features import indicators as ind


def test_add_all_has_no_lookahead(ohlcv):
    """Row t must be identical whether or not later bars exist."""
    full = ind.add_all(ohlcv)
    num = full.select_dtypes(include=[np.number, bool]).columns
    for t in (400, 777, 1200, len(ohlcv) - 1):
        cut = ind.add_all(ohlcv.iloc[: t + 1])
        a, b = full[num].iloc[t], cut[num].iloc[t]
        pd.testing.assert_series_equal(a, b, check_names=False, rtol=1e-9, obj=f"row {t}")


def test_atr_rsi_known_values():
    df = pd.DataFrame({"open": [10.0] * 30, "high": [11.0] * 30, "low": [9.0] * 30, "close": [10.0] * 30},
                      index=pd.date_range("2026-01-01", periods=30, freq="1h", tz="UTC"))
    assert ind.atr(df, 14).iloc[-1] == pytest.approx(2.0)
    up = pd.Series(np.arange(1.0, 40.0))
    assert ind.rsi(up, 14).iloc[-1] == pytest.approx(100.0)
    assert ind.rsi(-up, 14).iloc[-1] == pytest.approx(0.0)


def test_session_vwap_resets_daily_and_matches_hand_calc():
    idx = pd.to_datetime(["2026-01-01 22:00", "2026-01-01 23:00", "2026-01-02 00:00"], utc=True)
    df = pd.DataFrame({"high": [11.0, 13.0, 21.0], "low": [9.0, 11.0, 19.0], "close": [10.0, 12.0, 20.0],
                       "volume": [1.0, 3.0, 2.0]}, index=idx)
    v = ind.session_vwap(df)
    assert v["vwap"].iloc[1] == pytest.approx((10 * 1 + 12 * 3) / 4)
    assert v["vwap"].iloc[2] == pytest.approx(20.0)  # new UTC day
    var = (1 * (10 - 11.5) ** 2 + 3 * (12 - 11.5) ** 2) / 4
    assert v["vwap_sd"].iloc[1] == pytest.approx(np.sqrt(var))


def test_donchian_excludes_current_bar(ohlcv):
    dc = ind.donchian(ohlcv, 20)
    t = 500
    assert dc["dc_hi"].iloc[t] == ohlcv["high"].iloc[t - 20:t].max()


def test_swing_points_reported_on_confirmation_bar():
    highs = [1, 2, 3, 9, 3, 2, 1, 1, 1]
    df = pd.DataFrame({"high": highs, "low": [h - 0.5 for h in highs]},
                      index=pd.date_range("2026-01-01", periods=9, freq="1h", tz="UTC"))
    sw = ind.swing_points(df, k=3)
    assert np.isnan(sw["swing_hi"].iloc[5])
    assert sw["swing_hi"].iloc[6] == 9 and sw["swing_hi_pos"].iloc[6] == 3


def test_anchored_vwap_from_anchor():
    df = pd.DataFrame({"high": [1.0, 2, 3, 4], "low": [1.0, 2, 3, 4], "close": [1.0, 2, 3, 4], "volume": [1.0, 1, 1, 1]},
                      index=pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC"))
    av = ind.anchored_vwap(df, pd.Series([np.nan, 1, 1, 1], index=df.index))
    assert np.isnan(av.iloc[0]) and av.iloc[3] == pytest.approx(3.0)
