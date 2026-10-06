import numpy as np
import pandas as pd

from alpha.universe import eligibility


def _daily(vols):
    idx = pd.date_range("2024-01-01", periods=len(vols), freq="D", tz="UTC")
    return pd.DataFrame({"quote_volume": vols}, index=idx)


def test_needs_age_and_volume_known_before_the_day():
    vols = np.full(200, 30e6)
    e = eligibility(_daily(vols), min_age_days=90, window=30, min_median_vol=20e6)
    assert not e.iloc[:90].any()            # too young
    assert e.iloc[90:].all()


def test_volume_spike_on_day_d_does_not_make_day_d_eligible():
    vols = np.full(200, 5e6)
    vols[150:] = 100e6
    e = eligibility(_daily(vols), min_age_days=0, window=3, min_median_vol=20e6)
    assert not e.iloc[150] and not e.iloc[151]   # median of the 3 previous completed days still low
    assert e.iloc[152]                            # 2 of 3 previous days high -> median high
    assert eligibility(_daily([])).empty


def test_today_is_known_before_its_daily_candle_closes():
    vols = np.full(120, 30e6)                       # completed days up to yesterday
    e = eligibility(_daily(vols), min_age_days=90, window=30, min_median_vol=20e6)
    today = _daily(vols).index[-1] + pd.Timedelta(days=1)
    assert today in e.index and bool(e.loc[today])  # live code needs today's value
