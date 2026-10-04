import numpy as np
import pandas as pd
import pytest


def make_ohlcv(n: int = 1500, seed: int = 7, freq: str = "15min", start: str = "2026-01-01") -> pd.DataFrame:
    """Random-walk OHLCV with a realistic shape, UTC index of bar open times."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, 0.003, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    vol = rng.lognormal(3, 0.5, n)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    step = idx[1] - idx[0]
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol,
                         "taker_buy_base": vol * rng.uniform(0.3, 0.7, n),
                         "close_time": idx + step - pd.Timedelta(milliseconds=1)}, index=idx)


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()
