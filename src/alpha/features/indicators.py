"""Technical indicators on an OHLCV frame (columns: open, high, low, close, volume; UTC DatetimeIndex
of bar open times). Every value at row t uses only rows <= t, so nothing here looks ahead."""

import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    return pd.DataFrame({"bb_mid": mid, "bb_up": mid + k * sd, "bb_lo": mid - k * sd,
                         "bb_width": 2 * k * sd / mid, "bb_z": (close - mid) / sd})
