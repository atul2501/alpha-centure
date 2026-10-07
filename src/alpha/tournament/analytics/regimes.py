"""Evaluation regimes: fixed causal rules (not fitted, not the models' own regimes), assigned to each trade at entry.

    trend      bull if the trailing 7-day return > +1 x its 7-day vol band and the 30-day EMA slope is up,
               bear if both are down, else sideways
    vol        high_vol if trailing 30-day realized vol is above its trailing 365-day median, else low_vol
    volume     high_volume if trailing 7-day quote volume is above its trailing 90-day median, else low_volume
    character  trending if the trailing 7-day variance ratio VR(16) > 1, else mean_reverting
"""

import numpy as np
import pandas as pd

DAY = 96
DIMENSIONS = ("trend", "vol", "volume", "character")


def label(raw: pd.DataFrame) -> pd.DataFrame:
    c = raw["sol_close"].astype(float)
    lr = np.log(c / c.shift(1))
    r7 = np.log(c / c.shift(7 * DAY))
    band = lr.rolling(7 * DAY, min_periods=DAY).std() * np.sqrt(7 * DAY) * 0.5
    ema = c.ewm(span=30 * DAY, adjust=False).mean()
    slope = np.log(ema / ema.shift(DAY))
    trend = np.where((r7 > band) & (slope > 0), "bull", np.where((r7 < -band) & (slope < 0), "bear", "sideways"))
    rv = lr.rolling(30 * DAY, min_periods=7 * DAY).std()
    vol = np.where(rv > rv.rolling(365 * DAY, min_periods=60 * DAY).median(), "high_vol", "low_vol")
    qv = raw["sol_quote_volume"].astype(float).rolling(7 * DAY, min_periods=DAY).sum()
    volume = np.where(qv > qv.rolling(90 * DAY, min_periods=30 * DAY).median(), "high_volume", "low_volume")
    r16 = lr.rolling(16).sum()
    vr = r16.rolling(7 * DAY, min_periods=DAY).var() / (16 * lr.rolling(7 * DAY, min_periods=DAY).var())
    character = np.where(vr > 1, "trending", "mean_reverting")
    return pd.DataFrame({"trend": trend, "vol": vol, "volume": volume, "character": character}, index=raw.index)


def attach(trades: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Regime columns at each trade's decision bar (entry_ts - 15 min)."""
    if trades.empty:
        return trades.assign(**{d: pd.Series(dtype=object) for d in DIMENSIONS})
    at = pd.DatetimeIndex(trades["entry_ts"]) - pd.Timedelta(minutes=15)
    lab = labels.reindex(at)
    t = trades.copy()
    for d in DIMENSIONS:
        t[d] = lab[d].to_numpy()
    return t
