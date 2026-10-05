"""Point-in-time tradable universe: which perps were eligible on each day, using only what was known then.

Rule: listed (first daily perp candle) at least MIN_AGE_DAYS before, and the median daily quote volume of the
previous VOL_WINDOW completed days >= MIN_MEDIAN_QUOTE_VOL. Day D's eligibility uses candles that closed before D,
so a backtest never trades a coin because of volume it had not printed yet.

The symbol list itself (today's 14 coins) was chosen with hindsight; this rule limits, but cannot remove, that
survivorship bias. Reports should also show results without the late listings (SUI, HYPE).
"""

import pandas as pd
import psycopg

from alpha.config import perp_symbol
from alpha.features.build import load_candles

MIN_AGE_DAYS = 90
VOL_WINDOW = 30
MIN_MEDIAN_QUOTE_VOL = 20e6


def eligibility(daily: pd.DataFrame, min_age_days: int = MIN_AGE_DAYS, window: int = VOL_WINDOW,
                min_median_vol: float = MIN_MEDIAN_QUOTE_VOL) -> pd.Series:
    """daily: 1d candles indexed by open time (UTC midnight) with quote_volume. Returns bool per day."""
    if daily.empty:
        return pd.Series(dtype=bool)
    days = daily.index
    age = (days - days[0]).days
    med = daily["quote_volume"].rolling(window, min_periods=window).median().shift(1)  # completed days only
    return pd.Series((age >= min_age_days) & (med >= min_median_vol).to_numpy(), index=days, name="eligible")


def universe_mask(conn: psycopg.Connection, symbols: list[str], **kw) -> pd.DataFrame:
    """Day x perp symbol -> eligible (False before listing)."""
    cols = {}
    for s in symbols:
        d = load_candles(conn, perp_symbol(s), "1d")
        cols[perp_symbol(s)] = eligibility(d, **kw)
    return pd.DataFrame(cols).fillna(False).astype(bool)
