"""Candle loading from the DB (UTC timestamps)."""

from datetime import datetime

import pandas as pd
import psycopg

from alpha.audit import _df as _raw_df
from alpha.config import PERP_SUFFIX


def _df(conn: psycopg.Connection, sql: str, params=None) -> pd.DataFrame:
    """Query -> DataFrame with every timestamp column in UTC (the DB session may be in local time, which
    would silently shift UTC-day VWAP resets, hour-of-day features and funding settlement times)."""
    df = _raw_df(conn, sql, params)
    for c in df.columns:
        if isinstance(df[c].dtype, pd.DatetimeTZDtype):
            df[c] = df[c].dt.tz_convert("UTC")
    return df


def base_symbol(symbol: str) -> str:
    return symbol.removesuffix(PERP_SUFFIX)


def load_candles(conn: psycopg.Connection, symbol: str, interval: str,
                 start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
    df = _df(conn, """
        SELECT open_time, close_time, open::float8, high::float8, low::float8, close::float8,
               volume::float8, quote_volume::float8, trades, taker_buy_base::float8
        FROM candles
        WHERE symbol = %s AND interval = %s
          AND (%s::timestamptz IS NULL OR open_time >= %s) AND (%s::timestamptz IS NULL OR open_time < %s)
        ORDER BY open_time
    """, (symbol, interval, start, start, end, end))
    if df.empty:
        return df
    return df.set_index(pd.DatetimeIndex(df.pop("open_time")))
