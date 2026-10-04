"""Point-in-time feature frames built from the DB.

Rule: a row for the bar closing at T only contains information that existed at T. Every outside series is
joined with a backward merge_asof on an *availability* time (close time / publish time), never on its start.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import psycopg

from alpha.audit import _df as _raw_df
from alpha.binance.parse import interval_td
from alpha.config import PERP_SUFFIX
from alpha.features.indicators import add_all, atr, ema, rsi

WARMUP_BARS = 300
CONTEXT_BARS = 400
CONTEXT_TFS = ("4h", "1d")
BENCHMARK = "BTCUSDT"

CORE_COLS = [
    "ret_1", "ret_3", "ret_12", "ret_48", "atr_pct", "rsi", "ema20_dist", "ema50_slope", "bb_width", "bb_z",
    "vwap_z", "squeeze", "squeeze_len", "vol_z", "rvol", "pvol", "taker_buy_ratio", "hour", "dow",
    "h4_trend", "h4_slope", "h4_rsi", "h4_atr_pct", "h4_ret_6", "d1_trend", "d1_slope", "d1_rsi", "d1_atr_pct", "d1_ret_6",
    "btc_ret_12", "btc_rvol", "btc_corr", "funding", "funding_z",
]
RICH_COLS = ["oi_chg_1h", "oi_chg_4h", "ls_ratio", "ls_ratio_z", "taker_ratio", "flow_delta_norm", "cvd_slope",
             "book_imb", "spread_bps"]


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


def _asof(left: pd.DataFrame, right: pd.DataFrame, on_right: str, cols: list[str]) -> pd.DataFrame:
    """Attach right[cols] as known at each left close_time (right rows keyed by availability time)."""
    if right.empty:
        return left.assign(**{c: np.nan for c in cols})
    r = right[[on_right, *cols]].sort_values(on_right)
    merged = pd.merge_asof(left[["close_time"]].reset_index(), r, left_on="close_time", right_on=on_right,
                           direction="backward")
    merged.index = left.index
    return left.join(merged[cols])


def _context(conn, symbol: str, tf: str, start, end) -> pd.DataFrame:
    """Trend/vol context from a higher timeframe, keyed by that bar's close time."""
    ctx = load_candles(conn, base_symbol(symbol), tf, start, end)
    if ctx.empty:
        return pd.DataFrame(columns=["close_time"])
    p = "h4_" if tf == "4h" else "d1_"
    e50 = ema(ctx["close"], 50)
    out = pd.DataFrame({
        "close_time": ctx["close_time"],
        f"{p}trend": np.sign(ctx["close"] - e50),
        f"{p}slope": e50.pct_change(5),
        f"{p}rsi": rsi(ctx["close"]),
        f"{p}atr_pct": atr(ctx) / ctx["close"],
        f"{p}ret_6": np.log(ctx["close"] / ctx["close"].shift(6)),
    })
    return out


def build_frame(conn: psycopg.Connection, symbol: str, tf: str, start: datetime | None = None,
                end: datetime | None = None, rich: bool = False) -> pd.DataFrame:
    """Indicators + context features for one symbol/timeframe. Index = bar open time (UTC)."""
    step = interval_td(tf)
    load_from = start - step * WARMUP_BARS if start else None
    df = load_candles(conn, symbol, tf, load_from, end)
    if df.empty:
        return df
    df = add_all(df)
    df["symbol"] = symbol

    # higher-timeframe context (only closed bars: keyed by their close_time)
    for ctx_tf in CONTEXT_TFS:
        # enough context bars that its EMAs converge to the same values as on full history (live/research parity)
        ctx_from = load_from - interval_td(ctx_tf) * CONTEXT_BARS if load_from else None
        ctx = _context(conn, symbol, ctx_tf, ctx_from, end)
        df = _asof(df, ctx, "close_time", [c for c in ctx.columns if c != "close_time"])

    # benchmark (BTC) context for alts: same timeframe bars close at the same instant
    btc = load_candles(conn, BENCHMARK, tf, load_from, end)
    if not btc.empty:
        btc_ret1 = np.log(btc["close"]).diff()
        b = pd.DataFrame({"btc_ret_12": np.log(btc["close"] / btc["close"].shift(12)),
                          "btc_rvol": btc_ret1.rolling(20).std(ddof=0), "btc_ret_1": btc_ret1})
        df = df.join(b, how="left")
        df["btc_corr"] = df["ret_1"].rolling(96).corr(df.pop("btc_ret_1"))

    sym = base_symbol(symbol)
    fund = _df(conn, "SELECT funding_time AS t, funding_rate FROM funding_rate WHERE symbol = %s ORDER BY 1", (sym,))
    if not fund.empty:
        fr = fund["funding_rate"]
        fund["funding"] = fr
        fund["funding_z"] = (fr - fr.rolling(90).mean()) / fr.rolling(90).std(ddof=0)
    df = _asof(df, fund, "t", ["funding", "funding_z"])

    if rich:
        df = _add_rich(conn, df, sym, tf, load_from, end)

    if start is not None:
        df = df[df.index >= start]
    return df


def _add_rich(conn, df: pd.DataFrame, sym: str, tf: str, start, end) -> pd.DataFrame:
    """Futures positioning + order flow + book. Short history; only for the 'rich' feature set."""
    five = timedelta(minutes=5)
    oi = _df(conn, "SELECT ts, sum_open_interest_value AS oi FROM open_interest WHERE symbol = %s ORDER BY ts", (sym,))
    if not oi.empty:
        oi["avail"] = oi["ts"] + five  # 5m stats are published after their bucket closes
        oi["oi_chg_1h"] = oi["oi"].pct_change(12)
        oi["oi_chg_4h"] = oi["oi"].pct_change(48)
    df = _asof(df, oi, "avail", ["oi_chg_1h", "oi_chg_4h"])

    ls = _df(conn, "SELECT ts, long_short_ratio AS ls_ratio FROM long_short_ratio WHERE symbol = %s ORDER BY ts", (sym,))
    if not ls.empty:
        ls["avail"] = ls["ts"] + five
        ls["ls_ratio_z"] = (ls["ls_ratio"] - ls["ls_ratio"].rolling(288).mean()) / ls["ls_ratio"].rolling(288).std(ddof=0)
    df = _asof(df, ls, "avail", ["ls_ratio", "ls_ratio_z"])

    tr = _df(conn, "SELECT ts, buy_sell_ratio AS taker_ratio FROM taker_ratio WHERE symbol = %s ORDER BY ts", (sym,))
    if not tr.empty:
        tr["avail"] = tr["ts"] + five
    df = _asof(df, tr, "avail", ["taker_ratio"])

    n = max(1, int(interval_td(tf) / timedelta(minutes=1)))
    fl = _df(conn, """SELECT minute, delta, buy_vol + sell_vol AS vol FROM flow_1m
                      WHERE symbol = %s AND complete ORDER BY minute""", (sym,))
    if not fl.empty:
        fl["avail"] = fl["minute"] + timedelta(minutes=1)
        fl["flow_delta_norm"] = fl["delta"].rolling(n).sum() / fl["vol"].rolling(n).sum()
        cvd = fl["delta"].cumsum()
        fl["cvd_slope"] = (cvd - cvd.shift(n)) / fl["vol"].rolling(n).sum()
    df = _asof(df, fl, "avail", ["flow_delta_norm", "cvd_slope"])

    ob = _df(conn, "SELECT ts, imbalance AS book_imb, spread_bps FROM orderbook_snap WHERE symbol = %s ORDER BY ts", (sym,))
    if not ob.empty:
        ob["avail"] = ob["ts"] + timedelta(minutes=1)
    df = _asof(df, ob, "avail", ["book_imb", "spread_bps"])
    return df
