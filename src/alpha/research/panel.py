"""Hourly research panel for the perp universe: one row per (hour, symbol), everything as known at the bar close.

Columns (bar index = open time, decisions at close = open + 1h):
    close, ret (log, close-to-close), quote_volume, taker_buy_ratio
    funding_rate   rate of a settlement at this bar's close (0 elsewhere): what a position held over it pays
    premium        premium index close (perp vs index), known at close
    oi_value, ls_ratio, toptrader_ls, taker_ls   futures_metrics, published 5 min after their bucket closes
    depth_1pct     min(bid, ask) quote notional within 1% of mid (2023+), else NaN
    eligible       point-in-time universe rule (alpha.universe)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

from alpha.config import perp_symbol
from alpha.features.build import _df, load_candles
from alpha.universe import eligibility

CACHE = Path("data/research")


def _symbol_frame(conn, sym: str, start, end) -> pd.DataFrame:
    c = load_candles(conn, perp_symbol(sym), "1h", start, end)
    if c.empty:
        return c
    f = pd.DataFrame(index=c.index)
    f["close_time"] = c["close_time"]
    f["close"] = c["close"]
    f["ret"] = np.log(c["close"]).diff()
    f["quote_volume"] = c["quote_volume"]
    f["taker_buy_ratio"] = (c["taker_buy_base"] / c["volume"].replace(0, np.nan)).astype(float)
    bar_close = f.index + pd.Timedelta(hours=1)

    fund = _df(conn, "SELECT funding_time, funding_rate FROM funding_rate WHERE symbol = %s ORDER BY 1", (sym,))
    f["funding_rate"] = 0.0
    if not fund.empty:
        # a settlement at time T is paid by positions open over the bar that closes at T (bar open = T - 1h)
        st = fund.set_index(fund["funding_time"].dt.floor("h") - pd.Timedelta(hours=1))["funding_rate"]
        st = st.groupby(level=0).sum()
        f["funding_rate"] = st.reindex(f.index).fillna(0.0).to_numpy()
        # last known settled rate (for signals): asof at bar close
        known = pd.merge_asof(pd.DataFrame({"t": bar_close}), fund.rename(columns={"funding_time": "t"}),
                              on="t", direction="backward")
        f["funding_last"] = known["funding_rate"].to_numpy()

    prem = _df(conn, """SELECT close_time, close FROM premium_kline WHERE symbol = %s AND interval = '1h'
                        ORDER BY 1""", (sym,))
    if not prem.empty:
        p = pd.merge_asof(pd.DataFrame({"t": bar_close}), prem.rename(columns={"close_time": "t", "close": "premium"})
                          .assign(t=lambda d: d["t"] + pd.Timedelta(milliseconds=1)), on="t", direction="backward")
        f["premium"] = p["premium"].to_numpy()

    m = _df(conn, """SELECT ts, sum_open_interest_value AS oi_value, ls_ratio, toptrader_ls_position AS toptrader_ls,
                            taker_ls_vol_ratio AS taker_ls FROM futures_metrics WHERE symbol = %s ORDER BY ts""", (sym,))
    cols = ["oi_value", "ls_ratio", "toptrader_ls", "taker_ls"]
    if not m.empty:
        m["avail"] = m["ts"] + pd.Timedelta(minutes=5)
        j = pd.merge_asof(pd.DataFrame({"t": bar_close}), m.drop(columns="ts"), left_on="t", right_on="avail",
                          direction="backward", tolerance=pd.Timedelta(hours=2))
        for col in cols:
            f[col] = j[col].to_numpy()
    else:
        for col in cols:
            f[col] = np.nan

    d = _df(conn, """SELECT snap_time, least(bid_1, ask_1) AS depth_1pct FROM book_depth_5m WHERE symbol = %s
                     ORDER BY snap_time""", (perp_symbol(sym),))
    if not d.empty:
        j = pd.merge_asof(pd.DataFrame({"t": bar_close}), d, left_on="t", right_on="snap_time",
                          direction="backward", tolerance=pd.Timedelta(hours=1))
        f["depth_1pct"] = j["depth_1pct"].to_numpy()
    else:
        f["depth_1pct"] = np.nan

    daily = load_candles(conn, perp_symbol(sym), "1d", None, end)
    el = eligibility(daily)
    f["eligible"] = el.reindex(f.index.floor("D")).fillna(False).astype(bool).to_numpy() if len(el) else False
    f["symbol"] = sym
    return f


def build_panel(conn: psycopg.Connection, symbols: list[str], start=None, end=None) -> pd.DataFrame:
    parts = [_symbol_frame(conn, s, start, end) for s in symbols]
    df = pd.concat([p for p in parts if not p.empty])
    df.index.name = "time"
    return df.reset_index().set_index(["time", "symbol"]).sort_index()


def cached_panel(conn, symbols: list[str], tag: str = "panel_1h_v1") -> pd.DataFrame:
    path = CACHE / f"{tag}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = build_panel(conn, symbols)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def wide(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    """time x symbol matrix of one column."""
    return panel[col].unstack("symbol").sort_index()
