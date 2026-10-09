"""Data quality checks. Used by the dashboard and runnable as: uv run python -m alpha.audit"""

import pandas as pd
import psycopg

from alpha.binance.parse import INTERVAL_MS
from alpha.config import get_settings


def _intervals_values(intervals: list[str]) -> str:
    return ", ".join(f"('{i}', interval '{INTERVAL_MS[i] // 1000} seconds')" for i in intervals)


def _df(conn: psycopg.Connection, sql: str, params: tuple | dict | None = None) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def last_candles(conn, n: int = 10, symbol: str | None = None, interval: str | None = None) -> pd.DataFrame:
    """The n most recently fetched candles (by ingest time): what the collector wrote last.

    candles is partitioned (and compressed) by open_time, not ingested_at: without the open_time bound the sort reads
    and decompresses every chunk since 2020, which got Postgres OOM-killed on the 4 GB EC2 box."""
    return _df(conn, """
        SELECT ingested_at, symbol, interval, open_time, close_time, open, high, low, close, volume,
               quote_volume, trades, taker_buy_base, source,
               round(extract(epoch FROM ingested_at - close_time)::numeric, 2) AS latency_s
        FROM candles
        WHERE open_time > now() - interval '3 days'
          AND (%(s)s::text IS NULL OR symbol = %(s)s) AND (%(i)s::text IS NULL OR interval = %(i)s)
        ORDER BY ingested_at DESC, open_time DESC
        LIMIT %(n)s
    """, {"s": symbol, "i": interval, "n": n})


def last_fetches(conn, n: int = 10, kind: str | None = None) -> pd.DataFrame:
    return _df(conn, """
        SELECT fetched_at, kind, symbol, interval, ref_time, source, rows, latency_ms, status, error
        FROM fetch_log
        WHERE (%(k)s::text IS NULL OR kind = %(k)s)
        ORDER BY id DESC LIMIT %(n)s
    """, {"k": kind, "n": n})


def _feeds_values(symbols: list[str], intervals: list[str]) -> str:
    return ", ".join(f"('{s}', '{i}', interval '{INTERVAL_MS[i] // 1000} seconds')" for s in symbols for i in intervals)


def candle_health(conn, intervals: list[str], symbols: list[str]) -> pd.DataFrame:
    """Latest closed candle per symbol x interval and whether it is overdue.

    Index lookups per feed (first / last row on the primary key), never a scan of the whole table: with years of
    history a full count took ~15 s. `bars` is the span in bars since the first candle (gaps are reported apart)."""
    return _df(conn, f"""
        WITH f(symbol, interval, step) AS (VALUES {_feeds_values(symbols, intervals)})
        SELECT f.symbol, f.interval, l.close_time AS last_close,
               (extract(epoch FROM l.open_time - fo.open_time) / extract(epoch FROM f.step))::bigint + 1 AS bars,
               fo.open_time AS first_open,
               (now() - l.close_time) > (f.step + interval '2 minutes') AS stale
        FROM f
        CROSS JOIN LATERAL (SELECT open_time, close_time FROM candles c
                            WHERE c.symbol = f.symbol AND c.interval = f.interval
                            ORDER BY open_time DESC LIMIT 1) l
        CROSS JOIN LATERAL (SELECT open_time FROM candles c
                            WHERE c.symbol = f.symbol AND c.interval = f.interval
                            ORDER BY open_time LIMIT 1) fo
        ORDER BY f.symbol, f.step
    """)


def candle_gaps(conn, intervals: list[str], symbols: list[str], days: int = 7) -> pd.DataFrame:
    """Missing candles inside the recent window (lookback is at least 20 bars per interval), per feed on the index."""
    return _df(conn, f"""
        WITH f(symbol, interval, step) AS (VALUES {_feeds_values(symbols, intervals)}),
        seq AS (
            SELECT f.symbol, f.interval, f.step, w.open_time,
                   lead(w.open_time) OVER (PARTITION BY f.symbol, f.interval ORDER BY w.open_time) AS next_open
            FROM f CROSS JOIN LATERAL (
                SELECT open_time FROM candles c
                WHERE c.symbol = f.symbol AND c.interval = f.interval
                  AND c.open_time > now() - greatest(make_interval(days => %(d)s), f.step * 20)) w
        )
        SELECT symbol, interval, open_time AS gap_after, next_open AS gap_before,
               (extract(epoch FROM next_open - open_time) / extract(epoch FROM step))::int - 1 AS missing
        FROM seq WHERE next_open - open_time > step
        ORDER BY gap_after DESC
    """, {"d": days})


def ohlc_violations(conn, hours: int = 24) -> pd.DataFrame:
    return _df(conn, """
        SELECT symbol, interval, open_time, open, high, low, close, volume
        FROM candles
        WHERE open_time > now() - make_interval(hours => %(h)s)
          AND (low > least(open, close) OR high < greatest(open, close) OR low > high
               OR volume < 0 OR taker_buy_base > volume OR close_time <= open_time)
        ORDER BY open_time DESC LIMIT 100
    """, {"h": hours})


def resample_mismatch(conn, hours: int = 24) -> pd.DataFrame:
    """Rebuild 5m candles from 1m and compare with Binance's native 5m candles."""
    return _df(conn, """
        WITH agg AS (
            SELECT symbol, date_bin('5 minutes', open_time, timestamptz '2000-01-01') AS bucket,
                   (array_agg(open ORDER BY open_time))[1] AS open,
                   max(high) AS high, min(low) AS low,
                   (array_agg(close ORDER BY open_time DESC))[1] AS close,
                   sum(volume) AS volume, count(*) AS n
            FROM candles
            WHERE interval = '1m' AND open_time > now() - make_interval(hours => %(h)s)
            GROUP BY 1, 2
        )
        SELECT a.symbol, a.bucket, a.open, f.open AS open_5m, a.high, f.high AS high_5m,
               a.low, f.low AS low_5m, a.close, f.close AS close_5m, a.volume, f.volume AS volume_5m
        FROM agg a JOIN candles f
          ON f.symbol = a.symbol AND f.interval = '5m' AND f.open_time = a.bucket
        WHERE a.n = 5 AND (a.open <> f.open OR a.high <> f.high OR a.low <> f.low OR a.close <> f.close
                           OR abs(a.volume - f.volume) > 1e-6)
        ORDER BY a.bucket DESC LIMIT 100
    """, {"h": hours})


def feed_health(conn) -> pd.DataFrame:
    """Latest record per symbol for the non-candle feeds."""
    return _df(conn, """
        SELECT 'flow_1m' AS feed, symbol, max(minute) AS last_ts, count(*) AS rows FROM flow_1m GROUP BY symbol
        UNION ALL SELECT 'orderbook_snap', symbol, max(ts), count(*) FROM orderbook_snap GROUP BY symbol
        UNION ALL SELECT 'open_interest', symbol, max(ts), count(*) FROM open_interest GROUP BY symbol
        UNION ALL SELECT 'long_short_ratio', symbol, max(ts), count(*) FROM long_short_ratio GROUP BY symbol
        UNION ALL SELECT 'taker_ratio', symbol, max(ts), count(*) FROM taker_ratio GROUP BY symbol
        UNION ALL SELECT 'funding_rate', symbol, max(funding_time), count(*) FROM funding_rate GROUP BY symbol
        UNION ALL SELECT 'premium_snap', symbol, max(ts), count(*) FROM premium_snap GROUP BY symbol
        UNION ALL SELECT 'liquidations', symbol, max(ts), count(*) FROM liquidations GROUP BY symbol
        ORDER BY feed, symbol
    """)


def ws_events(conn, hours: int = 24) -> pd.DataFrame:
    return _df(conn, """
        SELECT interval AS stream, status, count(*) AS events, max(fetched_at) AS last_event
        FROM fetch_log WHERE kind = 'ws' AND fetched_at > now() - make_interval(hours => %(h)s)
        GROUP BY 1, 2 ORDER BY 1, 2
    """, {"h": hours})


def main() -> None:
    s = get_settings()
    feeds = list(dict.fromkeys(f.db_symbol for f in s.candle_feeds()))
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    with psycopg.connect(s.database_url) as conn:
        for title, df in [
            ("Candle health", candle_health(conn, s.all_intervals, feeds)),
            ("Gaps (7d)", candle_gaps(conn, s.all_intervals, feeds)),
            ("OHLC violations (24h)", ohlc_violations(conn)),
            ("1m->5m resample mismatches (24h)", resample_mismatch(conn)),
            ("Other feeds", feed_health(conn)),
            ("Websocket events (24h)", ws_events(conn)),
            ("Last 10 candles fetched", last_candles(conn)),
        ]:
            print(f"\n== {title} ({len(df)} rows) ==")
            print(df.to_string(index=False) if len(df) else "OK / none")


if __name__ == "__main__":
    main()
