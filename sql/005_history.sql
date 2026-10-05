-- Phase 0: history from data.binance.vision (USD-M perps). Safe to re-run.

-- 5m futures metrics (from ~Dec 2021). Kept apart from the live REST tables, which need buy/sell volumes the
-- dump does not carry. Values are published after their bucket closes: treat them as known at ts + 5 minutes.
CREATE TABLE IF NOT EXISTS futures_metrics (
    symbol                  text        NOT NULL,   -- plain symbol, like the other futures stats
    ts                      timestamptz NOT NULL,
    sum_open_interest       double precision,
    sum_open_interest_value double precision,
    toptrader_ls_count      double precision,       -- top traders long/short ratio (accounts)
    toptrader_ls_position   double precision,       -- top traders long/short ratio (positions)
    ls_ratio                double precision,       -- all accounts long/short ratio
    taker_ls_vol_ratio      double precision,       -- taker buy/sell volume ratio
    PRIMARY KEY (symbol, ts)
);

-- Book depth within +-1..5% of mid (bookDepth dump, ~2023 on), last snapshot of each 5-minute bucket.
-- Notional in quote currency, cumulative from mid outward.
CREATE TABLE IF NOT EXISTS book_depth_5m (
    symbol    text        NOT NULL,                 -- perp name (BTCUSDT.P)
    ts        timestamptz NOT NULL,                 -- bucket start; snapshot time is inside the bucket
    snap_time timestamptz NOT NULL,
    bid_1 double precision, bid_2 double precision, bid_3 double precision, bid_4 double precision, bid_5 double precision,
    ask_1 double precision, ask_2 double precision, ask_3 double precision, ask_4 double precision, ask_5 double precision,
    PRIMARY KEY (symbol, ts)
);

-- Premium index klines (perp vs index basis), from 2020.
CREATE TABLE IF NOT EXISTS premium_kline (
    symbol    text        NOT NULL,                 -- plain symbol
    interval  text        NOT NULL,
    open_time timestamptz NOT NULL,
    close_time timestamptz NOT NULL,
    open  double precision NOT NULL,
    high  double precision NOT NULL,
    low   double precision NOT NULL,
    close double precision NOT NULL,
    PRIMARY KEY (symbol, interval, open_time)
);

-- One row per dump file processed, so the backfill is resumable. rows = -1: file does not exist (404).
CREATE TABLE IF NOT EXISTS vision_files (
    path        text        PRIMARY KEY,
    dataset     text        NOT NULL,
    symbol      text        NOT NULL,
    period      text        NOT NULL,               -- YYYY-MM or YYYY-MM-DD
    rows        bigint      NOT NULL,
    ingested_at timestamptz NOT NULL DEFAULT now()
);

DO $$
DECLARE
    t record;
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        RETURN;
    END IF;
    FOR t IN SELECT * FROM (VALUES
        ('futures_metrics', 'ts',        'symbol',           interval '90 days'),
        ('book_depth_5m',   'ts',        'symbol',           interval '90 days'),
        ('premium_kline',   'open_time', 'symbol, interval', interval '90 days')
    ) AS v(tbl, col, seg, chunk)
    LOOP
        PERFORM create_hypertable(t.tbl::regclass, t.col::name, chunk_time_interval => t.chunk,
                                  if_not_exists => true, migrate_data => true);
        IF NOT EXISTS (SELECT 1 FROM timescaledb_information.hypertables
                       WHERE hypertable_name = t.tbl AND compression_enabled) THEN
            EXECUTE format('ALTER TABLE %I SET (timescaledb.compress, timescaledb.compress_segmentby = %L)',
                           t.tbl, t.seg);
            PERFORM add_compression_policy(t.tbl::regclass, interval '30 days', if_not_exists => true);
        END IF;
    END LOOP;
END $$;
