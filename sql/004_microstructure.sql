-- Phase 0: live perp microstructure for spread / slippage / latency calibration. Safe to re-run.
-- Perp flow and book reuse flow_1m / orderbook_snap under the perp name (BTCUSDT.P); rows under the plain
-- name are the old spot feed.

-- Top of book sampled every BOOK_TICK_SECONDS from depth20@100ms (perp, stored as BTCUSDT.P).
CREATE TABLE IF NOT EXISTS book_tick (
    symbol        text        NOT NULL,
    ts            timestamptz NOT NULL,              -- exchange event time, floored to the sample bucket
    best_bid      double precision NOT NULL,
    best_ask      double precision NOT NULL,
    bid_qty       double precision NOT NULL,         -- size at the best level, base asset
    ask_qty       double precision NOT NULL,
    bid_quote_20  double precision NOT NULL,         -- quote value resting on the 20 visible levels
    ask_quote_20  double precision NOT NULL,
    bid_reach_bps double precision NOT NULL,         -- distance from mid to the 20th level
    ask_reach_bps double precision NOT NULL,
    PRIMARY KEY (symbol, ts)
);

-- Current open interest, polled every OI_POLL_SECONDS (the 5m history endpoint only keeps ~30 days).
CREATE TABLE IF NOT EXISTS open_interest_live (
    symbol        text        NOT NULL,
    ts            timestamptz NOT NULL,              -- exchange time of the value
    open_interest double precision NOT NULL,         -- contracts (base asset)
    PRIMARY KEY (symbol, ts)
);

-- Websocket delivery latency per minute and stream kind: local receive time - exchange event time.
-- Includes local clock offset, so keep the host on NTP (chrony on EC2).
CREATE TABLE IF NOT EXISTS ws_latency (
    minute timestamptz NOT NULL,
    stream text        NOT NULL,                     -- aggTrade | depth20 | markPrice
    n      integer     NOT NULL,
    p50_ms double precision NOT NULL,
    p95_ms double precision NOT NULL,
    max_ms double precision NOT NULL,
    PRIMARY KEY (minute, stream)
);

DO $$
DECLARE
    t record;
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        RETURN;
    END IF;
    FOR t IN SELECT * FROM (VALUES
        ('book_tick',          'ts',     'symbol', interval '1 day'),
        ('open_interest_live', 'ts',     'symbol', interval '30 days'),
        ('ws_latency',         'minute', 'stream', interval '30 days')
    ) AS v(tbl, col, seg, chunk)
    LOOP
        PERFORM create_hypertable(t.tbl::regclass, t.col::name, chunk_time_interval => t.chunk,
                                  if_not_exists => true, migrate_data => true);
        IF NOT EXISTS (SELECT 1 FROM timescaledb_information.hypertables
                       WHERE hypertable_name = t.tbl AND compression_enabled) THEN
            EXECUTE format('ALTER TABLE %I SET (timescaledb.compress, timescaledb.compress_segmentby = %L)',
                           t.tbl, t.seg);
            PERFORM add_compression_policy(t.tbl::regclass, interval '7 days', if_not_exists => true);
        END IF;
    END LOOP;
END $$;
