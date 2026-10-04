-- Alpha Centure schema. Safe to re-run.
-- Uses TimescaleDB hypertables + compression when the extension is installed (EC2),
-- falls back to plain Postgres tables otherwise (e.g. local dev without timescale).

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb') THEN
        CREATE EXTENSION IF NOT EXISTS timescaledb;
    END IF;
END $$;

-- Exchange candles, closed only. Prices kept as numeric so they match Binance exactly.
CREATE TABLE IF NOT EXISTS candles (
    symbol          text        NOT NULL,
    interval        text        NOT NULL,
    open_time       timestamptz NOT NULL,
    close_time      timestamptz NOT NULL,
    open            numeric     NOT NULL,
    high            numeric     NOT NULL,
    low             numeric     NOT NULL,
    close           numeric     NOT NULL,
    volume          numeric     NOT NULL,
    quote_volume    numeric     NOT NULL,
    trades          integer     NOT NULL,
    taker_buy_base  numeric     NOT NULL,
    taker_buy_quote numeric     NOT NULL,
    source          text        NOT NULL,          -- 'ws' | 'rest'
    ingested_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, interval, open_time)
);
CREATE INDEX IF NOT EXISTS candles_ingested_idx ON candles (ingested_at DESC);

-- Futures: settled funding (every 8h)
CREATE TABLE IF NOT EXISTS funding_rate (
    symbol       text        NOT NULL,
    funding_time timestamptz NOT NULL,
    funding_rate double precision NOT NULL,
    mark_price   double precision,
    PRIMARY KEY (symbol, funding_time)
);

-- Futures: mark/index/predicted funding snapshot (every poll)
CREATE TABLE IF NOT EXISTS premium_snap (
    symbol            text        NOT NULL,
    ts                timestamptz NOT NULL,
    mark_price        double precision NOT NULL,
    index_price       double precision NOT NULL,
    last_funding_rate double precision NOT NULL,
    next_funding_time timestamptz,
    PRIMARY KEY (symbol, ts)
);

-- Futures 5m stats. Binance only serves the last ~30 days of these, so they must be recorded continuously.
CREATE TABLE IF NOT EXISTS open_interest (
    symbol                 text        NOT NULL,
    ts                     timestamptz NOT NULL,
    sum_open_interest      double precision NOT NULL,
    sum_open_interest_value double precision NOT NULL,
    PRIMARY KEY (symbol, ts)
);

CREATE TABLE IF NOT EXISTS long_short_ratio (
    symbol           text        NOT NULL,
    ts               timestamptz NOT NULL,
    long_short_ratio double precision NOT NULL,
    long_account     double precision NOT NULL,
    short_account    double precision NOT NULL,
    PRIMARY KEY (symbol, ts)
);

CREATE TABLE IF NOT EXISTS taker_ratio (
    symbol         text        NOT NULL,
    ts             timestamptz NOT NULL,
    buy_sell_ratio double precision NOT NULL,
    buy_vol        double precision NOT NULL,
    sell_vol       double precision NOT NULL,
    PRIMARY KEY (symbol, ts)
);

-- Futures liquidation orders (Binance pushes at most one per symbol per second).
CREATE TABLE IF NOT EXISTS liquidations (
    symbol     text        NOT NULL,
    ts         timestamptz NOT NULL,
    side       text        NOT NULL,   -- SELL = longs liquidated, BUY = shorts liquidated
    price      double precision NOT NULL,
    avg_price  double precision NOT NULL,
    qty        double precision NOT NULL,
    filled_qty double precision NOT NULL,
    status     text        NOT NULL,
    PRIMARY KEY (symbol, ts, side, price, qty)
);

-- Spot order flow per minute, built from aggTrade. complete=false when the stream was not
-- connected for the whole minute (startup / reconnect), so the model can drop those rows.
CREATE TABLE IF NOT EXISTS flow_1m (
    symbol     text        NOT NULL,
    minute     timestamptz NOT NULL,
    buy_vol    double precision NOT NULL,   -- aggressive buys (taker buy), base asset
    sell_vol   double precision NOT NULL,   -- aggressive sells, base asset
    delta      double precision NOT NULL,   -- buy_vol - sell_vol; cumsum = CVD
    buy_quote  double precision NOT NULL,
    sell_quote double precision NOT NULL,
    trades     integer     NOT NULL,
    max_trade_quote double precision NOT NULL,  -- largest single aggTrade in quote ccy (whale proxy)
    complete   boolean     NOT NULL,
    PRIMARY KEY (symbol, minute)
);

-- Spot order book top-20 snapshot, one per minute.
CREATE TABLE IF NOT EXISTS orderbook_snap (
    symbol     text        NOT NULL,
    ts         timestamptz NOT NULL,
    best_bid   double precision NOT NULL,
    best_ask   double precision NOT NULL,
    spread_bps double precision NOT NULL,
    bid_qty_20 double precision NOT NULL,
    ask_qty_20 double precision NOT NULL,
    imbalance  double precision NOT NULL,   -- (bid - ask) / (bid + ask), in [-1, 1]
    bids       jsonb       NOT NULL,
    asks       jsonb       NOT NULL,
    PRIMARY KEY (symbol, ts)
);

-- Audit trail of every write / connection event. Feeds the dashboard.
CREATE TABLE IF NOT EXISTS fetch_log (
    id         bigserial   PRIMARY KEY,
    fetched_at timestamptz NOT NULL DEFAULT now(),
    kind       text        NOT NULL,   -- candle | funding | premium | oi | ls_ratio | taker_ratio | liquidation | flow | orderbook | ws
    symbol     text,
    interval   text,
    ref_time   timestamptz,            -- time of the newest record written
    source     text        NOT NULL,   -- ws | rest | gap_repair
    rows       integer     NOT NULL DEFAULT 0,
    latency_ms bigint,                 -- fetched_at - record close time (large for history backfill)
    status     text        NOT NULL DEFAULT 'ok',
    error      text
);
CREATE INDEX IF NOT EXISTS fetch_log_fetched_idx ON fetch_log (fetched_at DESC);

DO $$
DECLARE
    t record;
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        RAISE NOTICE 'timescaledb not installed: using plain tables';
        RETURN;
    END IF;

    FOR t IN SELECT * FROM (VALUES
        ('candles',          'open_time',    'symbol, interval', interval '30 days'),
        ('funding_rate',     'funding_time', 'symbol',           interval '365 days'),
        ('premium_snap',     'ts',           'symbol',           interval '30 days'),
        ('open_interest',    'ts',           'symbol',           interval '30 days'),
        ('long_short_ratio', 'ts',           'symbol',           interval '30 days'),
        ('taker_ratio',      'ts',           'symbol',           interval '30 days'),
        ('liquidations',     'ts',           'symbol',           interval '30 days'),
        ('flow_1m',          'minute',       'symbol',           interval '30 days'),
        ('orderbook_snap',   'ts',           'symbol',           interval '7 days')
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
