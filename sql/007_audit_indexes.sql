-- Dashboard audit queries filter recent candles by time across all symbols (OHLC sanity, 1m->5m resample).
-- The primary key leads with symbol, so without this index they scan every row (~34M after the history backfill).
CREATE INDEX IF NOT EXISTS candles_open_time_idx ON candles (open_time DESC);
