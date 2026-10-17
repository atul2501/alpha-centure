-- Strategy View: how each decision was built (signals -> sub-books -> blend -> final), strategy-agnostic JSON
-- (alpha.strategy.trace.StrategyTrace). Filled on REBALANCE and SNAPSHOT rows. Safe to re-run.
ALTER TABLE paper_decisions ADD COLUMN IF NOT EXISTS trace jsonb;
