-- Coin candidates (alpha.coins): check -> shadow -> live, requested from the dashboard. Safe to re-run.
CREATE TABLE IF NOT EXISTS coin_candidates (
    symbol        text PRIMARY KEY,
    status        text NOT NULL CHECK (status IN ('queued', 'checking', 'checked', 'shadow', 'live', 'rejected')),
    verdict       text CHECK (verdict IN ('good', 'weak', 'reject')),
    requested_at  timestamptz NOT NULL DEFAULT now(),
    checked_at    timestamptz,
    shadow_start  timestamptz,
    promoted_at   timestamptz,
    report        jsonb,
    shadow        jsonb,
    error         text
);
-- Audit log: every request, check, decision and shadow update.
CREATE TABLE IF NOT EXISTS coin_events (
    id      bigserial PRIMARY KEY,
    ts      timestamptz NOT NULL DEFAULT now(),
    symbol  text NOT NULL,
    action  text NOT NULL,
    detail  jsonb NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS coin_events_symbol_ts ON coin_events (symbol, ts);
-- Rejected coins keep their report; their downloaded history is deleted after alpha.coins.PURGE_AFTER_DAYS.
ALTER TABLE coin_candidates ADD COLUMN IF NOT EXISTS purged_at timestamptz;
