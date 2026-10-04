-- Step 2.5: exit policy per signal (fixed | breakeven | trail | partial_trail). Safe to re-run.
ALTER TABLE signals ADD COLUMN IF NOT EXISTS exit_policy text NOT NULL DEFAULT 'fixed';
