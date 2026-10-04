-- Step 2: model registry + live decisions. Safe to re-run.

CREATE TABLE IF NOT EXISTS model_registry (
    version     text        PRIMARY KEY,               -- e.g. 20261004T221500Z
    created_at  timestamptz NOT NULL DEFAULT now(),
    status      text        NOT NULL,                  -- champion | challenger | retired | degraded
    train_end   timestamptz NOT NULL,
    path        text        NOT NULL,                  -- directory with regime.joblib, meta.joblib, policy.json
    policy      jsonb       NOT NULL,
    metrics     jsonb       NOT NULL,                  -- walk-forward / out-of-sample evaluation
    note        text
);

-- One row per candidate setup per closed bar (taken or passed), plus a 'none' row when nothing fired.
-- Outcome columns are filled by the scorer once the setup's time limit has passed, for PASS rows too
-- (counterfactual: would this passed setup have won?).
CREATE TABLE IF NOT EXISTS signals (
    symbol        text        NOT NULL,
    tf            text        NOT NULL,
    bar_time      timestamptz NOT NULL,                -- signal bar open time
    strategy      text        NOT NULL,                -- 'none' when no setup fired
    side          smallint    NOT NULL,                -- +1 / -1 / 0
    model_version text        NOT NULL,
    action        text        NOT NULL,                -- LONG | SHORT | PASS
    reason        text        NOT NULL,                -- '' when traded, else the PASS reason
    regime        text,
    regime_probs  jsonb,
    p_win         double precision,
    ev_r          double precision,
    close_px      double precision,                    -- signal bar close
    stop          double precision,
    target        double precision,
    max_bars      integer,
    features      jsonb,
    created_at    timestamptz NOT NULL DEFAULT now(),
    outcome       text,                                -- target | stop | timeout
    entry         double precision,
    exit          double precision,
    exit_time     timestamptz,
    r             double precision,
    pnl           double precision,
    scored_at     timestamptz,
    PRIMARY KEY (symbol, tf, bar_time, strategy, side, model_version)
);
CREATE INDEX IF NOT EXISTS signals_created_idx ON signals (created_at DESC);
CREATE INDEX IF NOT EXISTS signals_unscored_idx ON signals (bar_time) WHERE scored_at IS NULL AND strategy <> 'none';
