-- SOL 15m model tournament (alpha.tournament). Own schema, separate from the live model_registry / signals tables.
-- Safe to re-run (the collector applies every sql/*.sql at start-up).

CREATE SCHEMA IF NOT EXISTS tournament;

CREATE TABLE IF NOT EXISTS tournament.datasets (
    dataset_id   text        PRIMARY KEY,               -- e.g. sol15_v1_<hash12>
    created_at   timestamptz NOT NULL DEFAULT now(),
    symbol       text        NOT NULL,
    tf           text        NOT NULL,
    start_ts     timestamptz NOT NULL,
    end_ts       timestamptz NOT NULL,                  -- exclusive; never later than DEV_END outside shadow mode
    n_rows       integer     NOT NULL,
    content_hash text        NOT NULL,
    path         text        NOT NULL,
    coverage     jsonb       NOT NULL                   -- feature group -> first/last non-null bar, share non-null
);

CREATE TABLE IF NOT EXISTS tournament.feature_sets (
    feature_set_id text        PRIMARY KEY,             -- <name>_<hash8>
    name           text        NOT NULL,
    groups         text[]      NOT NULL,
    columns        text[]      NOT NULL,
    eval_start     timestamptz,                         -- first test bar for which every group has history
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tournament.models (
    model_id    text PRIMARY KEY,                       -- registry name, e.g. lightgbm, hsmm_lightgbm
    family      text NOT NULL,                          -- rules | classical | regime | linear | tree | deep | transformer
                                                        -- | ssm | foundation | rl | hybrid | meta | ensemble
    complexity  smallint NOT NULL,                      -- 0 rules .. 4 transformer/SSM/foundation/RL/ensemble
    input_kind  text NOT NULL,                          -- tabular | sequence | series | members
    experimental boolean NOT NULL DEFAULT false,
    description text NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS tournament.model_versions (
    model_version_id text        PRIMARY KEY,           -- <model_id>@<code_hash8>
    model_id         text        NOT NULL REFERENCES tournament.models,
    code_hash        text        NOT NULL,
    created_at       timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tournament.hyperparameters (
    hp_id    text  PRIMARY KEY,                         -- sha of the canonical JSON
    model_id text  NOT NULL,
    params   jsonb NOT NULL
);

CREATE TABLE IF NOT EXISTS tournament.experiments (
    experiment_id    text        PRIMARY KEY,           -- <stage>.<entry>.<config_hash10>
    stage            text        NOT NULL,
    entry            text        NOT NULL,              -- config entry name (model + variant)
    model_id         text        NOT NULL REFERENCES tournament.models,
    model_version_id text,
    dataset_id       text        NOT NULL REFERENCES tournament.datasets,
    feature_set_id   text        REFERENCES tournament.feature_sets,
    grid             jsonb       NOT NULL,              -- candidate configs; one is picked per fold on validation only
    n_configs        integer     NOT NULL,              -- counted as trials for the deflated Sharpe
    cost_config      jsonb       NOT NULL,
    seed             integer     NOT NULL,
    git_commit       text        NOT NULL,
    git_dirty_hash   text,                              -- sha of `git diff HEAD` + untracked tournament files (NULL = clean)
    config_hash      text        NOT NULL,
    oos_start        timestamptz,
    oos_end          timestamptz,
    status           text        NOT NULL,              -- queued | running | done | failed | gated_out | not_run
    status_reason    text        NOT NULL DEFAULT '',
    shuffled_control boolean     NOT NULL DEFAULT false,
    parent_id        text,                              -- robustness / control runs point at their base experiment
    created_at       timestamptz NOT NULL DEFAULT now(),
    finished_at      timestamptz
);
CREATE INDEX IF NOT EXISTS experiments_stage_idx ON tournament.experiments (stage, status);

CREATE TABLE IF NOT EXISTS tournament.training_runs (
    experiment_id  text        NOT NULL REFERENCES tournament.experiments ON DELETE CASCADE,
    fold           smallint    NOT NULL,
    config_idx     smallint    NOT NULL,
    train_start    timestamptz NOT NULL,
    train_end      timestamptz NOT NULL,
    n_train        integer     NOT NULL,
    fit_seconds    double precision,
    predict_seconds double precision,
    cpu_percent    double precision,
    rss_mb         double precision,
    gpu_mb         double precision,                    -- MPS allocated memory (Apple GPU), NULL on CPU
    model_bytes    bigint,
    PRIMARY KEY (experiment_id, fold, config_idx)
);

CREATE TABLE IF NOT EXISTS tournament.validation_runs (
    experiment_id text        NOT NULL REFERENCES tournament.experiments ON DELETE CASCADE,
    fold          smallint    NOT NULL,
    config_idx    smallint    NOT NULL,
    threshold_idx smallint    NOT NULL,
    val_start     timestamptz NOT NULL,
    val_end       timestamptz NOT NULL,
    val_net_bps   double precision,                     -- sum of net bps on validation (selection objective)
    val_trades    integer,
    chosen        boolean     NOT NULL DEFAULT false,
    PRIMARY KEY (experiment_id, fold, config_idx, threshold_idx)
);

CREATE TABLE IF NOT EXISTS tournament.walk_forward_runs (
    experiment_id text        NOT NULL REFERENCES tournament.experiments ON DELETE CASCADE,
    fold          smallint    NOT NULL,
    test_start    timestamptz NOT NULL,
    test_end      timestamptz NOT NULL,
    config_idx    smallint,                             -- NULL when the model abstained (no config positive on validation)
    threshold     double precision,
    abstained     boolean     NOT NULL,
    trades        integer     NOT NULL,
    gross_bps     double precision,
    net_bps       double precision,
    forced_trades integer,                              -- diagnostic: best validation config even when it was negative
    forced_gross_bps double precision,
    forced_net_bps   double precision,
    PRIMARY KEY (experiment_id, fold)
);

CREATE TABLE IF NOT EXISTS tournament.oos_runs (
    experiment_id text        NOT NULL REFERENCES tournament.experiments ON DELETE CASCADE,
    variant       text        NOT NULL,                 -- policy | forced | cost_x1.5 | slip_x2 | ...
    oos_start     timestamptz NOT NULL,
    oos_end       timestamptz NOT NULL,
    summary       jsonb       NOT NULL,                 -- scorecard Card.summary()
    passes        boolean,
    failed        text,
    PRIMARY KEY (experiment_id, variant)
);

CREATE TABLE IF NOT EXISTS tournament.predictions (
    experiment_id text        NOT NULL,
    fold          smallint    NOT NULL,
    bar_ts        timestamptz NOT NULL,                 -- bar open time; known at bar close
    score         real,                                 -- model score (expected fwd return or p_long - p_short)
    p_short       real,
    p_flat        real,
    p_long        real,
    target_h      smallint
);
CREATE INDEX IF NOT EXISTS predictions_exp_idx ON tournament.predictions (experiment_id);
CREATE INDEX IF NOT EXISTS predictions_ts_brin ON tournament.predictions USING brin (bar_ts);

CREATE TABLE IF NOT EXISTS tournament.signals (
    experiment_id text        NOT NULL,
    variant       text        NOT NULL,
    bar_ts        timestamptz NOT NULL,
    position      smallint    NOT NULL,                 -- target position from this bar's close: -1 / 0 / +1
    PRIMARY KEY (experiment_id, variant, bar_ts)
);

CREATE TABLE IF NOT EXISTS tournament.trades (
    experiment_id text        NOT NULL,
    variant       text        NOT NULL,
    trade_no      integer     NOT NULL,
    entry_ts      timestamptz NOT NULL,
    exit_ts       timestamptz NOT NULL,
    side          smallint    NOT NULL,
    bars          integer     NOT NULL,
    gross_bps     double precision NOT NULL,
    fee_bps       double precision NOT NULL,
    slip_bps      double precision NOT NULL,            -- spread + impact + latency
    funding_bps   double precision NOT NULL,
    net_bps       double precision NOT NULL,
    regime        jsonb,                                -- evaluation regime labels at entry
    fold          smallint,
    PRIMARY KEY (experiment_id, variant, trade_no)
);
CREATE INDEX IF NOT EXISTS trades_exp_idx ON tournament.trades (experiment_id, variant);

CREATE TABLE IF NOT EXISTS tournament.performance_metrics (
    experiment_id text NOT NULL,
    variant       text NOT NULL,
    scope         text NOT NULL,                        -- all | fold | regime | side | month | week | year | stress | mc | stat
    bucket        text NOT NULL,                        -- e.g. '3', 'bull', 'long', '2023-05', 'cost_x2'
    metric        text NOT NULL,
    value         double precision,
    PRIMARY KEY (experiment_id, variant, scope, bucket, metric)
);

CREATE TABLE IF NOT EXISTS tournament.feature_importance (
    experiment_id text NOT NULL,
    fold          smallint NOT NULL,
    feature       text NOT NULL,
    importance    double precision NOT NULL,
    kind          text NOT NULL,                        -- gain | coef | permutation
    PRIMARY KEY (experiment_id, fold, feature, kind)
);

CREATE TABLE IF NOT EXISTS tournament.regimes (
    dataset_id text        NOT NULL,
    bar_ts     timestamptz NOT NULL,
    trend      text,                                    -- bull | bear | sideways
    vol        text,                                    -- high_vol | low_vol
    volume     text,                                    -- high_volume | low_volume
    character  text,                                    -- trending | mean_reverting
    PRIMARY KEY (dataset_id, bar_ts)
);

CREATE TABLE IF NOT EXISTS tournament.ensemble_members (
    experiment_id        text NOT NULL,                 -- the ensemble / stack / meta experiment
    member_experiment_id text NOT NULL,
    role                 text NOT NULL,                 -- member | base | meta
    weight               double precision,
    PRIMARY KEY (experiment_id, member_experiment_id)
);

CREATE TABLE IF NOT EXISTS tournament.experiment_artifacts (
    experiment_id text        NOT NULL,
    kind          text        NOT NULL,                 -- model | config | patch | equity_png | json
    path          text        NOT NULL,
    bytes         bigint,
    created_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (experiment_id, kind, path)
);

-- Status history; the current status of a model version is its latest row.
CREATE TABLE IF NOT EXISTS tournament.registry (
    id            bigserial   PRIMARY KEY,
    experiment_id text        NOT NULL,
    status        text        NOT NULL CHECK (status IN ('EXPERIMENTAL', 'VALIDATING', 'PASSED', 'REJECTED',
                                                         'PAPER', 'PRODUCTION', 'RETIRED')),
    reason        text        NOT NULL DEFAULT '',
    changed_by    text        NOT NULL DEFAULT 'tournament',
    changed_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS registry_exp_idx ON tournament.registry (experiment_id, changed_at DESC);
