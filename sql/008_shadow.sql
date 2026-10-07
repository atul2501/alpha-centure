-- Shadow league (alpha.live.league): daily net per strategy from the research simulator. Never trades. Safe to re-run.

CREATE TABLE IF NOT EXISTS shadow_daily (
    day         date             NOT NULL,
    strategy    text             NOT NULL,
    gross       double precision NOT NULL,   -- all values: fraction of starting equity
    funding     double precision NOT NULL,   -- > 0 = paid
    cost        double precision NOT NULL,
    turnover    double precision NOT NULL,
    net         double precision NOT NULL,
    computed_at timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (day, strategy)
);
