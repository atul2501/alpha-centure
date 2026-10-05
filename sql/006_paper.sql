-- Phase 5: mainnet paper-trading ledger. Every decision, order, fill, funding payment and equity mark. Safe to re-run.

CREATE TABLE IF NOT EXISTS paper_decisions (
    id         bigserial   PRIMARY KEY,
    ts         timestamptz NOT NULL DEFAULT now(),
    bar_time   timestamptz NOT NULL,               -- open time of the last closed 1h bar the decision used
    model      text        NOT NULL,
    action     text        NOT NULL,               -- REBALANCE | HOLD | NO_TRADE | FLATTEN
    reason     text        NOT NULL,
    equity     double precision NOT NULL,
    targets    jsonb,                              -- symbol -> target weight
    current    jsonb                               -- symbol -> weight before acting
);

CREATE TABLE IF NOT EXISTS paper_orders (
    id          bigserial   PRIMARY KEY,
    decision_id bigint      REFERENCES paper_decisions(id),
    ts          timestamptz NOT NULL DEFAULT now(),
    symbol      text        NOT NULL,
    side        smallint    NOT NULL,              -- +1 buy, -1 sell
    kind        text        NOT NULL,              -- maker | taker
    price       double precision,                  -- limit price (maker)
    qty         double precision NOT NULL,
    filled      double precision NOT NULL DEFAULT 0,
    status      text        NOT NULL,              -- open | filled | expired | cancelled
    arrival_mid double precision NOT NULL,         -- mid when the order was decided (slippage reference)
    queue_ahead double precision,
    closed_at   timestamptz
);

CREATE TABLE IF NOT EXISTS paper_fills (
    id           bigserial   PRIMARY KEY,
    order_id     bigint      NOT NULL REFERENCES paper_orders(id),
    ts           timestamptz NOT NULL DEFAULT now(),
    symbol       text        NOT NULL,
    side         smallint    NOT NULL,
    price        double precision NOT NULL,
    qty          double precision NOT NULL,
    liquidity    text        NOT NULL,             -- maker | taker
    fee          double precision NOT NULL,
    arrival_mid  double precision NOT NULL,
    slippage_bps double precision NOT NULL,        -- side * (price / arrival_mid - 1), > 0 = worse than mid
    latency_ms   double precision,                 -- taker: simulated decision -> exchange delay used
    beyond_book  boolean     NOT NULL DEFAULT false
);

CREATE TABLE IF NOT EXISTS paper_funding (
    id      bigserial   PRIMARY KEY,
    ts      timestamptz NOT NULL,                  -- settlement time
    symbol  text        NOT NULL,
    qty     double precision NOT NULL,
    mark    double precision NOT NULL,
    rate    double precision NOT NULL,
    amount  double precision NOT NULL,             -- paid > 0, received < 0
    UNIQUE (symbol, ts)
);

CREATE TABLE IF NOT EXISTS paper_equity (
    ts           timestamptz PRIMARY KEY,
    equity       double precision NOT NULL,
    cash         double precision NOT NULL,
    unrealized   double precision NOT NULL,
    gross_lev    double precision NOT NULL,
    net_lev      double precision NOT NULL,
    margin_ratio double precision NOT NULL,
    fees         double precision NOT NULL,        -- cumulative
    funding      double precision NOT NULL,        -- cumulative paid
    positions    jsonb       NOT NULL              -- symbol -> {qty, entry, mark}
);

CREATE TABLE IF NOT EXISTS paper_state (
    key   text PRIMARY KEY,
    value jsonb NOT NULL
);
