# Alpha Centure

A cost-aware crypto perpetual-futures research and paper-trading system. It collects live Binance **USD-M perpetual** data for
**BTC, ETH, SOL, SUI, TRX, AAVE, BNB, XRP, HYPE, LINK, ADA, UNI, LTC, AVAX** (all `…USDT`, spot is off) into
PostgreSQL/TimescaleDB, with an audit dashboard on top.

## What gets collected

| Table | Content | Cadence |
|---|---|---|
| `candles` | Perp OHLCV (stored as `BTCUSDT.P`), quote volume, trades, taker-buy volume for 1m/5m/15m/1h/4h/1d. **Closed candles only.** | Live websocket + REST / data.binance.vision backfill |
| `flow_1m` | Perp aggressive buy/sell volume, delta (cumsum gives CVD), largest trade. `complete=false` marks partial minutes | Every minute (history: aggTrades dump) |
| `orderbook_snap` | Perp top-20 book: spread, bid/ask depth, imbalance, raw levels | Every minute |
| `book_tick` | Perp top of book + visible depth (quote value, reach in bps) | Every second |
| `open_interest_live` | Current open interest | Every minute |
| `ws_latency` | Websocket delivery latency p50/p95/max per stream kind | Every minute |
| `futures_metrics`, `book_depth_5m`, `premium_kline` | History from data.binance.vision: 5m OI + long/short + taker ratios, ±1–5% book depth, premium index klines | One-off backfill |
| `open_interest`, `long_short_ratio`, `taker_ratio` | Futures 5m stats. **Binance only keeps ~30 days, so collection must start early** | Every 5 min |
| `funding_rate`, `premium_snap` | Settled funding history, plus mark/index/predicted funding | Every 5 min |
| `liquidations` | Futures forced orders | Live websocket |
| `fetch_log` | Every write and connect/disconnect, with latency | Audit trail |

Reliability: missed candles are re-fetched over REST on startup, after every websocket reconnect, and every 30 minutes (gap repair). All writes are idempotent upserts.

## Run locally (Mac)

```bash
brew services start postgresql@16      # TimescaleDB optional locally; plain tables are used if missing
createdb alpha
cp .env.example .env                   # set DATABASE_URL=postgresql://localhost:5432/alpha
uv sync
uv run python -m alpha.main            # collector (backfills history first, then streams)
uv run streamlit run dashboard/app.py  # dashboard on http://localhost:8501
uv run python -m alpha.audit           # text audit report
uv run pytest                          # TEST_DATABASE_URL=postgresql://localhost/alpha_test for DB tests
```

`uv run python -m alpha.backfill` does a one-off history load and exits.

History from Binance's public dumps (resumable, skips files already loaded; `--report` prints coverage):

```bash
uv run python -m alpha.binance.vision                                     # 5m-1d klines, premium, metrics, book depth
uv run python -m alpha.binance.vision --datasets klines,aggTrades --intervals 1m   # large: run on EC2 (TimescaleDB)
uv run python -m alpha.binance.vision --report
```

Binance serves futures depth streams only on `wss://fstream.binance.com/public` and trades / mark price / klines only
on `/market`; the collector opens one connection to each.

## Deploy on EC2

1. Launch Ubuntu 24.04 in **ap-northeast-1 (Tokyo)** or **ap-south-1 (Mumbai)**. Binance blocks US regions with HTTP 451. `t4g.medium` with a **200 GB** gp3 disk (14 perps of 1m candles + 1m flow since 2020 ≈ 95M rows, ~15–20 GB before
TimescaleDB compression, plus 1s top-of-book going forward). An existing `/etc/alpha/.env` is not overwritten: update
`SYMBOLS`, `SPOT_ENABLED=false` and `PERP_INTERVALS=1m,5m,15m,1h,4h,1d` by hand.
2. Security group: only SSH (22) from your IP. Postgres and the dashboard stay on localhost.
3. Copy the repo to the instance and run `bash deploy/setup_ec2.sh`. It installs Postgres 16 + TimescaleDB, creates the DB and user, writes `/etc/alpha/.env`, and enables the systemd services.
4. Open the dashboard through a tunnel: `ssh -L 8501:localhost:8501 ubuntu@<ip>` → http://localhost:8501
5. Logs: `journalctl -u alpha-collector -f`
6. Backups: set `S3_BACKUP_URI` in `/etc/alpha/.env` and give the instance an IAM role with `s3:PutObject`. `alpha-backup.timer` runs daily at 02:30 UTC.

## Layout

```
sql/001_init.sql          schema (hypertables + compression when TimescaleDB exists)
src/alpha/binance/        REST client, websocket runner, pure payload parsers
src/alpha/collectors/     klines, orderflow (aggTrade + depth), futures (poll + liquidations)
src/alpha/backfill.py     history load + gap repair
src/alpha/audit.py        data quality queries (used by dashboard + CLI)
src/alpha/main.py         collector entrypoint
dashboard/app.py          Streamlit audit dashboard
deploy/                   EC2 setup script, systemd units, S3 backup
```

## Step 2: LONG / SHORT / PASS signals

Every time a perp candle (5m/15m/1h) closes, the predictor runs:

```
safety gate (stale data, wide spread, degraded model)        -> PASS
regime HMM on 4h (trend_up / trend_down / range / squeeze / extreme, forward-filtered: no look-ahead)
strategy playbook -> setups with entry, stop, target, time limit
    breakout · failed_breakout · trend_pullback · mean_reversion · vwap_reclaim · positioning_squeeze
regime gate: only (strategy, timeframe, regime) cells that made money after costs in training  -> else PASS
meta model (LightGBM): P(win) and expected R after fees/slippage/funding  -> below threshold PASS
one position per symbol, best expected R wins; long+short conflict -> PASS
```

Every candidate is stored in `signals` with its reason, and later scored (PASS rows too, so you can check whether passing was right).

| Command | What it does |
|---|---|
| `uv run python -m alpha.research.raw` | Raw strategy results (no ML), by strategy × timeframe / symbol |
| `uv run python -m alpha.backtest.walkforward --fold-months 3 --tfs 15m,1h` | Walk-forward backtest: refits everything per fold on past data only, compares raw / regime / regime+meta / shuffled-label sanity check |
| `uv run python -m alpha.trainer` | Weekly retrain: challenger vs champion, promotes only if better (`--dry-run`, `--force`) |
| `uv run python -m alpha.trainer --drift` | Daily health check: marks the model degraded (all PASS) on bad live results / feature drift |
| `uv run python -m alpha.predict` | Live predictor + scorer (needs a champion model). `--replay-hours 24` to evaluate recent bars offline |

Dashboard: the **Signals** page (sidebar) shows the champion model, last decisions with reasons, current regime, live results per strategy, and the "would-have" result of passed setups.

Notes:
- Trading is on USD-M perpetuals (`BTCUSDT.P` …); spot candles and futures data are used as context.
- Costs in all labels/backtests: 0.05% taker + 0.01% slippage per side, plus funding at 00/08/16 UTC.
- 5m setups need maker (limit) execution to beat costs; they stay gated off unless the backtest says otherwise.
- `tests/test_parity.py` guarantees live features equal research features for the same bar.

## Step 2.5: improvements (experiment log in `data/experiments/`)

All configs are compared walk-forward on 2020 → Sep 2025 (quarterly folds); **Oct 2025 → now is a frozen holdout** used once.
`uv run python -m alpha.research.experiment --list` shows every config; `--configs a,b --ref c` compares; `--final x` runs the holdout.

| Config | Trades | Avg R | Total R | Max DD | Note |
|---|---|---|---|---|---|
| baseline (original Step 2) | 626 | −0.015 | −9.2 | 23.6% | the earlier +30R did not survive a stricter purge |
| exits_breakeven | 519 | +0.028 | +14.4 | 25.8% | stop to entry after a close beyond +1R |
| exits_learn / exits_trail | 1,766 / 1,550 | +0.07 | +128 / +102 | 53% / 47% | big but fragile; collapses under careful gating |
| be_gates | 336 | +0.112 | +37.5 | 16.1% | + per-setup calibrated EV, cost cap, 3-window threshold |
| be_gates_roll730 | 328 | +0.144 | +47.1 | 10.3% | + train on the last 2 years only |
| **roll730_risk (production)** | 323 | **+0.148** | **+47.8** | **5.0%** | + EV sizing, max 2 same-side, −3R daily stop |
| holdout, roll730_risk | 13 | +0.151 | +2.0 | 1.1% | positive, but far too few trades to prove anything |

Production uses `PRODUCTION_CONFIG=roll730_risk` (15m + 1h, breakeven exits). Rejected: maker entries (+29.8R after gating),
recency half-life weighting (lower R), trailing exits (drawdown).
