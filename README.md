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

Run the collector + mainnet paper engine locally with one safe command (logs are appended to `logs/`, never wiped):

```bash
scripts/alphactl.sh start    # start whatever is not running
scripts/alphactl.sh status   # RUNNING / STOPPED
scripts/alphactl.sh logs     # follow both logs (Ctrl+C only stops watching)
scripts/alphactl.sh stop     # asks before stopping
```

History from Binance's public dumps (resumable, skips files already loaded; `--report` prints coverage):

```bash
uv run python -m alpha.binance.vision                                     # 5m-1d klines, premium, metrics, book depth
uv run python -m alpha.binance.vision --datasets klines,aggTrades --intervals 1m   # large: run on EC2 (TimescaleDB)
uv run python -m alpha.binance.vision --report
```

Binance serves futures depth streams only on `wss://fstream.binance.com/public` and trades / mark price / klines only
on `/market`; the collector opens one connection to each.

## Deploy on EC2 (paper trading)

The server builds its own database from Binance; only the code and the P6 model file are copied.

Quick start on the new server (details below):

```bash
git clone -b main_v1 https://github.com/atul2501/alpha-centure.git
cd alpha-centure
bash deploy/setup_ec2.sh
journalctl -u alpha-paper-gate -f     # wait for "starting alpha-paper"
```
Then stop the paper engine on your own machine.

1. **Launch** Ubuntu 24.04 in **Tokyo (ap-northeast-1)** or **Mumbai (ap-south-1)**. Binance blocks US regions
   (HTTP 451). Instance `t4g.medium`, **100 GB** gp3 disk (database ~13 GB at start; 1s top-of-book adds ~150 MB/day; 1m candles are kept for 30 days only). Security group: SSH (22) from your IP only; Postgres and the
   dashboard stay on localhost.
2. **Copy the repo** to the server (`git clone`, or copy the folder). The P6 model `models/p6_ridge_*.joblib` is in
   git, so it comes along; the rest of `models/` stays ignored.
3. **Run the setup** on the server, from the repo folder:
   ```bash
   bash deploy/setup_ec2.sh
   ```
   It installs Postgres 16 + TimescaleDB, creates the database, writes `/etc/alpha/.env`, copies the model (it stops
   with an error if the model file is missing) and starts the collector, dashboard, backups and history download.
4. **Wait for paper trading to start by itself.** `alpha-paper-gate` checks every 5 minutes that all 23 coins have
   200 days of 1h candles, funding, premium and open-interest history, then starts `alpha-paper` (a few hours):
   ```bash
   journalctl -u alpha-paper-gate -f     # prints the coins it is waiting for, then "starting alpha-paper"
   ```
5. **Stop the paper engine on your own machine** once AWS is trading, so only one paper account runs.

Day to day:
```bash
systemctl status alpha-collector alpha-paper     # running?
journalctl -u alpha-paper -f                     # live log
sudo systemctl restart alpha-paper               # after a code change
ssh -L 8501:localhost:8501 ubuntu@<ip>           # dashboard at http://localhost:8501
```
Backups: set `S3_BACKUP_URI` in `/etc/alpha/.env` and give the instance an IAM role with `s3:PutObject`
(`alpha-backup.timer`, daily 02:30 UTC). An existing `/etc/alpha/.env` is never overwritten.

### When it stops on its own

systemd restarts any crashed service within ~10 s and starts everything again after a reboot, so no extra
supervisor is needed. The engine itself stops or holds trading in these cases:

| Situation | What happens | What you do |
|---|---|---|
| Equity falls 30% from its peak (kill switch) | All positions closed, trading halted; stays halted across restarts | Review, then reset the `risk` row in `paper_state` by hand |
| Loss of 3% in one UTC day | Reduce-only for the rest of the day (no new exposure) | Nothing: resets next day |
| Order-book data stale (>10 s) for more than half the coins | That rebalance is skipped (logged as NO_TRADE) | Check `journalctl -u alpha-collector` / network |
| Instance stopped, disk full, or Binance unreachable | Collector and engine stop updating | Check `df -h`, AWS console, region |

Paper mode only: there is no real-order path in the code (`tests/test_no_real_orders.py`).

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
