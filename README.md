# Alpha Centure

A cost-aware crypto perpetual-futures research and paper-trading system. It collects live Binance **USD-M perpetual** data for
**BTC, ETH, SOL, SUI, TRX, AAVE, BNB, XRP, HYPE, LINK, ADA, UNI, LTC, AVAX** (all `…USDT`, spot is off) into
PostgreSQL/TimescaleDB, with an audit dashboard on top.

## Strategy on this branch: V4_C (V4_carry with slower carry: lower drawdowns and costs)

V4_C combines three books and trades the mix at full risk:

- **Ridge part:** a ridge regression turns 19 measurements per coin (1/2/4-week momentum, trend breakouts, funding,
  basis, open interest, long/short positioning, order flow, Bitcoin's trend) into a 72-hour forecast; retrained
  monthly. Positions are inverse-volatility weighted and scaled to a 20% yearly volatility target.
- **Rules part:** no trained model. Cross-sectional momentum (rank the coins by 1/2/4-week trend, long the strongest,
  short the weakest, dollar-neutral) plus time-series momentum (each coin with its own 2/4-week trend and 20-day
  breakout), 50/50, scaled to the same 20% target.
- **Carry part:** short the coins whose long holders pay the highest funding (24-hour average), long the coins with
  the lowest funding, dollar-neutral, scaled to the same 20% target. It earns funding and does not move with momentum.
  **Re-traded every 144h** (every second rebalance): funding moves slowly, so trading it every 72h mostly paid fees.
- **Mix:** 0.5 × ridge + 0.25 × rules + 0.25 × carry. Where the books agree the position is large, where they disagree
  it nets out. The three books diversify, so the mix is scaled back up to the 20% volatility target
  (max 3x gross, max 0.5x per coin, 1% no-trade band, 72h rebalance).

  | Coin (example) | Ridge | Rules | Carry | Mix before scaling |
  |---|---|---|---|---|
  | SOL | +$4,000 | +$6,000 | −$2,000 | +$3,000 |
  | DOGE | +$3,000 | −$3,000 | −$4,000 | −$250 |
  | LINK | −$2,000 | $0 | +$2,000 | −$500 |

Code: `book_weights` in `src/alpha/strategy/p6.py` (the module name is historical; it holds the ridge forecast).

### Replay, Jan 2021 → 7 Oct 2026 ($30,000 start, 23 coins, after fees, slippage and funding)

| Period | $30k → | Sharpe | Yearly swing | Worst drop | Worst month |
|---|---|---|---|---|---|
| Full 2021 → Oct 2026 | $223,314 | 1.59 | 23.6% | 17.4% | −9.2% |
| Jan 2021 – Jun 2024 | $97,095 | 1.58 | 22.9% | 17.4% | −9.2% |
| Jul 2024 – Sep 2025 | $44,513 | 1.42 | 24.3% | 12.5% | −5.7% |
| Oct 2025 – Oct 2026 | $46,502 | 1.84 | 25.0% | 11.3% | −4.4% |

By year: 2021 +44.0%, 2022 +0.4%, 2023 +63.9%, 2024 +73.1%, 2025 +18.9%, 2026 (to 7 Oct) +52.6%.
A $2,000 account with Binance's step sizes and minimum orders, 2022 → Oct 2026: $2,000 → $10,579, Sharpe 1.59,
worst drop 13.8% (V4_carry: $9,710, 1.55, 17.6%).
Same engine code as live, ridge retrained quarterly, frozen per-coin cost table (60% maker / 40% taker).

### Risks
- **Trades a lot:** about 56× equity a year (V4_carry about 70×). Check real paper fill costs early.
- **Weaker in older years:** Jan 2021 – Jun 2024 Sharpe 1.58 vs V4_carry's 1.88; its lead comes from mid-2024 on.
  Over 2021 → Oct 2026 the two are close (V4_carry $228.5k, Sharpe 1.64) and the gap is not statistically clear.
- **Not proven on unseen data:** chosen after looking at 2024–26. Treat paper results as the real test before any
  real money; V4_carry is the fallback (`git checkout v4-carry`).

### Deploy / revert
```bash
# on the server: switch the paper engine to V4_C (the paper account carries over; the next 72h rebalance moves it)
cd ~/alpha-centure && git fetch && git checkout v4-c && git pull
bash deploy/setup_ec2.sh
sudo systemctl disable --now alpha-league.timer 2>/dev/null; sudo rm -f /etc/systemd/system/alpha-league.*
sudo systemctl daemon-reload && sudo systemctl restart alpha-paper alpha-dashboard
# revert: git checkout v4-carry, then the same setup + restart
```

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
uv run python -m dashboard.server      # dashboard on http://localhost:8501 (workflow · ledger & P&L · market data)
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

The server builds its own database from Binance; only the code and the ridge model file are copied.

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
2. **Copy the repo** to the server (`git clone`, or copy the folder). The ridge model `models/p6_ridge_*.joblib` is in
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
# public instead: DASHBOARD_HOST=0.0.0.0 in /etc/alpha/.env (optional DASHBOARD_PASSWORD=...), your IP only in the security group
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
sql/                      schema, applied in order on every start (hypertables + compression when TimescaleDB exists)
src/alpha/binance/        REST client, websocket runner, payload parsers, data.binance.vision history loader
src/alpha/collectors/     klines, orderflow (aggTrade + depth), futures (poll + liquidations)
src/alpha/backfill.py     history load + gap repair
src/alpha/main.py         collector entrypoint
src/alpha/strategy/p6.py  V4_C strategy: ridge forecast + momentum rules + funding carry (144h) -> target weights (+ monthly retrain)
src/alpha/live/           paper engine (engine.py), paper vs backtest report (report.py), loss diagnosis (diagnose.py), dashboard checks
src/alpha/exec/           simulated orders, fills, costs, account ledger
src/alpha/research/       panel, signals, models, portfolio simulator, validation (VALID-A/B reproduction)
src/alpha/audit.py        data quality queries (dashboard)
dashboard/                web dashboard: JSON API (server.py, data.py) + one static page (static/); read-only
deploy/                   EC2 setup script, systemd units, S3 backup, paper start gate
```
