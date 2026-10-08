# Alpha Centure

A cost-aware crypto perpetual-futures research and paper-trading system. It collects live Binance **USD-M perpetual** data for
**BTC, ETH, SOL, SUI, TRX, AAVE, BNB, XRP, HYPE, LINK, ADA, UNI, LTC, AVAX** (all `…USDT`, spot is off) into
PostgreSQL/TimescaleDB, with an audit dashboard on top.

## Strategy on this branch: V1_rules_heavy (research)

V1_rules_heavy leans on the untrained momentum rules and trades the mix at full risk:

- **Ridge part:** a ridge regression turns 19 measurements per coin (1/2/4-week momentum, trend breakouts, funding,
  basis, open interest, long/short positioning, order flow, Bitcoin's trend) into a 72-hour forecast; retrained
  monthly. Positions are inverse-volatility weighted and scaled to a 20% yearly volatility target.
- **Rules part:** no trained model. Cross-sectional momentum (rank the coins by 1/2/4-week trend, long the strongest,
  short the weakest, dollar-neutral) plus time-series momentum (each coin with its own 2/4-week trend and 20-day
  breakout), 50/50, scaled to the same 20% target.
- **Mix:** 0.3 × ridge + 0.7 × rules. Where both agree the position is large, where they disagree the rules mostly win.
  The two books diversify, so the mix is scaled back up to the 20% volatility target
  (max 3x gross, max 0.5x per coin, 1% no-trade band, 72h rebalance).

  | Coin (example) | Ridge | Rules | Mix before scaling |
  |---|---|---|---|
  | SOL | +$4,000 | +$6,000 | +$5,400 |
  | DOGE | +$3,000 | −$3,000 | −$1,200 |
  | LINK | −$2,000 | $0 | −$600 |

Code: `book_weights` in `src/alpha/strategy/p6.py` (the module name is historical; it holds the ridge forecast).

### Replay, Jan 2021 → 7 Oct 2026 ($30,000 start, 23 coins, after fees, slippage and funding)

| Period | $30k → | Sharpe | Yearly swing | Worst drop | Worst month |
|---|---|---|---|---|---|
| Full 2021 → Oct 2026 | $200,508 | 1.52 | 23.4% | 24.1% | −7.6% |
| Jan 2021 – Jun 2024 | $109,099 | 1.74 | 22.7% | 15.5% | −7.6% |
| Jul 2024 – Sep 2025 | $33,543 | 0.49 | 23.8% | 24.1% | −6.5% |
| Oct 2025 – Oct 2026 | $49,312 | 2.05 | 25.4% | 9.1% | −3.1% |

By year: 2021 +72.0%, 2022 −0.9%, 2023 +44.8%, 2024 +57.9%, 2025 +12.8%, 2026 (to 7 Oct) +52.1%.
Same engine code as live, ridge retrained quarterly, frozen per-coin cost table (60% maker / 40% taker).

### Risks
- **Research branch:** it did not pass the pre-registered selection rule on Jan 2021 – Jun 2024; its strong last
  12 months (Sharpe 2.05) came after a weak Jul 2024 – Sep 2025 (+11.8%, Sharpe 0.49). A strong recent stretch
  after a weak one is what overfitting often looks like.
- **Deep drawdowns:** 24.1% at worst (Jul 2024 – Sep 2025). The kill switch flattens at −30%.
- **Not proven on unseen data:** treat paper results as the real test before any real money.

### Deploy / revert
```bash
# on the server: switch the paper engine to V1_rules_heavy (the paper account carries over; the next 72h rebalance moves it)
cd ~/alpha-centure && git fetch && git checkout v1-rules-heavy && git pull
bash deploy/setup_ec2.sh
sudo systemctl disable --now alpha-league.timer 2>/dev/null; sudo rm -f /etc/systemd/system/alpha-league.*
sudo systemctl daemon-reload && sudo systemctl restart alpha-paper alpha-dashboard
# revert: git checkout main_v2, then the same setup + restart
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
src/alpha/strategy/p6.py  V1_rules_heavy strategy: ridge forecast (30%) + momentum rules (70%) -> target weights (+ monthly retrain)
src/alpha/live/           paper engine (engine.py), paper vs backtest report (report.py), dashboard checks
src/alpha/exec/           simulated orders, fills, costs, account ledger
src/alpha/research/       panel, signals, models, portfolio simulator, validation (VALID-A/B reproduction)
src/alpha/audit.py        data quality queries (dashboard)
dashboard/                Streamlit: live workflow + paper pages
deploy/                   EC2 setup script, systemd units, S3 backup, paper start gate
```
