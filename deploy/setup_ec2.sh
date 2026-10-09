#!/usr/bin/env bash
# One-shot setup for Ubuntu 24.04 on EC2 (run as the ubuntu user with sudo, from the repo root):
#   bash deploy/setup_ec2.sh
# Region must NOT be in the US (Binance returns HTTP 451). Use ap-northeast-1 (Tokyo) or ap-south-1 (Mumbai).
set -euo pipefail
# Ubuntu's automatic security updates can hold the package lock for minutes: apt-get install waits up to 15 min for it.

APP_DIR=/opt/alpha
DB_NAME=alpha
DB_USER=alpha
DB_PASS=${DB_PASS:-$(openssl rand -hex 16)}

echo "==> PostgreSQL 16 + TimescaleDB"
sudo apt-get update
# libgomp1: OpenMP runtime that lightgbm/xgboost load at import (missing on minimal Ubuntu images)
sudo apt-get -o DPkg::Lock::Timeout=900 install -y gnupg curl lsb-release ca-certificates awscli chrony libgomp1
# websocket latency (ws_latency) = receive time - exchange time: the clock must be NTP-synced
sudo systemctl enable --now chrony
sudo install -d /usr/share/postgresql-common/pgdg
sudo curl -fsSL -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc https://www.postgresql.org/media/keys/ACCC4CF8.asc
echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" \
  | sudo tee /etc/apt/sources.list.d/pgdg.list
curl -fsSL https://packagecloud.io/timescale/timescaledb/gpgkey | sudo gpg --dearmor --yes -o /etc/apt/trusted.gpg.d/timescaledb.gpg
echo "deb https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/timescaledb.list
sudo apt-get update
sudo apt-get -o DPkg::Lock::Timeout=900 install -y postgresql-16 timescaledb-2-postgresql-16
sudo timescaledb-tune --quiet --yes
# 4 GB RAM, no swap by default: add 2 GB so a memory spike slows down instead of invoking the OOM killer
if [ ! -f /swapfile ]; then
  sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi
# If the OOM killer still takes one backend, the postmaster recovers by itself in under a second; systemd's default
# OOMPolicy=stop would instead shut the whole cluster down and leave it down.
sudo install -d /etc/systemd/system/postgresql@.service.d
printf '[Service]\nOOMPolicy=continue\n' | sudo tee /etc/systemd/system/postgresql@.service.d/oom.conf >/dev/null
sudo systemctl daemon-reload
sudo systemctl restart postgresql

echo "==> database + user (Postgres listens on localhost only by default)"
sudo -u postgres psql -v ON_ERROR_STOP=1 <<SQL
DO \$\$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${DB_USER}') THEN
    CREATE ROLE ${DB_USER} LOGIN PASSWORD '${DB_PASS}';
  END IF;
END \$\$;
SQL
sudo -u postgres createdb -O "$DB_USER" "$DB_NAME" 2>/dev/null || true
sudo -u postgres psql -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"

echo "==> app user, code, python env"
id alpha &>/dev/null || sudo useradd --system --create-home --shell /usr/sbin/nologin alpha
sudo mkdir -p "$APP_DIR"
# The ridge model trained locally must come along: without it the paper engine would train a new one at start-up on a
# database that is still nearly empty. models/p6_ridge_*.joblib is tracked in git, so a clone includes it.
MODEL=$(ls models/p6_ridge_*.joblib 2>/dev/null | sort | tail -n 1 || true)
if [ -z "$MODEL" ]; then
  echo "ERROR: models/p6_ridge_*.joblib not found (pull the latest main, or copy it from your machine):" >&2
  echo "  scp models/p6_ridge_20261001.joblib ubuntu@<ec2-ip>:<repo>/models/" >&2
  exit 1
fi
sudo rsync -a --delete --exclude .venv --exclude .git --exclude .env --exclude models --exclude data ./ "$APP_DIR"/
sudo mkdir -p "$APP_DIR/models"
sudo cp "$MODEL" "$APP_DIR/models/"
sudo chown -R alpha:alpha "$APP_DIR"
echo "    model: $(basename "$MODEL")"
sudo -u alpha bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
sudo -u alpha bash -c "cd $APP_DIR && ~/.local/bin/uv sync --no-dev"

echo "==> config"
sudo mkdir -p /etc/alpha
if [ ! -f /etc/alpha/.env ]; then
  # the role may survive from an earlier, interrupted run with another password: make it match the new .env
  sudo -u postgres psql -v ON_ERROR_STOP=1 -c "ALTER ROLE ${DB_USER} PASSWORD '${DB_PASS}';"
  sudo tee /etc/alpha/.env >/dev/null <<ENV
DATABASE_URL=postgresql://${DB_USER}:${DB_PASS}@localhost:5432/${DB_NAME}
# USD-M perpetuals only (spot off)
SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,SUIUSDT,TRXUSDT,AAVEUSDT,BNBUSDT,XRPUSDT,HYPEUSDT,LINKUSDT,ADAUSDT,UNIUSDT,LTCUSDT,AVAXUSDT,ATOMUSDT,DOGEUSDT,DOTUSDT,NEARUSDT,OPUSDT,ARBUSDT,WLDUSDT,CAKEUSDT,POLUSDT
SPOT_ENABLED=false
PERP_INTERVALS=1m,5m,15m,1h,4h,1d
BACKFILL_1M_DAYS=30
BACKFILL_START=2020-01-01
FUTURES_ENABLED=true
ORDERFLOW_ENABLED=true
BOOK_TICK_SECONDS=1
OI_POLL_SECONDS=60
MODELS_DIR=/opt/alpha/models
# S3_BACKUP_URI=s3://your-bucket/alpha-backups
ENV
  sudo chmod 640 /etc/alpha/.env
  sudo chown root:alpha /etc/alpha/.env
fi

echo "==> systemd"
sudo cp deploy/systemd/*.service deploy/systemd/*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now alpha-collector.service alpha-dashboard.service alpha-backup.timer
# History the strategy needs from data.binance.vision (5m-1d candles, premium, OI metrics, book depth; no 1m history).
# One-shot and resumable: re-run with `sudo systemctl start alpha-vision` if it is interrupted.
sudo systemctl start --no-block alpha-vision.service
# Mainnet PAPER trading of V4_carry (simulated fills; there is no real order path in the code). It is NOT started here:
# alpha-paper-gate starts it once every coin has 200 days of 1h candles, funding, premium and OI history
# (a few hours on a fresh server). Follow it with: journalctl -u alpha-paper-gate -f
sudo systemctl enable --now --no-block alpha-paper-gate.service
sudo -u alpha mkdir -p "$APP_DIR/models"

echo
echo "Done. Check: systemctl status alpha-collector; journalctl -u alpha-collector -f"
echo "Paper trading starts by itself when the history is ready: journalctl -u alpha-paper-gate -f"
echo "Dashboard: ssh -L 8501:localhost:8501 ubuntu@<ec2-ip> then open http://localhost:8501"
echo "History backfill: journalctl -u alpha-vision -f; coverage report:"
echo "  sudo -u alpha bash -c 'cd $APP_DIR && set -a && . /etc/alpha/.env && ~/.local/bin/uv run python -m alpha.binance.vision --report'"
