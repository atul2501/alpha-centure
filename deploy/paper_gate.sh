#!/usr/bin/env bash
# Starts the P6 paper engine only once every coin has the history P6 reads (p6.HISTORY_DAYS = 200 days of 1h
# candles, funding, premium index and open-interest metrics) and that data is current. On a fresh server the
# collector and alpha-vision fill these in a few hours; trading before that would decide on missing data.
# Run by alpha-paper-gate.service (root); checks every 5 minutes.
set -euo pipefail

while true; do
  # ON_ERROR_STOP: without it a SQL error (e.g. tables not created yet) exits 0 with empty output = "all ready"
  missing=$(psql "$DATABASE_URL" -Atq -v ON_ERROR_STOP=1 -v syms="$SYMBOLS" <<'SQL'
WITH s AS (SELECT unnest(string_to_array(:'syms', ',')) AS sym)
SELECT sym FROM s WHERE NOT coalesce(
  (SELECT min(open_time) <= now() - interval '200 days' AND max(open_time) >= now() - interval '3 hours'
     FROM candles WHERE symbol = sym || '.P' AND interval = '1h')
  AND (SELECT min(funding_time) <= now() - interval '200 days' AND max(funding_time) >= now() - interval '9 hours'
     FROM funding_rate WHERE symbol = sym)
  AND (SELECT min(open_time) <= now() - interval '200 days' AND max(open_time) >= now() - interval '3 hours'
     FROM premium_kline WHERE symbol = sym AND interval = '1h')
  AND (SELECT min(ts) <= now() - interval '200 days' AND max(ts) >= now() - interval '3 hours'
     FROM futures_metrics WHERE symbol = sym), false);
SQL
  ) || missing="(database not reachable yet)"
  if [ -z "$missing" ]; then
    echo "all coins ready: starting alpha-paper"
    systemctl enable --now alpha-paper.service
    exit 0
  fi
  echo "waiting for history: $(echo "$missing" | tr '\n' ' ')"
  sleep 300
done
