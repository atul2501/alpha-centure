#!/usr/bin/env bash
# Safe control for the local collector + paper engine.
#   scripts/alphactl.sh start     start whatever is not running (logs are appended, never wiped)
#   scripts/alphactl.sh status    show what is running
#   scripts/alphactl.sh logs      follow both logs (Ctrl+C stops watching, not the programs)
#   scripts/alphactl.sh stop      stop both (asks first)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
running() { pgrep -f "python -m $1" >/dev/null 2>&1; }
start_one() {
  local mod=$1 log=$2
  if running "$mod"; then echo "already running: $mod"; else
    echo "--- start $(date)" >> "logs/$log"
    nohup uv run python -m "$mod" >> "logs/$log" 2>&1 &
    echo "started: $mod (log: logs/$log)"
  fi
}
case "${1:-status}" in
  start)  start_one alpha.main collector.log; sleep 5; start_one alpha.live.engine paper.log ;;
  status) for m in alpha.main alpha.live.engine; do running "$m" && echo "RUNNING  $m" || echo "STOPPED  $m"; done ;;
  logs)   tail -n 20 -f logs/collector.log logs/paper.log ;;
  stop)   read -r -p "Stop the collector and the paper engine? [y/N] " a
          if [[ "$a" == "y" || "$a" == "Y" ]]; then pkill -f "python -m alpha.live.engine" || true
            pkill -f "python -m alpha.main" || true; echo "stopped"; else echo "nothing stopped"; fi ;;
  *) echo "usage: $0 start|status|logs|stop"; exit 1 ;;
esac
