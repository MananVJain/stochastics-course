#!/usr/bin/env bash
# Starts Pluto exactly like the launcher does and checks that it answers HTTP.
# Keep the Pluto.run(...) arguments in sync with docker/pluto_proxy/pluto_launcher/__init__.py
set -euo pipefail

PORT=18888
LOG=/tmp/pluto-smoke.log

stop_pluto() {
  # kill the whole process group (Pluto + its workers), hard, no waiting
  kill -KILL -- "-$PID" 2>/dev/null || true
  pkill -KILL -f julia 2>/dev/null || true
}

setsid julia --startup-file=no -e "import Pluto; Pluto.run(host=\"127.0.0.1\", port=$PORT, launch_browser=false, require_secret_for_open_links=false, require_secret_for_access=false)" >"$LOG" 2>&1 </dev/null &
PID=$!
trap stop_pluto EXIT
START=$(date +%s)

for _ in $(seq 1 180); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/" >/dev/null 2>&1; then
    echo "Pluto answered after $(( $(date +%s) - START )) seconds"
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "Pluto exited early:"; cat "$LOG"; exit 1
  fi
  sleep 1
done

echo "Pluto did not answer within 180 seconds:"; cat "$LOG"
exit 1
