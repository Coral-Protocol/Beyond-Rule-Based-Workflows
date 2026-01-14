#!/usr/bin/env bash
set -euo pipefail


PYTHON_BIN=python
ENTRY=run.py

RESTART_DELAY=3         
MAX_RESTARTS=0          


restart_count=0
PGID=""


cleanup() {
    local pgid="$1"
    [[ -z "$pgid" ]] && return

    echo "🧨 Cleaning up process group: $pgid"


    kill -TERM -"$pgid" 2>/dev/null || true
    sleep 2


    kill -KILL -"$pgid" 2>/dev/null || true
}


trap 'echo "🛑 Supervisor received signal, shutting down..."; cleanup "$PGID"; exit 0' SIGINT SIGTERM


while true; do
    restart_count=$((restart_count + 1))

    if [[ "$MAX_RESTARTS" -ne 0 && "$restart_count" -gt "$MAX_RESTARTS" ]]; then
        echo "❌ Reached max restarts ($MAX_RESTARTS), exiting supervisor."
        exit 1
    fi

    echo
    echo "========================================"
    echo "🚀 Launching run.py (attempt $restart_count)"
    echo "========================================"


    setsid bash -c "exec $PYTHON_BIN $ENTRY" &
    pid=$!
    PGID=$(ps -o pgid= "$pid" | tr -d ' ')

    echo "🔗 PID=$pid  PGID=$PGID"


    set +e
    wait "$pid"
    exit_code=$?
    set -e

    echo "💥 run.py exited with code $exit_code"


    if [[ "$exit_code" -eq 0 ]]; then
        echo "✅ run.py finished normally, supervisor exiting."
        exit 0
    fi


    echo "⚠️ run.py crashed or aborted, restarting..."
    cleanup "$PGID"

    echo "🔁 Restarting in $RESTART_DELAY seconds..."
    sleep "$RESTART_DELAY"
done
