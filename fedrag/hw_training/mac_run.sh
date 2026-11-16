#!/usr/bin/env bash
# Run one federated round on the Mac under an RSS/time watchdog.
#
# Copied to the Mac by training/fed/coordinator.py; the coordinator starts it with
# `ssh mac env FED_PYTHON=... FED_RSS_LIMIT_GB=... FED_TIMEOUT_S=... bash mac_run.sh ...`.
#
# Every 2 s the watchdog polls the client's resident size. If it passes
# FED_RSS_LIMIT_GB (default 13) or the round outlives FED_TIMEOUT_S (default
# 3600), the client is killed and this script exits non-zero with a clear reason,
# so the coordinator aborts the federated run instead of averaging a half round.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${FED_PYTHON:-$HOME/opt/fedtrain/.venv/bin/python}"
RSS_LIMIT_GB="${FED_RSS_LIMIT_GB:-13}"
TIMEOUT_S="${FED_TIMEOUT_S:-3600}"
LOG="${FED_LOG:-$SCRIPT_DIR/client.log}"
LIMIT_KB="$(awk -v gigabytes="$RSS_LIMIT_GB" 'BEGIN { printf "%d", gigabytes * 1024 * 1024 }')"

if [ ! -x "$PYTHON" ]; then
    echo "fed mac watchdog: no python at $PYTHON (set FED_PYTHON)" >&2
    exit 2
fi

echo "fed mac watchdog: rss_limit=${RSS_LIMIT_GB}GB timeout=${TIMEOUT_S}s log=$LOG" >&2
"$PYTHON" "$SCRIPT_DIR/client.py" "$@" >"$LOG" 2>&1 &
child=$!
started=$SECONDS

while kill -0 "$child" 2>/dev/null; do
    rss="$(ps -o rss= -p "$child" 2>/dev/null | tr -d ' ' || true)"
    if [ -n "${rss:-}" ] && [ "$rss" -gt "$LIMIT_KB" ]; then
        kill -9 "$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
        echo "fed mac watchdog: client RSS ${rss}KB exceeded ${LIMIT_KB}KB (${RSS_LIMIT_GB}GB); killed" >&2
        exit 3
    fi
    if [ $(( SECONDS - started )) -gt "$TIMEOUT_S" ]; then
        kill -9 "$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
        echo "fed mac watchdog: client exceeded ${TIMEOUT_S}s; killed" >&2
        exit 4
    fi
    sleep 2
done

status=0
wait "$child" || status=$?
if [ "$status" -ne 0 ]; then
    echo "fed mac watchdog: client failed with exit $status (log $LOG)" >&2
    tail -n 20 "$LOG" >&2 || true
fi
exit "$status"
