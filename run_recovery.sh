#!/usr/bin/env bash
# Runs after pipeline exits: re-runs deepdta (STATUS.json race fixed) + phase5
set -euo pipefail
PY=".venv/bin/python3"
LOGS="logs"
mkdir -p "$LOGS"

ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

log "Recovery: re-running phase4_deepdta (STATUS.json race condition fixed)"
"$PY" src/phase4_deepdta.py 2>&1 | tee "$LOGS/phase4_deepdta_recovery.log"
log "DONE phase4_deepdta"

# Wait for phase4_random_probe sentinel in case it's still running
until [[ -f checkpoints/phase4_random_probe.done ]]; do
    log "Waiting for phase4_random_probe to finish..."
    sleep 10
done
log "phase4_random_probe done"

log "Running phase5_stats"
"$PY" src/phase5_stats.py 2>&1 | tee "$LOGS/phase5_stats.log"
log "DONE phase5_stats"
log "Results: results/stats_report.json"
