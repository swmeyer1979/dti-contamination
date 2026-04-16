#!/usr/bin/env bash
# Revised pipeline runner — all 5 review fixes applied.
# Logs per-phase to logs/<phase>.log
# Run: bash run_pipeline_revised.sh

set -euo pipefail
PY=".venv/bin/python3"
LOGS="logs"
mkdir -p "$LOGS"

ts() { date '+%H:%M:%S'; }

log() { echo "[$(ts)] $*"; }

run_phase() {
    local phase="$1"; shift
    log "START $phase"
    "$PY" "src/${phase}.py" "$@" 2>&1 | tee "$LOGS/${phase}.log"
    local rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
        log "FAILED $phase (exit $rc) — check $LOGS/${phase}.log"
        exit $rc
    fi
    log "DONE $phase"
}

# ── phase 2: PubChem10M contamination (background, ~2h) ──────────────────────
log "Starting phase2_pubchem_fpsim in background (takes ~2h — pipeline continues without it)"
"$PY" src/phase2_pubchem_fpsim.py >"$LOGS/phase2_pubchem_fpsim.log" 2>&1 &
PUBCHEM_PID=$!
log "phase2_pubchem_fpsim PID=$PUBCHEM_PID"

# ── phase 3: temporal split + protein/pubchem contamination labels ────────────
run_phase phase3_temporal_split

# ── phase 4: all three models in parallel ────────────────────────────────────
log "Starting phase4_esm2_probe, phase4_deepdta, phase4_random_probe in parallel"

"$PY" src/phase4_esm2_probe.py  2>&1 | tee "$LOGS/phase4_esm2_probe.log"  &  P1=$!
"$PY" src/phase4_deepdta.py     2>&1 | tee "$LOGS/phase4_deepdta.log"     &  P2=$!
"$PY" src/phase4_random_probe.py 2>&1 | tee "$LOGS/phase4_random_probe.log" & P3=$!

wait_phase() {
    local pid=$1 name=$2
    if wait "$pid"; then
        log "DONE $name"
    else
        log "FAILED $name (exit $?) — check $LOGS/${name}.log"
        exit 1
    fi
}

wait_phase $P1 phase4_esm2_probe
wait_phase $P2 phase4_deepdta
wait_phase $P3 phase4_random_probe

# ── phase 5: stats ────────────────────────────────────────────────────────────
run_phase phase5_stats

log "Pipeline complete. Results in results/stats_report.json"
log "PubChem background job (PID=$PUBCHEM_PID) may still be running."
log "  Check: tail -f $LOGS/phase2_pubchem_fpsim.log"
log "  When done: rm checkpoints/phase3_temporal_split.done && bash run_pipeline_revised.sh"
