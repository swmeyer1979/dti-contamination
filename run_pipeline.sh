#!/bin/bash
set -euo pipefail

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required. Install with: brew install tmux"
  exit 1
fi

SESSION="dti-research"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYBIN="$SCRIPT_DIR/.venv/bin/python3"
if [ ! -f "$PYBIN" ]; then
  echo "venv not found at $SCRIPT_DIR/.venv — run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

tmux new-session -d -s "$SESSION" -c "$SCRIPT_DIR" -n "p1-chembl" "$PYBIN src/phase1_chembl.py"
tmux new-window -t "$SESSION" -c "$SCRIPT_DIR" -n "p1-pubchem" "$PYBIN src/phase1_pubchem.py"
tmux new-window -t "$SESSION" -c "$SCRIPT_DIR" -n "p1-uniprot" "$PYBIN src/phase1_uniprot.py"
tmux new-window -t "$SESSION" -c "$SCRIPT_DIR" -n "p2-fpsim2" "$PYBIN src/phase2_fpsim2.py"
tmux new-window -t "$SESSION" -c "$SCRIPT_DIR" -n "p2-mmseqs2" "$PYBIN src/phase2_mmseqs2.py"
# ChEMBL34 holdout arm (post-2021 Ki/Kd for DiD clean arm)
tmux new-window -t "$SESSION" -c "$SCRIPT_DIR" -n "p1-holdout" "$PYBIN src/phase1_chembl34_holdout.py"
# Structural novelty filter for holdout (Tanimoto < 0.4 vs ChEMBL27)
tmux new-window -t "$SESSION" -c "$SCRIPT_DIR" -n "p2-holdout-filter" "$PYBIN src/phase2_holdout_filter.py"
tmux new-window -t "$SESSION" -c "$SCRIPT_DIR" -n "p3-split" "$PYBIN src/phase3_temporal_split.py"
tmux new-window -t "$SESSION" -c "$SCRIPT_DIR" -n "p4-esm2" "$PYBIN src/phase4_esm2_probe.py"
tmux new-window -t "$SESSION" -c "$SCRIPT_DIR" -n "p4-deepdta" "$PYBIN src/phase4_deepdta.py"
tmux new-window -t "$SESSION" -c "$SCRIPT_DIR" -n "p5-stats" "$PYBIN src/phase5_stats.py"

echo "Attach with: tmux attach -t $SESSION | Status: cat STATUS.json"
echo ""
echo "Pipeline windows:"
echo "  p1-chembl       — ChEMBL27 benchmark timestamps"
echo "  p1-pubchem      — PubChem compound deposit dates"
echo "  p1-uniprot      — UniProt target sequences + timestamps"
echo "  p1-holdout      — ChEMBL34 post-2021 holdout API query  [DiD clean arm]"
echo "  p2-fpsim2       — Morgan fingerprint similarity vs ChEMBL27"
echo "  p2-holdout-filter — Tanimoto < 0.4 novelty gate for holdout"
echo "  p2-mmseqs2      — Protein sequence similarity vs UniRef50"
echo "  p3-split        — Temporal/contamination splits"
echo "  p4-esm2         — ESM-2 + ChemBERTa probe (treated model)"
echo "  p4-deepdta      — DeepDTA from scratch (control model)"
echo "  p5-stats        — DiD estimator + figures"
