# When Random Initialization Wins: Evaluation Artifacts in Drug-Target Interaction Benchmarks

**Paper**: _When Random Initialization Wins: Evaluation Artifacts in Drug-Target Interaction Benchmarks_ (Samuel W. Meyer, 2026)

---

## Summary

We identify three interacting evaluation artifacts that distort drug-target interaction (DTI) benchmarks and demonstrate that correcting for them reverses model rankings on the same holdout.

**Key findings:**

- **Single-target dominance.** One kinase (Q92918/MAP3K1) contributes 428/1,028 pairs (42%) of a ChEMBL34 novel-chemistry holdout and 428/801 pairs (53%) of the kinase-only subset. DeepDTA's pooled Pearson r drops from +0.329 to −0.184 when this target is excluded.
- **Pair-level bootstrap undercovers 5×.** On a synthetic two-way clustered DGP, pair-level 95% CIs achieve 17% empirical coverage; two-way protein×compound cluster bootstrap achieves 76%.
- **Rank reversal under macro-averaged metrics.** Under macro-averaged per-target Pearson r (equal weight per target, n≥10 pairs), a randomly-initialized ESM-2+ChemBERTa transformer (+0.103) outperforms its pretrained counterpart (−0.035). This is a diagnostic of benchmark fragility, not a claim about random initialization.

**Actionable recommendation:** Report macro-averaged per-target r as the primary metric; pair-count distribution per target in every paper; two-way cluster bootstrap CIs.

---

## Repository Structure

```
├── src/
│   ├── phase1_*.py                  # Data acquisition (ChEMBL34, UniProt)
│   ├── phase2_*.py                  # Novelty filtering (FPSim2 Tanimoto gate)
│   ├── phase3_*.py                  # Holdout construction and temporal splits
│   ├── phase4_*.py                  # Model training (ESM-2 probe, DeepDTA, random baselines)
│   ├── phase5_stats.py              # Statistical analysis (bootstrap, DiD, macro-avg)
│   ├── phase5b_bootstrap_coverage_sim.py  # Coverage simulation
│   ├── phase5c_sensitivity.py       # Cutoff sensitivity + leave-one-target-out
│   ├── compute_influence.py         # Target influence decomposition
│   └── utils/                       # Shared utilities
├── data/
│   └── splits/                      # Holdout and training split parquets
├── results/
│   ├── *_predictions.parquet        # Per-model predictions on all datasets
│   ├── stats_report.json            # Full statistical analysis output
│   ├── sensitivity.json             # Cutoff sensitivity and bootstrap results
│   ├── influence_decomposition.json # Per-target influence on pooled r
│   └── stats_figures/               # Publication figures (PNG, 300 DPI)
├── paper/
│   ├── main.tex                     # LaTeX source
│   ├── main.pdf                     # Compiled paper
│   └── figures/                     # Figures referenced in paper
├── run_pipeline_revised.sh          # Full pipeline runner
└── requirements.txt
```

---

## Quickstart

### 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python ≥ 3.10. RDKit via pip. FPSim2 requires a C++ compiler.
MPS acceleration (Apple Silicon) is used automatically.

### 2. Run the full pipeline

```bash
bash run_pipeline_revised.sh
```

Approximate runtimes on Apple M-series:

| Phase | Time |
|-------|------|
| Phase 1: ChEMBL data pull | ~20 min |
| Phase 2: Tanimoto novelty filter | ~30 min |
| Phase 3: Holdout construction | ~5 min |
| Phase 4: ESM-2 probe | ~45 min |
| Phase 4: DeepDTA | ~55 min |
| Phase 4: Random baselines | ~15 min |
| Phase 5: Statistics + figures | ~5 min |

### 3. Reproduce statistical results only

Pre-computed predictions are included in `results/`. To regenerate all statistics and figures from those:

```bash
python3 src/phase5_stats.py
python3 src/phase5b_bootstrap_coverage_sim.py
python3 src/phase5c_sensitivity.py
python3 src/compute_influence.py
python3 src/make_figure_concentration.py
```

---

## Holdout Construction

- **Source**: ChEMBL34 documents dated ≥ 2022
- **Activity types**: K_d, K_i (binding assays only)
- **Affinity threshold**: pChEMBL ≥ 5.0
- **Novelty gate**: max Tanimoto < 0.4 to any ChEMBL27 compound (Morgan radius-2, 2048-bit FPSim2)
- **Deduplication**: one record per (canonical SMILES, UniProt), highest pChEMBL retained
- **Result**: 1,028 pairs, 85 targets, 941 compounds

**Note on contamination scope**: ESM-2 has a verified 2021-04 dataset stamp. ChemBERTa was trained on ~100k SMILES from ZINC; its exact held-out split is not public. The Tanimoto gate versus ChEMBL27 is a structural-novelty proxy, not confirmed exclusion.

---

## License

Code: MIT. Derived data splits: CC BY-SA 4.0 (consistent with upstream ChEMBL CC BY-SA 3.0).

---

## Citation

```bibtex
@article{meyer2026randomwins,
  title={When Random Initialization Wins: Evaluation Artifacts in Drug-Target Interaction Benchmarks},
  author={Meyer, Samuel W.},
  year={2026},
  note={Preprint}
}
```
