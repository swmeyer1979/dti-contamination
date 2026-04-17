# DTI Contamination: Causal Analysis of Pretraining Contamination in Drug-Target Interaction Benchmarks

**Paper**: _Pretraining Contamination Is Not Identifiable on Drug-Target Interaction Benchmarks: A Causal Analysis_ (arXiv link after submission)

---

## Summary

We test whether pretraining contamination inflates performance of ESM-2 + ChemBERTa on the Davis and KIBA drug-target interaction benchmarks, using a causal difference-in-differences (DiD) estimator.

**Key findings:**
- **No contamination inflation.** DiD = −0.45 (pooled, 95% CI [−0.55, −0.36], p = 1.0 against contamination hypothesis)
- **Pretraining contributes negligible benchmark signal.** A random-weight probe matches ESM-2 on both benchmarks (KIBA: 0.786 vs 0.786)
- **All models fail on novel chemistry.** Pearson r ≈ 0 on a clean ChEMBL34 holdout (r = 0.017 for ESM-2, −0.375 for DeepDTA, −0.287 for random probe)
- **Protein contamination is untestable** on Davis/KIBA — every target is a human kinase with 100% UniRef50 identity

The actionable conclusion: Davis and KIBA don't measure generalisation. Contamination is not the problem; the benchmarks are.

---

## Repository Structure

```
├── src/                        # Pipeline source code
│   ├── phase1_*.py             # Data acquisition (ChEMBL, UniProt, PubChem)
│   ├── phase2_*.py             # Contamination labelling (FPSim2, MMseqs2)
│   ├── phase3_temporal_split.py  # Train/test/holdout splits
│   ├── phase4_*.py             # Model training (ESM-2, DeepDTA, Random probe)
│   ├── phase5_stats.py         # Statistical analysis and figures
│   └── utils/                  # Shared utilities
├── config/
│   └── cutoffs.json            # Pretraining cutoff dates and contamination threshold
├── data/
│   ├── splits/                 # Temporal split parquets (Davis, KIBA, holdout)
│   └── processed/              # Contamination labels (FPSim2, MMseqs2)
├── results/
│   ├── *_metrics.json          # Per-model performance metrics
│   ├── stats_report.json       # Full statistical analysis output
│   └── stats_figures/          # Publication figures (PNG, 300 DPI)
├── run_pipeline_revised.sh     # Full pipeline runner
├── preflight.py                # Environment check
└── requirements.txt            # Python dependencies
```

---

## Quickstart

### 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python ≥ 3.10. RDKit is installed via pip (`rdkit` package). FPSim2 requires a C++ compiler.

For MPS acceleration (Apple Silicon), PyTorch will use MPS automatically. For CUDA, set `CUDA_VISIBLE_DEVICES` as needed.

### 2. Preflight check

```bash
python3 preflight.py
```

Verifies all dependencies and writes `preflight_report.json`.

### 3. Run the pipeline

```bash
bash run_pipeline_revised.sh
```

**Runtime estimate** (Apple M-series CPU/MPS):
| Phase | Time |
|-------|------|
| Phase 1: Data acquisition | ~20 min |
| Phase 2: FPSim2 + MMseqs2 | ~30 min |
| Phase 3: Temporal splits | ~5 min |
| Phase 4: ESM-2 probe | ~45 min (requires HuggingFace download on first run) |
| Phase 4: DeepDTA | ~55 min |
| Phase 4: Random probe | ~10 min |
| Phase 5: Statistics | ~2 min |

Phases 4a–4c run in parallel. Total: ~2–3 hours.

### 4. Skip to results

If you only want to reproduce the statistical analysis from the pre-computed predictions:

```bash
# Data splits and contamination labels are included in the repo
# Phase 4 outputs must be generated (or downloaded from Releases)
python3 src/phase5_stats.py
```

Results appear in `results/stats_report.json` and `results/stats_figures/`.

---

## Causal Design

The DiD estimator isolates contamination from general difficulty differences:

```
DiD = [r(ESM-2, benchmark) − r(ESM-2, holdout)]
    − [r(control, benchmark) − r(control, holdout)]
```

- **Treated**: ESM-2 + ChemBERTa (pretrained encoders, subject to corpus overlap)
- **Control (primary)**: DeepDTA (trained from scratch, no pretraining)
- **Control (ablation)**: Random-weight probe (same architecture as ESM-2, random init)
- **Benchmark arm**: Davis/KIBA test set (compounds ≥ 0.6 Tanimoto to ChEMBL27)
- **Holdout arm**: ChEMBL34 pairs with document year ≥ 2022 and Tanimoto < 0.4 to any ChEMBL27 compound

Positive DiD → contamination inflates ESM-2.  
Negative DiD (what we observe) → ESM-2 does NOT benefit disproportionately from the contaminated benchmark.

---

## Results

### Model Performance

| Model | Davis r | KIBA r | Holdout r |
|-------|---------|--------|-----------|
| ESM-2 + ChemBERTa | 0.666 [0.650, 0.683] | 0.786 [0.776, 0.796] | 0.017 [−0.051, 0.086] |
| DeepDTA | 0.709 [0.685, 0.732] | 0.854 [0.845, 0.862] | −0.375 [−0.429, −0.320] |
| Random-Weight Probe | 0.650 [0.623, 0.679] | 0.786 [0.776, 0.797] | −0.287 [−0.350, −0.219] |

### DiD Estimates (pooled)

| Control | DiD | 95% CI | p (H₁: DiD > 0) |
|---------|-----|--------|-----------------|
| DeepDTA | −0.453 | [−0.546, −0.361] | 1.000 |
| Random Probe | −0.300 | [−0.397, −0.202] | 1.000 |

---

## Data

### Included in this repository
- `data/splits/`: Davis, KIBA, and holdout temporal split parquets
- `data/processed/compound_contamination.parquet`: ECFP4 Tanimoto similarity to ChEMBL27
- `data/processed/protein_contamination.parquet`: MMseqs2 sequence identity to UniRef50

### Not included (generated by pipeline)
- `data/raw/`: ChEMBL FPSim2 database (~500 MB), raw downloads
- ESM-2 and ChemBERTa model weights (downloaded automatically from HuggingFace)

### Contamination labelling
- **Compound**: ECFP4 Morgan fingerprints (radius=2, 2048 bits), Tanimoto ≥ 0.6 vs ChEMBL27
- **Protein**: MMseqs2 easy-search, min-seq-id 0.4, vs UniRef50 (April 2021)
- **Threshold note**: ChemBERTa-zinc-base-v1 was pretrained on ZINC (~100K molecules), not ChEMBL27. The ZINC training split is not publicly accessible; ChEMBL27 is used as a conservative proxy.

---

## Citation

```bibtex
@article{meyer2026dti,
  title={Pretraining Contamination Does Not Inflate Drug-Target Interaction Benchmarks: A Causal Analysis},
  author={Meyer, Sam},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
