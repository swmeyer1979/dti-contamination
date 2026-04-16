"""
Statistical analysis of DTI benchmark contamination — causal DiD design.

Primary analysis: Difference-in-Differences (DiD) estimator.
  Treated group: ESM-2 + ChemBERTa (pretrained encoders — subject to pretraining corpus overlap)
  Control group: DeepDTA (trained from scratch — no pretraining contamination)
  Benchmark arm: Davis / KIBA test sets (post-2011/2014, overlap with ChEMBL27 pretraining corpus)
  Holdout arm:   ChEMBL34 clean holdout (document_year >= 2022, max_tanimoto < 0.4 vs ChEMBL27)

  DiD = [r_ESM2_benchmark - r_ESM2_holdout] - [r_DeepDTA_benchmark - r_DeepDTA_holdout]

  Parallel trends: DeepDTA's benchmark-vs-holdout gap should be near zero (random variation only).
  If DiD > 0 and DeepDTA gap ≈ 0, pretraining contamination causally inflates ESM-2 scores.

Secondary analysis: Continuous dose-response.
  Spearman r between max_tanimoto (compound similarity to ChEMBL27) and |y_pred - y_true|.
  Negative ρ = higher contamination → smaller error → inflated scores.

Robustness analysis: Contamination-class breakdown.
  Pearson r on contaminated (Tanimoto ≥ 0.6) vs clean (< 0.6) subsets within each benchmark.

Requires:
  - phase4_esm2_probe
  - phase4_deepdta

Inputs:
  - results/esm2_probe_predictions.parquet
  - results/deepdta_predictions.parquet
    columns: smiles, affinity (y_true), y_pred, dataset, split, subset, sequence
  - data/processed/compound_contamination.parquet
    columns: smiles, max_tanimoto, nearest_neighbor_id, contamination_class, cutoff_used

Outputs:
  - results/stats_report.json
  - results/stats_figures/did_estimates.png
  - results/stats_figures/tanimoto_error_scatter.png
  - results/stats_figures/contamination_breakdown.png

Sentinels:
  - writes checkpoints/phase5_stats.done
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from utils.logging_utils import setup_logging
from utils.sentinel import require_sentinel, write_sentinel
from utils.status import StatusUpdater


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── figure style ────────────────────────────────────────────────────────────
_STYLE: dict[str, Any] = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "legend.frameon": False,
}

MODEL_LABELS = {
    "esm2_probe": "ESM-2 + ChemBERTa",
    "deepdta": "DeepDTA",
    "random_probe": "Random-Weight Probe",
}
MODEL_COLORS = {
    "esm2_probe": "#2171b5",
    "deepdta": "#cb181d",
    "random_probe": "#737373",
}


# ── data loading ─────────────────────────────────────────────────────────────

def _load_predictions(results_dir: Path, logger) -> pd.DataFrame:
    """Load and concatenate phase4 prediction parquets."""
    frames = []
    required_models = {
        "esm2_probe": results_dir / "esm2_probe_predictions.parquet",
        "deepdta": results_dir / "deepdta_predictions.parquet",
    }
    optional_models = {
        "random_probe": results_dir / "random_probe_predictions.parquet",
    }
    spec = {**required_models}
    for model_name, path in optional_models.items():
        if path.exists():
            spec[model_name] = path
            logger.info("Optional model %s found — including in all analyses.", model_name)
        else:
            logger.info(
                "Optional model %s not found (%s) — run phase4_random_probe.py to include architecture-matched null.",
                model_name, path,
            )
    for model_name, path in spec.items():
        if model_name in required_models and not path.exists():
            raise FileNotFoundError(
                f"Phase 4 output missing: {path}\n"
                f"Run phase4_{model_name}.py before phase5_stats.py."
            )
        df = pd.read_parquet(path)
        df["model"] = model_name
        frames.append(df)
        logger.info("Loaded %d rows from %s.", len(df), path.name)

    preds = pd.concat(frames, ignore_index=True)

    # Normalise column names: phase4 uses "affinity" / "sequence" internally
    col_map: dict[str, str] = {}
    if "affinity" in preds.columns and "y_true" not in preds.columns:
        col_map["affinity"] = "y_true"
    if "sequence" in preds.columns and "protein_sequence" not in preds.columns:
        col_map["sequence"] = "protein_sequence"
    if "subset" in preds.columns and "split_subset" not in preds.columns:
        col_map["subset"] = "split_subset"
    if col_map:
        preds = preds.rename(columns=col_map)
        logger.info("Normalised columns: %s", col_map)

    required = {"smiles", "y_true", "y_pred", "model"}
    missing = required - set(preds.columns)
    if missing:
        raise ValueError(f"Prediction parquets missing columns: {missing}")

    preds["y_true"] = pd.to_numeric(preds["y_true"], errors="coerce")
    preds["y_pred"] = pd.to_numeric(preds["y_pred"], errors="coerce")
    preds = preds.dropna(subset=["y_true", "y_pred"])

    # Infer split_subset if absent
    if "split_subset" not in preds.columns:
        preds["split_subset"] = preds.get("split", "all").astype(str)
        logger.warning("split_subset column absent; inferred from 'split' or set to 'all'.")

    # Ensure dataset column exists
    if "dataset" not in preds.columns:
        preds["dataset"] = "unknown"
        logger.warning("'dataset' column absent; DiD estimator will be skipped.")

    # Propagate protein contamination columns if present in phase4 predictions
    for col in ["max_pident", "protein_contamination_class", "max_tanimoto_pubchem", "subset_pubchem"]:
        if col not in preds.columns:
            preds[col] = float("nan") if "max" in col or "pident" in col else "unknown"

    logger.info(
        "Total prediction rows after cleaning: %d. Datasets: %s",
        len(preds), sorted(preds["dataset"].unique().tolist()),
    )
    return preds


def _load_contamination(logger) -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / "compound_contamination.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Compound contamination file missing: {path}\n"
            "Run phase2_fpsim2.py before phase5_stats.py."
        )
    df = pd.read_parquet(path)
    df["max_tanimoto"] = pd.to_numeric(df["max_tanimoto"], errors="coerce")
    df = df.dropna(subset=["smiles", "max_tanimoto"])
    logger.info("Loaded %d compound contamination rows.", len(df))
    return df


# ── helper ───────────────────────────────────────────────────────────────────

def _pearson_r_safe(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    r, _ = pearsonr(a, b)
    return float(r)


# ── analysis 1: DiD causal estimator ─────────────────────────────────────────

def _did_estimator(
    preds: pd.DataFrame,
    n_boot: int = 10_000,
    seed: int = 42,
    logger=None,
) -> dict[str, Any]:
    """
    Difference-in-Differences causal estimator.

    Treated: ESM-2 (pretrained encoder — contaminated by pretraining corpus)
    Control: DeepDTA (no pretraining — pure task data)
    Benchmark arm: Davis / KIBA test set (contaminated benchmark compounds)
    Holdout arm:   ChEMBL34 clean holdout (novel post-2021, low Tanimoto)

    DiD = [r_ESM2_benchmark - r_ESM2_holdout] - [r_DeepDTA_benchmark - r_DeepDTA_holdout]

    Positive DiD → ESM-2 gains MORE from the contaminated benchmark relative to clean data
    than DeepDTA does. The DeepDTA term controls for difficulty differences between the
    benchmark and holdout (parallel trends assumption: DeepDTA gap ≈ 0).

    Bootstrap CIs by block-resampling within each (model, arm) group independently.
    """
    rng = np.random.default_rng(seed)

    if "dataset" not in preds.columns or "unknown" in preds["dataset"].unique():
        if logger:
            logger.warning("DiD requires 'dataset' column; skipping.")
        return {"stats": {}, "distributions": {}, "skipped": True, "reason": "no_dataset_column"}

    has_holdout = "holdout" in preds["dataset"].values
    if not has_holdout:
        if logger:
            logger.warning(
                "No holdout arm found in predictions (ChEMBL34 clean holdout not yet available). "
                "Run phase1_chembl34_holdout + phase2_holdout_filter + re-run phases 3/4."
            )
        return {"stats": {}, "distributions": {}, "skipped": True, "reason": "no_holdout_arm"}

    # Primary DiD: ESM-2 (treated, pretrained) vs DeepDTA (control, from-scratch)
    # Secondary DiD: ESM-2 vs random_probe (same architecture, no pretraining) — isolates
    #   pretraining benefit from architecture advantage. Only run if random_probe present.
    model_pairs = [("esm2_probe", "deepdta")]
    if "random_probe" in preds["model"].values:
        model_pairs.append(("esm2_probe", "random_probe"))

    treated_model = "esm2_probe"
    control_model = "deepdta"
    benchmarks = sorted(ds for ds in preds["dataset"].unique() if ds != "holdout")
    eval_groups = benchmarks + ["pooled"]

    results: dict[str, Any] = {}
    distributions: dict[str, list[float]] = {}

    for treated_model, control_model in model_pairs:
        pair_key = f"{treated_model}_vs_{control_model}"
        for benchmark in eval_groups:
            if benchmark == "pooled":
                bench_preds = preds[preds["dataset"] != "holdout"]
            else:
                bench_preds = preds[preds["dataset"] == benchmark]
            holdout_preds = preds[preds["dataset"] == "holdout"]

            # Collect (y_true, y_pred) arrays for each (model, arm)
            arms: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
            all_present = True
            for model in [treated_model, control_model]:
                for arm_name, arm_df in [("benchmark", bench_preds), ("holdout", holdout_preds)]:
                    mdf = arm_df[arm_df["model"] == model]
                    if len(mdf) < 3:
                        if logger:
                            logger.warning(
                                "DiD[%s|%s]: insufficient data model=%s arm=%s (n=%d); skipping.",
                                pair_key, benchmark, model, arm_name, len(mdf),
                            )
                        all_present = False
                        break
                    arms[(model, arm_name)] = (
                        mdf["y_true"].to_numpy(dtype=float),
                        mdf["y_pred"].to_numpy(dtype=float),
                    )
                if not all_present:
                    break

            if not all_present:
                continue

            r_tb = _pearson_r_safe(*arms[(treated_model, "benchmark")])
            r_th = _pearson_r_safe(*arms[(treated_model, "holdout")])
            r_cb = _pearson_r_safe(*arms[(control_model, "benchmark")])
            r_ch = _pearson_r_safe(*arms[(control_model, "holdout")])

            within_treated = r_tb - r_th
            within_control = r_cb - r_ch
            did = within_treated - within_control

            # Bootstrap: resample each (model, arm) group independently
            boot_did = np.empty(n_boot)
            for i in range(n_boot):
                bv: dict[tuple[str, str], float] = {}
                for (model, arm), (yt, yp) in arms.items():
                    n = len(yt)
                    idx = rng.integers(0, n, size=n)
                    bv[(model, arm)] = _pearson_r_safe(yt[idx], yp[idx])
                b_treated = bv[(treated_model, "benchmark")] - bv[(treated_model, "holdout")]
                b_control = bv[(control_model, "benchmark")] - bv[(control_model, "holdout")]
                boot_did[i] = b_treated - b_control

            ci_lo = float(np.nanpercentile(boot_did, 2.5))
            ci_hi = float(np.nanpercentile(boot_did, 97.5))
            p_value = float(np.mean(boot_did <= 0))

            result_key = f"{pair_key}__{benchmark}"
            results[result_key] = {
                "benchmark": benchmark,
                "treated_model": treated_model,
                "control_model": control_model,
                f"r_{treated_model}_benchmark": r_tb,
                f"r_{treated_model}_holdout": r_th,
                f"r_{control_model}_benchmark": r_cb,
                f"r_{control_model}_holdout": r_ch,
                "within_treated_gap": within_treated,
                "within_control_gap": within_control,
                "did": did,
                "did_ci95": [ci_lo, ci_hi],
                "p_value_one_sided": p_value,
                f"n_{treated_model}_benchmark": int(len(arms[(treated_model, "benchmark")][0])),
                f"n_{treated_model}_holdout": int(len(arms[(treated_model, "holdout")][0])),
                f"n_{control_model}_benchmark": int(len(arms[(control_model, "benchmark")][0])),
                f"n_{control_model}_holdout": int(len(arms[(control_model, "holdout")][0])),
                "n_bootstrap": n_boot,
            }
            distributions[result_key] = boot_did.tolist()

            if logger:
                logger.info(
                    "DiD[%s|%s]: treated(bench=%.4f, hold=%.4f, gap=%.4f) "
                    "| control(bench=%.4f, hold=%.4f, gap=%.4f) "
                    "| DiD=%.4f [%.4f, %.4f] p=%.4f",
                    pair_key, benchmark, r_tb, r_th, within_treated,
                    r_cb, r_ch, within_control,
                    did, ci_lo, ci_hi, p_value,
                )

    return {"stats": results, "distributions": distributions}


# ── analysis 2: spearman r of max_tanimoto vs prediction error ───────────────

def _tanimoto_error_correlation(
    preds: pd.DataFrame,
    contam: pd.DataFrame,
    logger=None,
) -> dict[str, Any]:
    """
    Spearman ρ between compound contamination similarity and |y_pred - y_true| per model.

    Uses the correct contamination reference per model:
      - DeepDTA: ChEMBL27 (no pretraining, but used as shared baseline)
      - ESM-2 + ChemBERTa: both ChEMBL27 (compound proxy) and PubChem10M (correct ChemBERTa corpus)
        if max_tanimoto_pubchem is available in predictions.
      - random_probe: ChEMBL27 (same as ESM-2, architecture-matched null)

    Negative ρ = higher similarity → smaller error → inflated scores (hypothesis direction).
    """
    contam_lookup_chembl = contam.set_index("smiles")["max_tanimoto"]

    results: dict[str, Any] = {}
    scatter_data: list[dict[str, Any]] = []

    for model in sorted(preds["model"].unique()):
        mdf = preds[preds["model"] == model].copy()
        mdf["abs_error"] = (mdf["y_pred"] - mdf["y_true"]).abs()

        # Always compute vs ChEMBL27 (primary analysis, shared across models)
        mdf["max_tanimoto_chembl27"] = mdf["smiles"].map(contam_lookup_chembl)
        joined = mdf.dropna(subset=["max_tanimoto_chembl27", "abs_error"])

        if len(joined) < 5:
            if logger:
                logger.warning(
                    "Tanimoto correlation: too few matched rows for %s (%d).", model, len(joined)
                )
            continue

        tanimoto = joined["max_tanimoto_chembl27"].to_numpy(dtype=float)
        errors = joined["abs_error"].to_numpy(dtype=float)
        rho, p_val = spearmanr(tanimoto, errors)

        results[model] = {
            "model": model,
            "corpus": "chembl27",
            "n_matched": int(len(joined)),
            "spearman_r": float(rho),
            "spearman_p": float(p_val),
        }
        for t, e in zip(tanimoto.tolist(), errors.tolist()):
            scatter_data.append({"model": model, "max_tanimoto": t, "abs_error": e, "corpus": "chembl27"})

        if logger:
            logger.info("%s (vs ChEMBL27) | Spearman ρ=%.4f p=%.4e n=%d", model, rho, p_val, len(joined))

        # For ESM-2 probe: also compute vs PubChem10M (correct ChemBERTa corpus) if available
        if model == "esm2_probe" and "max_tanimoto_pubchem" in mdf.columns:
            joined_pub = mdf.dropna(subset=["max_tanimoto_pubchem", "abs_error"])
            joined_pub = joined_pub[joined_pub["max_tanimoto_pubchem"].notna() & (joined_pub["max_tanimoto_pubchem"] != "unknown")]
            try:
                joined_pub["max_tanimoto_pubchem"] = pd.to_numeric(joined_pub["max_tanimoto_pubchem"], errors="coerce")
                joined_pub = joined_pub.dropna(subset=["max_tanimoto_pubchem"])
            except Exception:
                joined_pub = pd.DataFrame()

            if len(joined_pub) >= 5:
                tan_pub = joined_pub["max_tanimoto_pubchem"].to_numpy(dtype=float)
                err_pub = joined_pub["abs_error"].to_numpy(dtype=float)
                rho_pub, p_pub = spearmanr(tan_pub, err_pub)
                results[f"{model}__pubchem"] = {
                    "model": model,
                    "corpus": "pubchem10m",
                    "n_matched": int(len(joined_pub)),
                    "spearman_r": float(rho_pub),
                    "spearman_p": float(p_pub),
                }
                for t, e in zip(tan_pub.tolist(), err_pub.tolist()):
                    scatter_data.append({"model": model, "max_tanimoto": t, "abs_error": e, "corpus": "pubchem10m"})
                if logger:
                    logger.info(
                        "%s (vs PubChem10M) | Spearman ρ=%.4f p=%.4e n=%d",
                        model, rho_pub, p_pub, len(joined_pub),
                    )
            elif logger:
                logger.info(
                    "%s: max_tanimoto_pubchem not populated (run phase2_pubchem_fpsim to get ChemBERTa-specific labels).",
                    model,
                )

    return {"stats": results, "scatter_data": scatter_data}


# ── analysis 3: contamination class breakdown ─────────────────────────────────

def _contamination_class_breakdown(
    preds: pd.DataFrame,
    logger=None,
) -> dict[str, Any]:
    """
    Per-contamination-class Pearson r within each benchmark.

    Uses split_subset column: values 'contaminated' (Tanimoto ≥ 0.6 vs ChEMBL27)
    vs 'clean' (Tanimoto < 0.6). If contamination inflates scores, ESM-2 should
    show r_contaminated >> r_clean while DeepDTA should show a smaller gap.
    """
    results: dict[str, Any] = {}
    has_dataset = "dataset" in preds.columns
    benchmarks = (
        sorted(ds for ds in preds["dataset"].unique() if ds != "holdout")
        if has_dataset else ["all"]
    )

    for model in sorted(preds["model"].unique()):
        mdf = preds[preds["model"] == model]

        for benchmark in benchmarks:
            bdf = mdf[mdf["dataset"] == benchmark] if has_dataset else mdf

            for subset in ["contaminated", "clean"]:
                sub = bdf[bdf["split_subset"] == subset] if "split_subset" in bdf.columns else pd.DataFrame()
                if len(sub) < 5:
                    continue

                key = f"{model}__{benchmark}__{subset}"
                r = _pearson_r_safe(
                    sub["y_true"].to_numpy(dtype=float),
                    sub["y_pred"].to_numpy(dtype=float),
                )
                results[key] = {
                    "model": model,
                    "benchmark": benchmark,
                    "subset": subset,
                    "pearson_r": r,
                    "n": int(len(sub)),
                }
                if logger:
                    logger.info("  %s: Pearson r=%.4f n=%d", key, r, len(sub))

    return {"stats": results}


# ── analysis 4: protein sequence identity stratification ──────────────────────

def _protein_contamination_analysis(
    preds: pd.DataFrame,
    logger=None,
) -> dict[str, Any]:
    """
    Stratify Pearson r by protein contamination class (max sequence identity vs UniRef50 pre-cutoff).

    ESM-2 pretrains on UniRef50; if protein memorization drives performance inflation,
    ESM-2 should show higher r on proteins with high UniRef50 identity while DeepDTA
    (no protein pretraining) should show a smaller gap.

    Strata (MMseqs2 easy-search hits, min-seq-id 0.4):
      contaminated: max_pident >= 90  (essentially identical to a UniRef50 sequence)
      partial:      40 <= max_pident < 90
      clean:        max_pident < 40 (no MMseqs2 hit above 40% — truly novel)

    Note: For Davis/KIBA (human kinases), virtually all proteins will be >=90%
    identical to UniRef50 entries. Zero variance in this stratum means the protein
    contamination channel cannot be tested with this benchmark selection. This null
    finding should be reported explicitly.
    """
    if "max_pident" not in preds.columns or "protein_contamination_class" not in preds.columns:
        if logger:
            logger.warning(
                "Protein contamination columns (max_pident, protein_contamination_class) absent "
                "from predictions — re-run phase3 with protein_contamination.parquet in place."
            )
        return {"stats": {}, "skipped": True, "reason": "missing_protein_contamination_columns"}

    results: dict[str, Any] = {}
    has_dataset = "dataset" in preds.columns
    benchmarks = (
        sorted(ds for ds in preds["dataset"].unique() if ds != "holdout")
        if has_dataset else ["all"]
    )

    for model in sorted(preds["model"].unique()):
        mdf = preds[preds["model"] == model]

        # Spearman ρ: max_pident vs absolute error (continuous dose-response on protein side)
        valid = mdf.dropna(subset=["max_pident"])
        if len(valid) >= 5:
            abs_err = (valid["y_pred"] - valid["y_true"]).abs().to_numpy(dtype=float)
            pident_vals = valid["max_pident"].to_numpy(dtype=float)
            if np.std(pident_vals) < 1e-8:
                # Zero variance: all proteins have the same identity (e.g. all human kinases = 100%)
                # Cannot compute Spearman correlation — report as explicit null finding.
                results[f"{model}__protein_spearman"] = {
                    "model": model,
                    "spearman_r_pident_vs_error": None,
                    "spearman_p": None,
                    "n": int(len(valid)),
                    "all_one_class": True,
                    "note": (
                        f"All {len(valid)} matched proteins have max_pident={pident_vals[0]:.1f} "
                        "(zero variance). Davis/KIBA consist entirely of human kinases with near-100% "
                        "UniRef50 identity — no variation to stratify on. This benchmark is unsuitable "
                        "for protein contamination analysis; it is itself a reportable null finding."
                    ),
                }
                if logger:
                    logger.info(
                        "%s | Protein Spearman: zero variance (all max_pident=%.1f, n=%d) — "
                        "human kinase benchmark unsuitble for protein contamination stratification.",
                        model, pident_vals[0], len(valid),
                    )
            else:
                rho, p_val = spearmanr(pident_vals, abs_err)
                results[f"{model}__protein_spearman"] = {
                    "model": model,
                    "spearman_r_pident_vs_error": float(rho),
                    "spearman_p": float(p_val),
                    "n": int(len(valid)),
                    "all_one_class": False,
                }
                if logger:
                    logger.info("%s | Protein Spearman ρ(pident, |error|)=%.4f p=%.4e n=%d", model, rho, p_val, len(valid))

        for benchmark in benchmarks:
            bdf = mdf[mdf["dataset"] == benchmark] if has_dataset else mdf

            strata_counts: dict[str, int] = {}
            for pclass in ["contaminated", "partial", "clean"]:
                sub = bdf[bdf["protein_contamination_class"] == pclass]
                strata_counts[pclass] = len(sub)
                if len(sub) < 5:
                    continue
                key = f"{model}__{benchmark}__prot_{pclass}"
                r = _pearson_r_safe(
                    sub["y_true"].to_numpy(dtype=float),
                    sub["y_pred"].to_numpy(dtype=float),
                )
                results[key] = {
                    "model": model,
                    "benchmark": benchmark,
                    "protein_class": pclass,
                    "pearson_r": r,
                    "n": int(len(sub)),
                }
                if logger:
                    logger.info("  %s: Pearson r=%.4f n=%d", key, r, len(sub))

            if logger:
                logger.info("  %s | %s strata counts: %s", model, benchmark, strata_counts)

    return {"stats": results}


# ── figures ───────────────────────────────────────────────────────────────────

def _fig_did(did_result: dict[str, Any], out_path: Path, logger) -> None:
    """Bar chart of DiD estimates per (model-pair × benchmark), with 95% CI error bars."""
    stats = did_result.get("stats", {})
    if not stats:
        logger.warning("No DiD stats to plot.")
        return

    keys_ordered = list(stats.keys())
    labels = []
    for k in keys_ordered:
        row = stats[k]
        bench = row["benchmark"]
        ctrl = row.get("control_model", "deepdta")
        bench_label = bench.upper() if bench != "pooled" else "Pooled"
        ctrl_label = MODEL_LABELS.get(ctrl, ctrl)
        labels.append(f"{bench_label}\nvs {ctrl_label}")

    dids = [stats[k]["did"] for k in keys_ordered]
    ci_los = [stats[k]["did_ci95"][0] for k in keys_ordered]
    ci_his = [stats[k]["did_ci95"][1] for k in keys_ordered]
    p_vals = [stats[k]["p_value_one_sided"] for k in keys_ordered]

    x = np.arange(len(keys_ordered))
    err_lo = [d - lo for d, lo in zip(dids, ci_los)]
    err_hi = [hi - d for d, hi in zip(dids, ci_his)]

    colors = [
        "#2171b5" if "deepdta" in k else "#737373"
        for k in keys_ordered
    ]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(max(5, 2.0 * len(keys_ordered)), 4.5))

        ax.bar(x, dids, color=colors, width=0.5, zorder=3)
        ax.errorbar(
            x, dids,
            yerr=[err_lo, err_hi],
            fmt="none", color="black", linewidth=1.2, capsize=5, zorder=4,
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.7, zorder=2)

        ci_all = ci_his + [abs(v) for v in dids]
        y_top = max(ci_all) * 1.15 if ci_all else 0.1
        for i, (d, p) in enumerate(zip(dids, p_vals)):
            p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
            ax.text(
                i, y_top,
                f"DiD={d:.3f}\n{p_str}",
                ha="center", va="bottom", fontsize=7, color="#222222",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("Benchmark × Control Model")
        ax.set_ylabel("DiD Estimate  (Pearson r)")
        ax.set_title(
            "Difference-in-Differences: Causal Effect of Pretraining Contamination\n"
            "[ESM-2 benchmark−holdout gap] − [control benchmark−holdout gap]"
        )
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

        # Legend: blue = vs DeepDTA (primary), grey = vs random probe (ablation)
        from matplotlib.patches import Patch
        legend_items = [
            Patch(facecolor="#2171b5", label="vs DeepDTA (primary DiD)"),
            Patch(facecolor="#737373", label="vs Random Probe (ablation)"),
        ]
        ax.legend(handles=legend_items, fontsize=8, loc="upper right")

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    logger.info("Saved %s.", out_path)


def _fig_tanimoto_scatter(tanimoto_result: dict[str, Any], out_path: Path, logger) -> None:
    """Scatter of max_tanimoto vs |error| with Spearman ρ annotated, per model."""
    scatter_data = tanimoto_result.get("scatter_data", [])
    stats = tanimoto_result["stats"]

    if not scatter_data:
        logger.warning("No scatter data for tanimoto plot.")
        return

    df_plot = pd.DataFrame(scatter_data)
    models = sorted(df_plot["model"].unique())
    ncols = len(models)

    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4), sharey=False, squeeze=False)

        for ax, model in zip(axes[0], models):
            mdf = df_plot[df_plot["model"] == model]
            color = MODEL_COLORS.get(model, "#555555")

            plot_df = mdf if len(mdf) <= 3000 else mdf.sample(3000, random_state=0)
            ax.scatter(
                plot_df["max_tanimoto"],
                plot_df["abs_error"],
                alpha=0.25, s=6, color=color, linewidths=0, rasterized=True,
            )

            x = plot_df["max_tanimoto"].to_numpy(dtype=float)
            y = plot_df["abs_error"].to_numpy(dtype=float)
            if len(x) > 2:
                m, b = np.polyfit(x, y, 1)
                xfit = np.linspace(x.min(), x.max(), 200)
                ax.plot(xfit, m * xfit + b, color=color, linewidth=1.5, alpha=0.9)

            if model in stats:
                rho = stats[model]["spearman_r"]
                p = stats[model]["spearman_p"]
                p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
                ax.text(
                    0.97, 0.97,
                    f"Spearman ρ={rho:.3f}\n{p_str}\nn={stats[model]['n_matched']:,}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    color="#222222",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8),
                )

            ax.set_xlabel("Max Tanimoto Similarity to ChEMBL27")
            ax.set_ylabel("|Prediction Error|  (|ŷ − y|)")
            ax.set_title(MODEL_LABELS.get(model, model))
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        fig.suptitle(
            "Compound Contamination vs. Prediction Error\n"
            "(Negative ρ = higher similarity → smaller error → inflated scores)",
            fontsize=10, y=1.02,
        )
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    logger.info("Saved %s.", out_path)


def _fig_contamination_breakdown(breakdown_result: dict[str, Any], out_path: Path, logger) -> None:
    """Grouped bar chart: Pearson r per model × contamination class × benchmark."""
    stats = breakdown_result.get("stats", {})
    if not stats:
        logger.warning("No contamination breakdown stats to plot.")
        return

    records = list(stats.values())
    df = pd.DataFrame(records)

    models_present = sorted(df["model"].unique())
    benchmarks_present = sorted(df["benchmark"].unique())

    with plt.rc_context(_STYLE):
        ncols = len(benchmarks_present)
        fig, axes = plt.subplots(
            1, ncols, figsize=(3.5 * ncols, 4.5), sharey=True, squeeze=False
        )

        for col, benchmark in enumerate(benchmarks_present):
            ax = axes[0][col]
            bdf = df[df["benchmark"] == benchmark]

            x = np.arange(len(models_present))
            width = 0.35

            for j, subset in enumerate(["contaminated", "clean"]):
                heights = []
                for model in models_present:
                    row = bdf[(bdf["model"] == model) & (bdf["subset"] == subset)]
                    heights.append(float(row["pearson_r"].values[0]) if len(row) > 0 else 0.0)

                color = "#2171b5" if subset == "contaminated" else "#74c476"
                ax.bar(
                    x + (j - 0.5) * width, heights, width,
                    label=subset.capitalize(), color=color, alpha=0.85,
                )

            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models_present], rotation=15, ha="right")
            ax.set_title(benchmark.upper())
            ax.set_ylabel("Pearson r" if col == 0 else "")
            ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)

            if col == 0:
                ax.legend(title="Contamination class", fontsize=8)

        fig.suptitle(
            "Pearson r by Contamination Class Within Each Benchmark\n"
            "(Contaminated = Tanimoto ≥ 0.6 vs ChEMBL27; Clean = Tanimoto < 0.6)",
            fontsize=10, y=1.02,
        )
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    logger.info("Saved %s.", out_path)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    phase = "phase5_stats"
    status = StatusUpdater(PROJECT_ROOT)
    logger = setup_logging(PROJECT_ROOT, phase)
    status.update(phase, "running", progress=0.0)

    try:
        for prereq in ("phase4_esm2_probe", "phase4_deepdta"):
            try:
                require_sentinel(prereq)
            except RuntimeError as e:
                logger.error(str(e))
                status.update(phase, "blocked", error=str(e))
                return 1

        results_dir = PROJECT_ROOT / "results"
        figures_dir = results_dir / "stats_figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading prediction parquets...")
        preds = _load_predictions(results_dir, logger)
        status.update(phase, "running", progress=0.10)

        logger.info("Loading compound contamination parquet...")
        contam = _load_contamination(logger)
        status.update(phase, "running", progress=0.15)

        # ── analysis 1: DiD causal estimator ──
        logger.info("Running DiD causal estimator (10,000 bootstrap iterations)...")
        did_result = _did_estimator(preds, n_boot=10_000, seed=42, logger=logger)
        status.update(phase, "running", progress=0.50)

        # ── analysis 2: continuous dose-response ──
        logger.info("Computing Spearman ρ (max_tanimoto vs |error|)...")
        tanimoto_result = _tanimoto_error_correlation(preds, contam, logger=logger)
        status.update(phase, "running", progress=0.65)

        # ── analysis 3: contamination class breakdown ──
        logger.info("Computing contamination-class breakdown (contaminated vs clean)...")
        breakdown_result = _contamination_class_breakdown(preds, logger=logger)
        status.update(phase, "running", progress=0.75)

        # ── analysis 4: protein sequence identity stratification ──
        logger.info("Computing protein contamination stratification (sequence identity vs UniRef50)...")
        prot_contam_result = _protein_contamination_analysis(preds, logger=logger)
        status.update(phase, "running", progress=0.80)

        # ── figures ──
        logger.info("Generating figures...")
        if not did_result.get("skipped"):
            _fig_did(did_result, figures_dir / "did_estimates.png", logger)
        _fig_tanimoto_scatter(tanimoto_result, figures_dir / "tanimoto_error_scatter.png", logger)
        _fig_contamination_breakdown(breakdown_result, figures_dir / "contamination_breakdown.png", logger)
        status.update(phase, "running", progress=0.92)

        # ── compute contamination class balance per benchmark for reporting ──
        contam_balance: dict[str, Any] = {}
        if "dataset" in preds.columns and "split_subset" in preds.columns:
            test_preds = preds[preds["dataset"] != "holdout"]
            for ds in sorted(test_preds["dataset"].unique()):
                ds_df = test_preds[test_preds["dataset"] == ds]
                # Use first model to count (same for all models)
                model0 = ds_df["model"].iloc[0]
                m0 = ds_df[ds_df["model"] == model0]
                counts = m0["split_subset"].value_counts().to_dict()
                total = sum(counts.values())
                contam_balance[ds] = {
                    "n_total": total,
                    "n_contaminated": int(counts.get("contaminated", 0)),
                    "n_clean": int(counts.get("clean", 0)),
                    "pct_contaminated": round(100 * counts.get("contaminated", 0) / max(total, 1), 1),
                    "note": (
                        "Clean n is too small for reliable contaminated-vs-clean comparison."
                        if counts.get("clean", 0) < 200 else ""
                    ),
                }
                logger.info(
                    "Contamination balance %s: contaminated=%d (%.1f%%), clean=%d (%.1f%%)",
                    ds,
                    counts.get("contaminated", 0), 100 * counts.get("contaminated", 0) / max(total, 1),
                    counts.get("clean", 0), 100 * counts.get("clean", 0) / max(total, 1),
                )

        # ── write JSON report ──
        report: dict[str, Any] = {
            "phase": phase,
            "design": {
                "estimand": "DiD causal effect of pretraining corpus contamination on DTI benchmark Pearson r",
                "treated": "ESM-2 + ChemBERTa (pretrained encoders, subject to corpus overlap)",
                "control_primary": "DeepDTA (trained from scratch, no pretraining contamination)",
                "control_ablation": "Random-Weight Probe (same architecture as ESM-2+ChemBERTa, random init — isolates pretraining from architecture)",
                "benchmark_arm": "Davis / KIBA test sets (compounds overlap with ChEMBL27 pretraining corpus)",
                "holdout_arm": "ChEMBL34 post-2021 holdout (document_year>=2022, max_tanimoto<0.4 vs ChEMBL27)",
                "tanimoto_threshold_benchmark": 0.6,
                "tanimoto_threshold_holdout_gate": 0.4,
                "compound_contamination_reference": {
                    "chembl27": (
                        "Used for all models (primary). ChEMBL27 = MolBERT pretraining corpus. "
                        "Also serves as proxy for ChemBERTa contamination (see limitation below)."
                    ),
                    "zinc_limitation": (
                        "ChemBERTa variant used (seyonec/ChemBERTa-zinc-base-v1) was pretrained on "
                        "a ZINC subset, NOT PubChem10M. ZINC-based contamination scoring was not "
                        "computed (HuggingFace dataset access requires authentication; ZINC ~250M "
                        "compounds makes FPSim2 DB build impractical). ChEMBL27 is used as a "
                        "conservative proxy since ZINC overlaps substantially with ChEMBL compounds."
                    ),
                },
                "protein_contamination_reference": "UniRef50 2021_04 (ESM-2 pretraining corpus) via MMseqs2 sequence identity.",
                "contamination_class_balance": contam_balance,
            },
            "analysis_1_did": {
                "description": (
                    "DiD = [r_ESM2_benchmark - r_ESM2_holdout] - [r_DeepDTA_benchmark - r_DeepDTA_holdout]. "
                    "Positive DiD indicates ESM-2 gains disproportionately more from contaminated benchmarks "
                    "than from clean data, after controlling for intrinsic difficulty differences via DeepDTA. "
                    "Bootstrap CIs by independent block-resampling of each (model, arm) group."
                ),
                **did_result,
            },
            "analysis_2_tanimoto_error": {
                "description": (
                    "Continuous dose-response: Spearman ρ between max_tanimoto (compound similarity to "
                    "ChEMBL27 pretraining corpus) and |y_pred - y_true|. Negative ρ indicates higher "
                    "contamination → smaller prediction error → inflated benchmark scores."
                ),
                "stats": tanimoto_result["stats"],
            },
            "analysis_3_contamination_breakdown": {
                "description": (
                    "Within-benchmark heterogeneity: Pearson r on contaminated (Tanimoto ≥ 0.6 vs ChEMBL27) "
                    "vs clean (<0.6) subsets. If contamination drives inflation, ESM-2 should show "
                    "r_contaminated >> r_clean while DeepDTA shows a smaller gap."
                ),
                "stats": breakdown_result["stats"],
            },
            "analysis_4_protein_contamination": {
                "description": (
                    "Protein sequence identity stratification: Pearson r by protein contamination class "
                    "(max_pident vs UniRef50 pre-ESM2 cutoff via MMseqs2). "
                    "Strata: contaminated (>=90%), partial (40-90%), clean (<40%). "
                    "If ESM-2 memorizes protein sequences from pretraining, r should be higher for "
                    "contaminated proteins vs clean, while DeepDTA gap should be near zero. "
                    "Note: Davis/KIBA consist of human kinases — expect near-100% identity for all "
                    "benchmark proteins, so variance may be zero (itself a reportable null finding)."
                ),
                **prot_contam_result,
            },
        }

        report_path = results_dir / "stats_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        logger.info("Wrote %s.", report_path)
        status.update(phase, "running", progress=0.97)

        write_sentinel(phase)
        status.update(phase, "completed", progress=1.0)
        logger.info("Phase 5 complete.")
        return 0

    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.error("Fatal error in %s: %s", phase, e)
        logger.error(tb)
        status.update(phase, "error", error=str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
