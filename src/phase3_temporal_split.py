"""
Constructs pre/post-cutoff test subsets from Davis and KIBA benchmarks.

Requires:
  - phase2_fpsim2
  - phase2_mmseqs2

Outputs:
  - data/splits/{dataset}_temporal_splits.parquet
  - data/splits/split_statistics.json

Sentinels:
  - writes checkpoints/phase3_temporal_split.done
"""

from __future__ import annotations

import ast
import json
import traceback
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from utils.logging_utils import setup_logging
from utils.sentinel import require_sentinel, write_sentinel
from utils.smiles import append_invalid_smiles, validate_smiles
from utils.status import StatusUpdater


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _download_if_missing(url: str, out_path: Path, timeout: float = 60) -> bool:
    """Download url to out_path. Returns True on success, False on 404."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    with requests.get(url, stream=True, timeout=timeout) as r:
        if r.status_code == 404:
            return False
        r.raise_for_status()
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        tmp.replace(out_path)
    return True


def _generate_fold_files(raw: Path, dataset: str, y_matrix: Any, rng_seed: int = 42) -> tuple[list[int], list[int]]:
    """Generate reproducible 80/20 random split when official fold files are unavailable."""
    y = np.array(y_matrix, dtype=float)
    finite_indices = [int(i) for i in np.argwhere(np.isfinite(y.ravel())).ravel()]
    rng = np.random.default_rng(rng_seed)
    rng.shuffle(finite_indices)
    split = int(len(finite_indices) * 0.8)
    train_idx, test_idx = finite_indices[:split], finite_indices[split:]
    # persist to disk so downstream can re-read
    (raw / f"{dataset}_train_fold_setting1.txt").write_text(json.dumps(train_idx))
    (raw / f"{dataset}_test_fold_setting1.txt").write_text(json.dumps(test_idx))
    return train_idx, test_idx


def _load_dict_literal(path: Path) -> dict[str, Any]:
    data = ast.literal_eval(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict literal in {path}, got {type(data)}")
    return data


def _load_array_like(path: Path) -> Any:
    # Try pickle first (DeepDTA Y files are Python 2 pickled numpy arrays)
    try:
        import pickle
        with path.open("rb") as f:
            data = pickle.load(f, encoding="bytes")
        if hasattr(data, "tolist"):
            return data  # return numpy array directly
        return data
    except Exception:
        pass
    text = path.read_text()
    try:
        return json.loads(text)
    except Exception:
        return ast.literal_eval(text)


def _parse_indices(path: Path) -> list[int]:
    obj = _load_array_like(path)
    if isinstance(obj, list) and obj and isinstance(obj[0], list):
        out = []
        for sub in obj:
            if not isinstance(sub, list):
                continue
            out.extend(int(x) for x in sub)
        return out
    if isinstance(obj, list):
        return [int(x) for x in obj]
    raise ValueError(f"Unexpected fold format in {path}: {type(obj)}")


def _murcko_scaffold(smiles: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return MurckoScaffold.GetMurckoSmiles(mol=mol)
    except Exception:  # noqa: BLE001
        return None


def _subset_label(deposit_date: Optional[pd.Timestamp], max_tanimoto: Optional[float], cutoff: pd.Timestamp, threshold: float = 0.6) -> str:
    # Label is based solely on Tanimoto similarity to ChEMBL27 (MolBERT training corpus).
    # deposit_date is kept for metadata but not used in labeling:
    # Davis/KIBA compounds are all pre-2020, so a deposit-date criterion produces empty "clean" sets.
    if max_tanimoto is None or pd.isna(max_tanimoto):
        return "ambiguous"
    if max_tanimoto >= threshold:
        return "contaminated"   # in ChEMBL27 → MolBERT saw this compound
    return "clean"              # novel vs ChEMBL27 → uncontaminated


def _build_pairs_df(
    dataset: Literal["davis", "kiba"],
    ligands: dict[str, str],
    proteins: dict[str, str],
    y_matrix: Any,
    train_indices: list[int],
    test_indices: list[int],
    logger,
) -> pd.DataFrame:
    lig_keys = list(ligands.keys())
    prot_keys = list(proteins.keys())
    n_lig = len(lig_keys)
    n_prot = len(prot_keys)

    y = np.array(y_matrix, dtype=float)
    if y.shape != (n_lig, n_prot):
        raise ValueError(f"{dataset}: Y shape {y.shape} does not match ligands/proteins ({n_lig},{n_prot})")

    finite_mask = np.isfinite(y)
    observed_pairs = list(zip(*np.where(finite_mask)))
    observed_count = len(observed_pairs)
    full_count = n_lig * n_prot

    max_index = max(train_indices + test_indices) if (train_indices or test_indices) else -1
    if max_index >= full_count:
        mode = "observed"
        if max_index >= observed_count:
            raise ValueError(
                f"{dataset}: fold indices up to {max_index} exceed observed pairs {observed_count}. "
                "This likely indicates unexpected DeepDTA fold format."
            )
        logger.info("%s: interpreting fold indices as OBSERVED-pair indices (n=%d).", dataset, observed_count)

        def idx_to_pair(idx: int) -> tuple[int, int]:
            return observed_pairs[idx][0], observed_pairs[idx][1]

    else:
        mode = "full"
        logger.info("%s: interpreting fold indices as FULL-matrix indices (n=%d).", dataset, full_count)

        def idx_to_pair(idx: int) -> tuple[int, int]:
            return idx // n_prot, idx % n_prot

    def build_rows(indices: list[int], split: str) -> list[dict]:
        rows = []
        for idx in indices:
            i, j = idx_to_pair(int(idx))
            aff = float(y[i, j])
            if not np.isfinite(aff):
                continue
            smi = str(ligands[lig_keys[i]])
            seq = str(proteins[prot_keys[j]])
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "ligand_key": lig_keys[i],
                    "protein_key": prot_keys[j],
                    "smiles": smi,
                    "sequence": seq,
                    "affinity": aff,
                    "pair_index_mode": mode,
                }
            )
        return rows

    rows = build_rows(train_indices, "train") + build_rows(test_indices, "test")
    return pd.DataFrame(rows)


def main() -> int:
    phase = "phase3_temporal_split"
    prereqs = ["phase2_fpsim2", "phase2_mmseqs2"]
    optional_prereqs = ["phase2_holdout_filter"]  # DiD holdout arm — processed if available
    status = StatusUpdater(PROJECT_ROOT)
    logger = setup_logging(PROJECT_ROOT, phase)
    status.update(phase, "running", progress=0.0)

    try:
        for p in prereqs:
            try:
                require_sentinel(p)
            except RuntimeError as e:
                logger.warning(str(e))
                status.update(phase, "blocked", error=str(e))
                return 0

        cutoffs = json.loads((PROJECT_ROOT / "config" / "cutoffs.json").read_text())
        molbert_cutoff = pd.Timestamp(cutoffs["molbert"]["cutoff_date"], tz="UTC")
        contamination_threshold = float(cutoffs.get("contamination_threshold", 0.6))

        raw = PROJECT_ROOT / "data" / "raw"
        for ds in ["davis", "kiba"]:
            _download_if_missing(
                f"https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/{ds}/ligands_can.txt",
                raw / f"{ds}_ligands.txt",
            )
            _download_if_missing(
                f"https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/{ds}/proteins.txt",
                raw / f"{ds}_proteins.txt",
            )
            _download_if_missing(
                f"https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/{ds}/Y",
                raw / f"{ds}_affinity.txt",
            )

        chembl = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "chembl_timestamps.parquet")
        chembl["first_known_date"] = pd.to_datetime(chembl["first_known_date"], errors="coerce", utc=True)
        chembl_dates = chembl.dropna(subset=["smiles"]).groupby("smiles")["first_known_date"].min().reset_index()
        chembl_dates = chembl_dates.rename(columns={"first_known_date": "deposit_date"})

        contam = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "compound_contamination.parquet")
        contam = contam.dropna(subset=["smiles"]).drop_duplicates(subset=["smiles"])

        # Optional: PubChem10M contamination for ChemBERTa-specific labeling
        pubchem_contam_path = PROJECT_ROOT / "data" / "processed" / "compound_contamination_pubchem.parquet"
        pubchem_contam = None
        if pubchem_contam_path.exists():
            pubchem_contam = pd.read_parquet(pubchem_contam_path)
            pubchem_contam = pubchem_contam.dropna(subset=["smiles"]).drop_duplicates(subset=["smiles"])
            logger.info("Loaded PubChem10M contamination for %d compounds.", len(pubchem_contam))
        else:
            logger.info("compound_contamination_pubchem.parquet not found — run phase2_pubchem_fpsim to get ChemBERTa-specific labels.")

        # Protein contamination (MMseqs2 vs UniRef50 pre-cutoff)
        prot_contam_path = PROJECT_ROOT / "data" / "processed" / "protein_contamination.parquet"
        prot_contam = None
        if prot_contam_path.exists():
            prot_contam = pd.read_parquet(prot_contam_path)
            prot_contam = prot_contam.dropna(subset=["uniprot_id"]).drop_duplicates(subset=["uniprot_id"])
            logger.info("Loaded protein contamination for %d UniProt IDs.", len(prot_contam))

        stats: dict[str, Any] = {"generated_at": pd.Timestamp.utcnow().isoformat(), "molbert_cutoff": str(molbert_cutoff.date())}
        warnings: list[str] = []

        for dataset in ["davis", "kiba"]:
            lig = _load_dict_literal(raw / f"{dataset}_ligands.txt")
            prot = _load_dict_literal(raw / f"{dataset}_proteins.txt")
            y = _load_array_like(raw / f"{dataset}_affinity.txt")

            train_fold_ok = _download_if_missing(
                f"https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/{dataset}/train_fold_setting1.txt",
                raw / f"{dataset}_train_fold_setting1.txt",
            )
            test_fold_ok = _download_if_missing(
                f"https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/{dataset}/test_fold_setting1.txt",
                raw / f"{dataset}_test_fold_setting1.txt",
            )
            if not train_fold_ok or not test_fold_ok:
                logger.warning(
                    "%s: official DeepDTA fold files not available (404). "
                    "Generating reproducible 80/20 random split (seed=42). "
                    "Note in paper: splits differ from original DeepDTA evaluation.",
                    dataset,
                )
                train_idx, test_idx = _generate_fold_files(raw, dataset, y, rng_seed=42)
                stats.setdefault("fold_source", {})[dataset] = "generated_random_80_20_seed42"
            else:
                train_idx = _parse_indices(raw / f"{dataset}_train_fold_setting1.txt")
                test_idx = _parse_indices(raw / f"{dataset}_test_fold_setting1.txt")
                stats.setdefault("fold_source", {})[dataset] = "deepdta_official"

            df = _build_pairs_df(dataset, lig, prot, y, train_idx, test_idx, logger=logger)
            if df.empty:
                raise RuntimeError(f"{dataset}: constructed empty pairs dataframe; check raw inputs.")

            # Validate and canonicalize SMILES before writing splits
            canon_smiles: list[Optional[str]] = []
            drop = np.zeros(len(df), dtype=bool)
            for i, smi in enumerate(df["smiles"].tolist()):
                ok, canon = validate_smiles(str(smi))
                if not ok or canon is None:
                    append_invalid_smiles(PROJECT_ROOT, source=phase, key=f"{dataset}:row_{i}", smiles=str(smi), reason="rdkit_invalid")
                    canon_smiles.append(None)
                    drop[i] = True
                else:
                    canon_smiles.append(canon)
            df = df.copy()
            df["smiles"] = canon_smiles
            df = df[~drop]

            df = df.merge(chembl_dates, on="smiles", how="left")
            df = df.merge(contam[["smiles", "max_tanimoto"]], on="smiles", how="left")
            df["subset"] = [
                _subset_label(d, s, molbert_cutoff, threshold=contamination_threshold)
                for d, s in zip(df["deposit_date"].tolist(), df["max_tanimoto"].tolist())
            ]
            df["contamination_threshold_used"] = contamination_threshold

            # ChemBERTa-specific contamination label (PubChem10M corpus)
            if pubchem_contam is not None:
                df = df.merge(pubchem_contam[["smiles", "max_tanimoto_pubchem"]], on="smiles", how="left")
                df["subset_pubchem"] = [
                    _subset_label(None, t, molbert_cutoff, threshold=contamination_threshold)
                    for t in df["max_tanimoto_pubchem"].tolist()
                ]
            else:
                df["max_tanimoto_pubchem"] = float("nan")
                df["subset_pubchem"] = "unknown"

            # Protein contamination (sequence identity vs UniRef50 pre-ESM-2 cutoff)
            if prot_contam is not None:
                df = df.merge(
                    prot_contam[["uniprot_id", "max_pident", "contamination_class"]].rename(
                        columns={"contamination_class": "protein_contamination_class"}
                    ),
                    left_on="protein_key",
                    right_on="uniprot_id",
                    how="left",
                ).drop(columns=["uniprot_id"], errors="ignore")
            else:
                df["max_pident"] = float("nan")
                df["protein_contamination_class"] = "unknown"

            out_path = PROJECT_ROOT / "data" / "splits" / f"{dataset}_temporal_splits.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_path, index=False)
            logger.info("Wrote %s (%d rows).", out_path, len(df))

            test_df = df[df["split"] == "test"]
            counts = test_df["subset"].value_counts().to_dict()
            for subset_name in ["contaminated", "clean"]:
                if int(counts.get(subset_name, 0)) < 20:
                    w = f"{dataset}: subset '{subset_name}' has < 20 pairs (n={counts.get(subset_name, 0)})."
                    warnings.append(w)
                    logger.warning(w)

            subset_stats: dict[str, Any] = {}
            for subset_name in ["contaminated", "clean", "ambiguous"]:
                sub = test_df[test_df["subset"] == subset_name]
                unique_smiles = sub["smiles"].dropna().unique().tolist()
                scaffolds = [s for s in (_murcko_scaffold(smi) for smi in unique_smiles) if s]
                scaffold_count = len(set(scaffolds))
                dates = pd.to_datetime(sub["deposit_date"], errors="coerce")
                date_min = str(dates.min().date()) if dates.notna().any() else None
                date_max = str(dates.max().date()) if dates.notna().any() else None
                subset_stats[subset_name] = {
                    "pairs": int(len(sub)),
                    "unique_compounds": int(len(unique_smiles)),
                    "unique_scaffolds": int(scaffold_count),
                    "deposit_date_min": date_min,
                    "deposit_date_max": date_max,
                }

            stats[dataset] = {
                "total_pairs": int(len(df)),
                "train_pairs": int((df["split"] == "train").sum()),
                "test_pairs": int((df["split"] == "test").sum()),
                "test_subset_stats": subset_stats,
            }

        # ── ChEMBL34 clean holdout (DiD arm) ──────────────────────────────────
        holdout_path = PROJECT_ROOT / "data" / "processed" / "holdout_clean.parquet"
        if holdout_path.exists():
            holdout_df = pd.read_parquet(holdout_path)
            if not holdout_df.empty:
                holdout_df = holdout_df.rename(columns={"pchembl_value": "affinity"})
                holdout_df["dataset"] = "holdout"
                holdout_df["split"] = "test"      # entire holdout is evaluation-only
                holdout_df["subset"] = "clean"    # by construction: Tanimoto < 0.4
                holdout_df["pair_index_mode"] = "n/a"
                holdout_df["ligand_key"] = holdout_df["smiles"].astype(str)
                holdout_df["protein_key"] = holdout_df["uniprot_id"].astype(str)
                holdout_df["deposit_date"] = pd.NaT
                holdout_df["contamination_threshold_used"] = contamination_threshold
                # Holdout is novel by construction — propagate that to per-model labels
                holdout_df["subset_pubchem"] = "clean"
                holdout_df["max_tanimoto_pubchem"] = float("nan")
                # Protein contamination for holdout targets
                if prot_contam is not None:
                    holdout_df = holdout_df.merge(
                        prot_contam[["uniprot_id", "max_pident", "contamination_class"]].rename(
                            columns={"contamination_class": "protein_contamination_class"}
                        ),
                        on="uniprot_id",
                        how="left",
                    )
                else:
                    holdout_df["max_pident"] = float("nan")
                    holdout_df["protein_contamination_class"] = "unknown"

                # Validate SMILES
                canon_list: list[Optional[str]] = []
                drop_h = np.zeros(len(holdout_df), dtype=bool)
                for i, smi in enumerate(holdout_df["smiles"].astype(str).tolist()):
                    ok, canon = validate_smiles(smi)
                    if not ok or canon is None:
                        append_invalid_smiles(PROJECT_ROOT, source=phase, key=f"holdout:{i}", smiles=smi, reason="rdkit_invalid")
                        drop_h[i] = True
                        canon_list.append(None)
                    else:
                        canon_list.append(canon)
                holdout_df = holdout_df.copy()
                holdout_df["smiles"] = canon_list
                holdout_df = holdout_df[~drop_h]

                holdout_out = PROJECT_ROOT / "data" / "splits" / "holdout_temporal_splits.parquet"
                holdout_df.to_parquet(holdout_out, index=False)
                logger.info(
                    "Wrote holdout split: %d pairs, %d unique compounds, %d unique targets.",
                    len(holdout_df), holdout_df["smiles"].nunique(), holdout_df["uniprot_id"].nunique(),
                )
                stats["holdout"] = {
                    "pairs": int(len(holdout_df)),
                    "unique_compounds": int(holdout_df["smiles"].nunique()),
                    "unique_targets": int(holdout_df["uniprot_id"].nunique()),
                    "document_year_min": int(holdout_df["document_year"].min()) if "document_year" in holdout_df.columns else None,
                    "document_year_max": int(holdout_df["document_year"].max()) if "document_year" in holdout_df.columns else None,
                }
            else:
                logger.warning("holdout_clean.parquet is empty — no clean holdout arm for DiD analysis.")
                stats["holdout"] = {"pairs": 0, "note": "empty after Tanimoto filter"}
        else:
            logger.info("No holdout_clean.parquet yet — run phase1_chembl34_holdout + phase2_holdout_filter to build DiD clean arm.")

        stats["warnings"] = warnings
        stats_path = PROJECT_ROOT / "data" / "splits" / "split_statistics.json"
        stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True))
        logger.info("Wrote %s.", stats_path)

        write_sentinel(phase)
        status.append_blocking_decision(
            {
                "phase": 3,
                "question": "Review data/splits/split_statistics.json and confirm Tanimoto threshold (current 0.6) before Phase 4 starts.",
                "required_before": "phase4",
                "resolved": False,
            }
        )
        status.update(phase, "completed", progress=1.0)
        return 0

    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.error("Fatal error: %s", e)
        logger.error(tb)
        status.update(phase, "error", error=str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
