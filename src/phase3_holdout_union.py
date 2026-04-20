"""
Union the kinase-only holdout (phase2_holdout_filter) with the broad ChEMBL34
pull (phase2_holdout_filter_broad) into a single unified holdout split.

Requires:
  - phase2_holdout_filter
  - phase2_holdout_filter_broad

Outputs:
  - data/splits/holdout_temporal_splits.parquet  (overwritten with unified set)
    same schema as before + extra columns preserved from broad pull (protein_class_l1)

Sentinels:
  - writes checkpoints/phase3_holdout_union.done
"""

from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from utils.logging_utils import setup_logging
from utils.sentinel import require_sentinel, write_sentinel
from utils.status import StatusUpdater

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    phase = "phase3_holdout_union"
    status = StatusUpdater(PROJECT_ROOT)
    logger = setup_logging(PROJECT_ROOT, phase)
    status.update(phase, "running", progress=0.0)

    try:
        for prereq in ("phase2_holdout_filter", "phase2_holdout_filter_broad"):
            try:
                require_sentinel(prereq)
            except RuntimeError as e:
                logger.warning(str(e))
                status.update(phase, "blocked", error=str(e))
                return 0

        kin = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "holdout_clean.parquet")
        brd = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "holdout_clean_broad.parquet")

        logger.info("Kinase-only: %d pairs, %d targets", len(kin), kin["uniprot_id"].nunique())
        logger.info("Broad:       %d pairs, %d targets", len(brd), brd["uniprot_id"].nunique())

        kin = kin.copy()
        kin["source"] = "kinase_ki_kd"
        if "protein_class_l1" not in kin.columns:
            kin["protein_class_l1"] = "Kinase"  # Davis/KIBA are kinase-only

        brd = brd.copy()
        brd["source"] = "chembl34_broad"

        # Union and deduplicate on (smiles, uniprot_id), preferring broad entry for
        # richer metadata (protein_class_l1) when duplicate.
        combined = pd.concat([brd, kin], ignore_index=True)
        combined = combined.drop_duplicates(subset=["smiles", "uniprot_id"], keep="first")
        logger.info("Combined union: %d pairs, %d targets, %d compounds",
                    len(combined), combined["uniprot_id"].nunique(), combined["smiles"].nunique())

        # Load the existing holdout split to inherit schema (needs affinity, dataset, split, subset, etc.)
        existing = pd.read_parquet(PROJECT_ROOT / "data" / "splits" / "holdout_temporal_splits.parquet")
        logger.info("Existing split holdout: %d rows", len(existing))

        # Build new split with same schema
        # Core required columns for downstream models: smiles, sequence, affinity, uniprot_id, split, subset, dataset
        # Use pchembl_value as affinity (higher = stronger binding, matches holdout convention)
        out = combined.copy()
        out["affinity"] = pd.to_numeric(out["pchembl_value"], errors="coerce")
        out = out.dropna(subset=["affinity"])
        out["split"] = "test"
        out["subset"] = "holdout"
        out["dataset"] = "holdout"
        out["pair_index_mode"] = "union"
        out["ligand_key"] = out["compound_chembl_id"].astype(str)
        out["protein_key"] = out["uniprot_id"].astype(str)
        out["deposit_date"] = pd.NaT
        out["contamination_threshold_used"] = 0.4
        out["max_tanimoto_pubchem"] = float("nan")
        out["subset_pubchem"] = "unknown"
        out["max_pident"] = float("nan")
        out["protein_contamination_class"] = "unknown"

        # Column order: match existing schema + extras
        out_cols = list(existing.columns)
        # Add new columns at end
        for c in ["source", "protein_class_l1"]:
            if c in out.columns and c not in out_cols:
                out_cols.append(c)
        # Only keep columns we have
        out_cols = [c for c in out_cols if c in out.columns]
        out_final = out[out_cols]

        split_path = PROJECT_ROOT / "data" / "splits" / "holdout_temporal_splits.parquet"
        # Back up existing
        backup_path = split_path.with_suffix(".parquet.kinase_backup")
        if split_path.exists() and not backup_path.exists():
            existing.to_parquet(backup_path, index=False)
            logger.info("Backed up kinase-only holdout to %s", backup_path)

        out_final.to_parquet(split_path, index=False)
        logger.info("Wrote unified holdout split: %s (%d rows)", split_path, len(out_final))

        logger.info("\nBreakdown by source:")
        for src, n in out_final["source"].value_counts().items():
            logger.info("  %s: %d pairs", str(src), int(n))

        logger.info("\nBreakdown by protein family:")
        if "protein_class_l1" in out_final.columns:
            for fam, n in out_final["protein_class_l1"].fillna("None").value_counts().items():
                logger.info("  %s: %d pairs", str(fam), int(n))

        logger.info("\nBreakdown by document year:")
        for yr, n in out_final["document_year"].value_counts().sort_index().items():
            logger.info("  %d: %d pairs", int(yr), int(n))

        write_sentinel(phase)
        status.update(phase, "completed", progress=1.0)
        return 0

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Fatal error: %s\n%s", e, tb)
        status.update(phase, "error", error=str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
