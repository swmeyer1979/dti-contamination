"""
Tanimoto<0.4 novelty filter for the broad ChEMBL34 pull.

Same Tanimoto-gate logic as phase2_holdout_filter but operates on the broad target
set (phase1_chembl34_broad_pull output) and preserves protein-family labels + year
for stratified evaluation.

Requires:
  - phase1_chembl34_broad_pull
  - phase2_fpsim2

Outputs:
  - data/processed/holdout_clean_broad.parquet
    [uniprot_id, sequence, smiles, pchembl_value, standard_type, document_year,
     max_tanimoto, compound_chembl_id, protein_class_l1, target_chembl_id]

Sentinels:
  - writes checkpoints/phase2_holdout_filter_broad.done
"""

from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.logging_utils import setup_logging
from utils.sentinel import require_sentinel, write_sentinel
from utils.smiles import append_invalid_smiles, validate_smiles
from utils.status import StatusUpdater


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TANIMOTO_THRESHOLD = 0.4


def main() -> int:
    phase = "phase2_holdout_filter_broad"
    status = StatusUpdater(PROJECT_ROOT)
    logger = setup_logging(PROJECT_ROOT, phase)
    status.update(phase, "running", progress=0.0)

    try:
        for prereq in ("phase1_chembl34_broad_pull", "phase2_fpsim2"):
            try:
                require_sentinel(prereq)
            except RuntimeError as e:
                logger.warning(str(e))
                status.update(phase, "blocked", error=str(e))
                return 0

        raw_path = PROJECT_ROOT / "data" / "raw" / "chembl34_broad_raw.parquet"
        df = pd.read_parquet(raw_path)
        logger.info("Loaded %d raw broad records.", len(df))
        logger.info("  Unique targets: %d", df["target_chembl_id"].nunique())
        logger.info("  Unique compounds: %d", df["compound_chembl_id"].nunique())

        if df.empty:
            logger.warning("Broad raw is empty.")
            write_sentinel(phase)
            return 0

        # Canonicalize SMILES
        logger.info("Validating and canonicalizing SMILES...")
        canon_smiles: list[str | None] = []
        drop = np.zeros(len(df), dtype=bool)
        for i, smi in enumerate(df["smiles"].astype(str).tolist()):
            ok, canon = validate_smiles(smi)
            if not ok or canon is None:
                append_invalid_smiles(PROJECT_ROOT, source=phase, key=f"broad_row_{i}",
                                      smiles=smi, reason="rdkit_invalid")
                drop[i] = True
                canon_smiles.append(None)
            else:
                canon_smiles.append(canon)
        df = df.copy()
        df["smiles"] = canon_smiles
        df = df[~drop].dropna(subset=["smiles"])
        logger.info("%d records after SMILES validation.", len(df))

        # Deduplicate: keep highest pChEMBL per (smiles, uniprot_id)
        df = (
            df.sort_values("pchembl_value", ascending=False)
            .drop_duplicates(subset=["smiles", "uniprot_id"])
            .reset_index(drop=True)
        )
        logger.info("%d records after deduplication.", len(df))

        # Tanimoto vs ChEMBL27
        db_path = PROJECT_ROOT / "data" / "raw" / "chembl27_fps.h5"
        if not db_path.exists():
            raise RuntimeError(f"FPSim2 DB not found at {db_path}. Run phase2_fpsim2 first.")

        try:
            from FPSim2 import FPSim2Engine
        except Exception as e:
            raise RuntimeError("FPSim2Engine import failed.") from e

        engine = FPSim2Engine(str(db_path))
        unique_smiles = df["smiles"].unique().tolist()
        logger.info("Computing Tanimoto to ChEMBL27 for %d unique SMILES...", len(unique_smiles))

        smi_to_tanimoto: dict[str, float] = {}
        for j, smi in enumerate(tqdm(unique_smiles, desc="Tanimoto", unit="cmpd")):
            try:
                results = engine.similarity(smi, 0.0, n_workers=1)
                smi_to_tanimoto[smi] = float(results["coeff"].max()) if len(results) > 0 else 0.0
            except Exception:
                smi_to_tanimoto[smi] = float("nan")
            if (j + 1) % 100 == 0:
                status.update(phase, "running", progress=0.1 + 0.85 * (j + 1) / len(unique_smiles))

        df["max_tanimoto"] = df["smiles"].map(smi_to_tanimoto)
        df = df.dropna(subset=["max_tanimoto"])

        before = len(df)
        df_clean = df[df["max_tanimoto"] < TANIMOTO_THRESHOLD].reset_index(drop=True)
        logger.info(
            "Tanimoto < %.1f filter: %d → %d records (%.1f%% retained).",
            TANIMOTO_THRESHOLD, before, len(df_clean),
            100.0 * len(df_clean) / max(1, before),
        )

        out_cols = [
            "target_chembl_id", "uniprot_id", "sequence", "smiles", "pchembl_value",
            "standard_type", "document_year", "max_tanimoto", "compound_chembl_id",
            "protein_class_l1",
        ]
        out_path = PROJECT_ROOT / "data" / "processed" / "holdout_clean_broad.parquet"
        df_clean[[c for c in out_cols if c in df_clean.columns]].to_parquet(out_path, index=False)
        logger.info(
            "Wrote %s: %d pairs, %d unique compounds, %d unique targets.",
            out_path, len(df_clean),
            df_clean["smiles"].nunique(),
            df_clean["uniprot_id"].nunique(),
        )

        # Distribution summaries
        logger.info("\nDistribution by protein family:")
        if "protein_class_l1" in df_clean.columns:
            fam_n = df_clean.groupby("protein_class_l1").agg(
                n_pairs=("smiles", "size"),
                n_targets=("uniprot_id", "nunique"),
                n_compounds=("smiles", "nunique"),
            ).sort_values("n_pairs", ascending=False)
            for fam, row in fam_n.iterrows():
                logger.info("  %s: pairs=%d, targets=%d, compounds=%d",
                            str(fam), int(row["n_pairs"]),
                            int(row["n_targets"]), int(row["n_compounds"]))

        logger.info("\nDistribution by document year:")
        for yr, count in df_clean["document_year"].value_counts().sort_index().items():
            logger.info("  %d: %d pairs", int(yr), int(count))

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
