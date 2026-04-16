"""
Filters ChEMBL34 holdout compounds to Tanimoto < 0.4 vs ChEMBL27 corpus.

A compound passes if its max Morgan fingerprint Tanimoto similarity to any
ChEMBL27 compound is below 0.4 — ensuring it was structurally novel at the
time MolBERT/ChemBERTa were pretrained. This is the structural novelty gate
for the DiD clean holdout arm.

Requires:
  - phase1_chembl34_holdout
  - phase2_fpsim2  (ChEMBL27 fingerprint DB already built)

Outputs:
  - data/processed/holdout_clean.parquet
    [uniprot_id, sequence, smiles, pchembl_value, standard_type,
     document_year, max_tanimoto, compound_chembl_id]

Sentinels:
  - writes checkpoints/phase2_holdout_filter.done
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
TANIMOTO_THRESHOLD = 0.4   # compounds with max_tanimoto < this are "clean"


def main() -> int:
    phase = "phase2_holdout_filter"
    status = StatusUpdater(PROJECT_ROOT)
    logger = setup_logging(PROJECT_ROOT, phase)
    status.update(phase, "running", progress=0.0)

    try:
        for prereq in ("phase1_chembl34_holdout", "phase2_fpsim2"):
            try:
                require_sentinel(prereq)
            except RuntimeError as e:
                logger.warning(str(e))
                status.update(phase, "blocked", error=str(e))
                return 0

        raw_path = PROJECT_ROOT / "data" / "raw" / "chembl34_holdout_raw.parquet"
        df = pd.read_parquet(raw_path)
        logger.info("Loaded %d raw holdout records.", len(df))

        if df.empty:
            logger.warning("ChEMBL34 holdout is empty. Possible API issue or no matching records.")
            # Write empty output so downstream phases don't crash
            out_path = PROJECT_ROOT / "data" / "processed" / "holdout_clean.parquet"
            pd.DataFrame(columns=["uniprot_id","sequence","smiles","pchembl_value",
                                   "standard_type","document_year","max_tanimoto",
                                   "compound_chembl_id"]).to_parquet(out_path, index=False)
            write_sentinel(phase)
            status.update(phase, "completed", progress=1.0)
            return 0

        # Canonicalize SMILES
        logger.info("Validating and canonicalizing SMILES...")
        canon_smiles: list[str | None] = []
        drop = np.zeros(len(df), dtype=bool)
        for i, smi in enumerate(df["smiles"].astype(str).tolist()):
            ok, canon = validate_smiles(smi)
            if not ok or canon is None:
                append_invalid_smiles(PROJECT_ROOT, source=phase, key=f"chembl34_row_{i}",
                                      smiles=smi, reason="rdkit_invalid")
                drop[i] = True
                canon_smiles.append(None)
            else:
                canon_smiles.append(canon)
        df = df.copy()
        df["smiles"] = canon_smiles
        df = df[~drop].dropna(subset=["smiles"])
        logger.info("%d records after SMILES validation.", len(df))

        # Deduplicate by (smiles, uniprot_id) — keep highest pChEMBL
        df = (
            df.sort_values("pchembl_value", ascending=False)
            .drop_duplicates(subset=["smiles", "uniprot_id"])
            .reset_index(drop=True)
        )
        logger.info("%d records after deduplication.", len(df))

        # Compute max Tanimoto to ChEMBL27 using FPSim2
        db_path = PROJECT_ROOT / "data" / "raw" / "chembl27_fps.h5"
        if not db_path.exists():
            raise RuntimeError(
                f"FPSim2 DB not found at {db_path}. Run phase2_fpsim2 first."
            )

        try:
            from FPSim2 import FPSim2Engine  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("FPSim2Engine import failed.") from e

        engine = FPSim2Engine(str(db_path))
        unique_smiles = df["smiles"].unique().tolist()
        logger.info("Computing Tanimoto similarity for %d unique SMILES...", len(unique_smiles))

        smi_to_tanimoto: dict[str, float] = {}
        for j, smi in enumerate(tqdm(unique_smiles, desc="Tanimoto filter", unit="cmpd")):
            try:
                results = engine.similarity(smi, 0.0, n_workers=1)
                if len(results) > 0:
                    smi_to_tanimoto[smi] = float(results["coeff"].max())
                else:
                    smi_to_tanimoto[smi] = 0.0
            except Exception:  # noqa: BLE001
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

        if len(df_clean) == 0:
            logger.warning(
                "No holdout records passed Tanimoto < %.1f filter. "
                "The clean holdout is empty — DiD analysis will not have a clean arm. "
                "Consider lowering the pChEMBL threshold or expanding the target list.",
                TANIMOTO_THRESHOLD,
            )

        out_cols = ["uniprot_id", "sequence", "smiles", "pchembl_value",
                    "standard_type", "document_year", "max_tanimoto", "compound_chembl_id"]
        out_path = PROJECT_ROOT / "data" / "processed" / "holdout_clean.parquet"
        df_clean[[c for c in out_cols if c in df_clean.columns]].to_parquet(out_path, index=False)
        logger.info(
            "Wrote %s: %d clean holdout pairs, %d unique compounds, %d unique targets.",
            out_path, len(df_clean),
            df_clean["smiles"].nunique(),
            df_clean["uniprot_id"].nunique(),
        )

        write_sentinel(phase)
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
