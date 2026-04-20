"""
Query ChEMBL34 SQLite dump for all single-protein human targets with sufficient
post-2022 activities meeting pChEMBL threshold.

This is the target-discovery step for the DTI-Clean-2024 broader benchmark pull.
Output is a target list that downstream phases use to fetch activities,
compounds, and sequences.

Pipeline:
  1. Extract compressed SQLite dump (~5 GB compressed → ~25 GB uncompressed)
  2. Query: all SINGLE_PROTEIN targets with pref_name.organism = 'Homo sapiens'
     having ≥ MIN_ACTIVITIES post-2022 Ki/Kd activities with pchembl_value ≥ 5.0
  3. Cross-reference with UniProt for sequences
  4. Write target list parquet

Requires:
  - data/raw/chembl34_sqlite/chembl_34_sqlite.tar.gz (pre-downloaded)

Outputs:
  - data/processed/chembl34_target_audit.parquet
    [target_chembl_id, uniprot_id, organism, preferred_name, protein_class_l1,
     protein_class_l2, n_post2022_activities, n_unique_compounds]

Sentinels:
  - writes checkpoints/phase1_chembl34_target_audit.done
"""

from __future__ import annotations

import sqlite3
import subprocess
import tarfile
import traceback
from pathlib import Path

import pandas as pd

from utils.logging_utils import setup_logging
from utils.sentinel import require_sentinel, write_sentinel
from utils.status import StatusUpdater

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MIN_POST2022_ACTIVITIES = 50
MIN_PCHEMBL = 5.0
# Include IC50 alongside Ki/Kd: most DTI benchmarks do, and restricting to only
# biochemical binding measurements drops the target count from ~500 to ~42.
STANDARD_TYPES = ("Kd", "Ki", "IC50")


def _extract_sqlite(tar_path: Path, out_dir: Path, logger) -> Path:
    """Extract chembl_34.db from the tarball. Uses streaming extraction to avoid
    loading the full tar into memory."""
    db_path = out_dir / "chembl_34.db"
    if db_path.exists() and db_path.stat().st_size > 10_000_000_000:
        logger.info("SQLite db already extracted at %s (%.1f GB), skipping.",
                    db_path, db_path.stat().st_size / 1e9)
        return db_path

    logger.info("Extracting %s into %s (this takes 5-10 min)...", tar_path, out_dir)
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            if member.name.endswith(".db"):
                # Rename to fixed path regardless of inner directory
                logger.info("  Found %s (%.1f GB)", member.name, member.size / 1e9)
                extracted = tf.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"Could not extract {member.name}")
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(db_path, "wb") as fout:
                    # 64 MB chunks
                    while True:
                        chunk = extracted.read(64 * 1024 * 1024)
                        if not chunk:
                            break
                        fout.write(chunk)
                logger.info("  Extracted to %s (%.1f GB)", db_path, db_path.stat().st_size / 1e9)
                return db_path

    raise RuntimeError("No .db file found in tarball")


def _audit_targets(db_path: Path, logger) -> pd.DataFrame:
    """Query SQLite for all single-protein human targets with sufficient activities.

    Schema notes (ChEMBL34):
      - target_dictionary: target_chembl_id, pref_name, organism, target_type
      - target_components: tid, component_id
      - component_sequences: component_id, accession (UniProt), sequence
      - activities: activity_id, molregno, assay_id, standard_type, pchembl_value
      - assays: assay_id, tid, doc_id
      - docs: doc_id, year
      - protein_classification (via component_class): for protein family labels
    """
    logger.info("Opening SQLite: %s", db_path)
    conn = sqlite3.connect(str(db_path))

    # Step 1: identify candidate targets
    logger.info("Identifying candidate single-protein human targets...")
    candidates_sql = """
    SELECT
        td.tid,
        td.chembl_id AS target_chembl_id,
        td.pref_name,
        cs.accession AS uniprot_id,
        cs.sequence
    FROM target_dictionary td
    JOIN target_components tc ON td.tid = tc.tid
    JOIN component_sequences cs ON tc.component_id = cs.component_id
    WHERE td.target_type = 'SINGLE PROTEIN'
      AND td.organism = 'Homo sapiens'
      AND cs.accession IS NOT NULL
      AND cs.sequence IS NOT NULL
    """
    candidates = pd.read_sql(candidates_sql, conn)
    logger.info("  Found %d single-protein human targets.", len(candidates))

    # Step 2: count qualifying post-2022 activities per target
    logger.info("Counting post-2022 activities per target (≥%s, pchembl ≥ %s, %s)...",
                MIN_POST2022_ACTIVITIES, MIN_PCHEMBL, STANDARD_TYPES)
    types_clause = ",".join(f"'{t}'" for t in STANDARD_TYPES)
    activity_sql = f"""
    SELECT
        ass.tid,
        COUNT(DISTINCT act.molregno) AS n_unique_compounds,
        COUNT(*) AS n_activities
    FROM activities act
    JOIN assays ass ON act.assay_id = ass.assay_id
    JOIN docs d ON ass.doc_id = d.doc_id
    WHERE act.pchembl_value IS NOT NULL
      AND act.pchembl_value >= {MIN_PCHEMBL}
      AND act.standard_type IN ({types_clause})
      AND d.year >= 2022
    GROUP BY ass.tid
    HAVING COUNT(*) >= {MIN_POST2022_ACTIVITIES}
    """
    activity_counts = pd.read_sql(activity_sql, conn)
    logger.info("  Found %d targets with ≥%d qualifying activities.",
                len(activity_counts), MIN_POST2022_ACTIVITIES)

    # Step 3: protein classification
    logger.info("Pulling protein classification (family labels)...")
    class_sql = """
    SELECT
        tc.tid,
        pc.pref_name AS protein_class_l1
    FROM target_components tc
    JOIN component_class cc ON tc.component_id = cc.component_id
    JOIN protein_classification pc ON cc.protein_class_id = pc.protein_class_id
    WHERE pc.class_level = 1
    """
    try:
        classifications = pd.read_sql(class_sql, conn)
        # Keep first class per target
        classifications = classifications.drop_duplicates(subset=["tid"], keep="first")
    except Exception as e:
        logger.warning("Could not query protein_classification: %s", e)
        classifications = pd.DataFrame(columns=["tid", "protein_class_l1"])

    # Merge — inner with activity_counts (must meet threshold), LEFT with classifications
    # (don't lose targets with missing family labels; fill with 'Unclassified')
    result = (
        candidates
        .merge(activity_counts, on="tid", how="inner")
        .merge(classifications, on="tid", how="left")
    )
    result["protein_class_l1"] = result["protein_class_l1"].fillna("Unclassified")
    result = result[[
        "target_chembl_id", "uniprot_id", "pref_name", "protein_class_l1",
        "n_activities", "n_unique_compounds", "sequence"
    ]].rename(columns={"pref_name": "preferred_name", "n_activities": "n_post2022_activities"})

    conn.close()
    logger.info("Final target audit: %d targets", len(result))

    # Family distribution
    if "protein_class_l1" in result.columns:
        fam_counts = result["protein_class_l1"].value_counts(dropna=False)
        logger.info("Family distribution:")
        for fam, n in fam_counts.items():
            logger.info("  %s: %d targets", str(fam), n)

    return result


def main() -> int:
    phase = "phase1_chembl34_target_audit"
    status = StatusUpdater(PROJECT_ROOT)
    logger = setup_logging(PROJECT_ROOT, phase)
    status.update(phase, "running", progress=0.0)

    try:
        sqlite_dir = PROJECT_ROOT / "data" / "raw" / "chembl34_sqlite"
        tar_path = sqlite_dir / "chembl_34_sqlite.tar.gz"
        if not tar_path.exists():
            msg = (
                f"ChEMBL34 SQLite tarball not found at {tar_path}. "
                "Download from https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_34/"
            )
            logger.error(msg)
            status.update(phase, "blocked", error=msg)
            return 1

        db_path = _extract_sqlite(tar_path, sqlite_dir, logger)
        status.update(phase, "running", progress=0.4)

        result = _audit_targets(db_path, logger)
        status.update(phase, "running", progress=0.9)

        out_path = PROJECT_ROOT / "data" / "processed" / "chembl34_target_audit.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(out_path, index=False)
        logger.info("Wrote %s (%d targets)", out_path, len(result))

        write_sentinel(phase)
        status.update(phase, "completed", progress=1.0)
        return 0

    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.error("Fatal error: %s\n%s", e, tb)
        status.update(phase, "error", error=str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
