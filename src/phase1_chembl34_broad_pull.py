"""
Pull all post-2022 activities for the audited target list from ChEMBL34 SQLite.

Requires:
  - phase1_chembl34_target_audit (target list written)

Outputs:
  - data/raw/chembl34_broad_raw.parquet
    [uniprot_id, target_chembl_id, compound_chembl_id, smiles, pchembl_value,
     standard_type, document_year, sequence, protein_class_l1]

Sentinels:
  - writes checkpoints/phase1_chembl34_broad_pull.done
"""

from __future__ import annotations

import sqlite3
import traceback
from pathlib import Path

import pandas as pd

from utils.logging_utils import setup_logging
from utils.sentinel import require_sentinel, write_sentinel
from utils.status import StatusUpdater

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MIN_PCHEMBL = 5.0
STANDARD_TYPES = ("Kd", "Ki")


def _pull_activities(db_path: Path, target_chembl_ids: list[str], logger) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))

    types_clause = ",".join(f"'{t}'" for t in STANDARD_TYPES)
    # Chunk targets into batches of 500 to avoid huge IN clauses
    frames = []
    batch_size = 500
    for i in range(0, len(target_chembl_ids), batch_size):
        batch = target_chembl_ids[i:i + batch_size]
        targets_clause = ",".join(f"'{t}'" for t in batch)

        sql = f"""
        SELECT
            td.chembl_id AS target_chembl_id,
            cs.accession AS uniprot_id,
            cs.sequence,
            md.chembl_id AS compound_chembl_id,
            cpd.canonical_smiles AS smiles,
            act.pchembl_value,
            act.standard_type,
            d.year AS document_year
        FROM activities act
        JOIN assays ass ON act.assay_id = ass.assay_id
        JOIN target_dictionary td ON ass.tid = td.tid
        JOIN target_components tc ON td.tid = tc.tid
        JOIN component_sequences cs ON tc.component_id = cs.component_id
        JOIN docs d ON ass.doc_id = d.doc_id
        JOIN molecule_dictionary md ON act.molregno = md.molregno
        JOIN compound_structures cpd ON md.molregno = cpd.molregno
        WHERE td.chembl_id IN ({targets_clause})
          AND act.pchembl_value IS NOT NULL
          AND act.pchembl_value >= {MIN_PCHEMBL}
          AND act.standard_type IN ({types_clause})
          AND d.year >= 2022
          AND cpd.canonical_smiles IS NOT NULL
        """
        logger.info("  Batch %d/%d (%d targets)...", i // batch_size + 1,
                    (len(target_chembl_ids) + batch_size - 1) // batch_size, len(batch))
        df_batch = pd.read_sql(sql, conn)
        frames.append(df_batch)
        logger.info("    %d activities.", len(df_batch))

    conn.close()
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    phase = "phase1_chembl34_broad_pull"
    status = StatusUpdater(PROJECT_ROOT)
    logger = setup_logging(PROJECT_ROOT, phase)
    status.update(phase, "running", progress=0.0)

    try:
        try:
            require_sentinel("phase1_chembl34_target_audit")
        except RuntimeError as e:
            logger.error(str(e))
            status.update(phase, "blocked", error=str(e))
            return 1

        audit_path = PROJECT_ROOT / "data" / "processed" / "chembl34_target_audit.parquet"
        audit = pd.read_parquet(audit_path)
        target_ids = audit["target_chembl_id"].tolist()
        logger.info("Pulling activities for %d audited targets...", len(target_ids))

        db_path = PROJECT_ROOT / "data" / "raw" / "chembl34_sqlite" / "chembl_34.db"
        if not db_path.exists():
            raise RuntimeError(f"SQLite DB missing: {db_path}. Run target_audit first.")

        activities = _pull_activities(db_path, target_ids, logger)
        logger.info("Total activities pulled: %d", len(activities))
        logger.info("Unique compounds: %d", activities["compound_chembl_id"].nunique())
        logger.info("Unique targets (post-filter): %d", activities["target_chembl_id"].nunique())

        # Merge family labels from audit
        activities = activities.merge(
            audit[["target_chembl_id", "protein_class_l1"]],
            on="target_chembl_id",
            how="left",
        )

        out_path = PROJECT_ROOT / "data" / "raw" / "chembl34_broad_raw.parquet"
        activities.to_parquet(out_path, index=False)
        logger.info("Wrote %s (%d rows)", out_path, len(activities))

        # Summary by year and family
        logger.info("\nBy year:")
        for yr, count in activities["document_year"].value_counts().sort_index().items():
            logger.info("  %d: %d activities", int(yr), int(count))

        if "protein_class_l1" in activities.columns:
            logger.info("\nBy protein family (top 10):")
            fam_acts = activities.groupby("protein_class_l1")["pchembl_value"].count()
            for fam, count in fam_acts.sort_values(ascending=False).head(10).items():
                logger.info("  %s: %d activities", str(fam), int(count))

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
