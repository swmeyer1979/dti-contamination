"""
Fetches post-2021 bioactivity data from ChEMBL34 for Davis/KIBA kinase targets.

Design: ChEMBL34 (released March 2024) contains activities deposited through 2023.
We pull Ki/Kd records for the same UniProt targets as Davis/KIBA, published in
documents dated >= 2022 (conservative post-cutoff: ESM-2 Apr 2021, MolBERT Nov 2020,
ChemBERTa Jun 2020). These form the clean holdout arm for the DiD causal analysis.

Requires:
  - phase1_uniprot  (for UniProt ID list)

Outputs:
  - data/raw/chembl34_holdout_raw.parquet
    [uniprot_id, target_chembl_id, compound_chembl_id, smiles,
     pchembl_value, standard_type, document_year, sequence]

Sentinels:
  - writes checkpoints/phase1_chembl34_holdout.done
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from tqdm import tqdm

from utils.checkpointing import load_processed_keys, next_batch_index, write_batch, concat_batches_to_parquet
from utils.http_utils import RateLimiter, request_with_backoff
from utils.logging_utils import setup_logging
from utils.sentinel import require_sentinel, write_sentinel
from utils.status import StatusUpdater


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
# Conservative cutoff: all three pretraining corpora are pre-2021
DOCUMENT_YEAR_GTE = 2022
PCHEMBL_MIN = 5.0
STANDARD_TYPES = "Kd,Ki"
PAGE_SIZE = 1000


def _get_chembl_target_id(
    uniprot_id: str,
    session: requests.Session,
    limiter: RateLimiter,
    logger,
) -> Optional[str]:
    """Map UniProt accession → ChEMBL target ID (single protein targets only)."""
    limiter.wait()
    url = (
        f"{CHEMBL_API}/target.json"
        f"?target_components__accession={uniprot_id}"
        f"&target_type=SINGLE+PROTEIN"
        f"&limit=5"
    )
    resp = request_with_backoff("GET", url, session=session, timeout=30,
                                max_retries=5, backoff_base_s=2.0, backoff_max_s=60.0,
                                retry_statuses=(429, 503))
    resp.raise_for_status()
    data = resp.json()
    targets = data.get("targets", [])
    if not targets:
        logger.debug("No ChEMBL target found for UniProt %s", uniprot_id)
        return None
    return str(targets[0]["target_chembl_id"])


def _fetch_activities_for_target(
    target_chembl_id: str,
    session: requests.Session,
    limiter: RateLimiter,
    logger,
) -> list[dict[str, Any]]:
    """Paginate through ChEMBL34 activities for a target, post-cutoff only."""
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        limiter.wait()
        url = (
            f"{CHEMBL_API}/activity.json"
            f"?target_chembl_id={target_chembl_id}"
            f"&standard_type__in={STANDARD_TYPES}"
            f"&pchembl_value__isnull=false"
            f"&pchembl_value__gte={PCHEMBL_MIN}"
            f"&document_year__gte={DOCUMENT_YEAR_GTE}"
            f"&assay_type=B"  # binding assays only
            f"&limit={PAGE_SIZE}&offset={offset}"
        )
        resp = request_with_backoff("GET", url, session=session, timeout=60,
                                    max_retries=5, backoff_base_s=2.0, backoff_max_s=120.0,
                                    retry_statuses=(429, 503))
        resp.raise_for_status()
        data = resp.json()
        activities = data.get("activities", [])
        for act in activities:
            smiles = act.get("canonical_smiles") or act.get("molecule_structures", {})
            if isinstance(smiles, dict):
                smiles = smiles.get("canonical_smiles")
            if not smiles:
                continue
            pchembl = act.get("pchembl_value")
            if pchembl is None:
                continue
            rows.append({
                "target_chembl_id": target_chembl_id,
                "compound_chembl_id": act.get("molecule_chembl_id", ""),
                "smiles": str(smiles),
                "pchembl_value": float(pchembl),
                "standard_type": str(act.get("standard_type", "")),
                "document_year": int(act.get("document_year") or 0),
            })
        page_meta = data.get("page_meta", {})
        total = int(page_meta.get("total_count", 0))
        if offset + PAGE_SIZE >= total:
            break
        offset += PAGE_SIZE
    return rows


def main() -> int:
    phase = "phase1_chembl34_holdout"
    prereq = "phase1_uniprot"
    status = StatusUpdater(PROJECT_ROOT)
    logger = setup_logging(PROJECT_ROOT, phase)
    status.update(phase, "running", progress=0.0)

    try:
        try:
            require_sentinel(prereq)
        except RuntimeError as e:
            logger.warning(str(e))
            status.update(phase, "blocked", error=str(e))
            return 0

        # Load UniProt IDs + sequences from phase1_uniprot output
        uniprot_df = pd.read_parquet(
            PROJECT_ROOT / "data" / "processed" / "uniprot_timestamps.parquet"
        )
        uniprot_df = uniprot_df.dropna(subset=["uniprot_id", "sequence"])
        uniprot_df = uniprot_df.drop_duplicates(subset=["uniprot_id"])
        uniprot_ids = uniprot_df["uniprot_id"].tolist()
        id_to_seq = dict(zip(uniprot_df["uniprot_id"], uniprot_df["sequence"]))
        logger.info("Loaded %d unique UniProt targets from phase1_uniprot.", len(uniprot_ids))

        checkpoints_dir = PROJECT_ROOT / "checkpoints"
        processed = load_processed_keys(checkpoints_dir, phase, key_cols=["uniprot_id"])
        batch_idx = next_batch_index(checkpoints_dir, phase)
        buffer: list[dict] = []

        # 5 req/s max for ChEMBL API (be conservative)
        limiter = RateLimiter(min_interval_s=0.25)
        session = requests.Session()
        session.headers["Accept"] = "application/json"

        total = len(uniprot_ids)
        target_id_cache: dict[str, Optional[str]] = {}

        for i, uniprot_id in enumerate(tqdm(uniprot_ids, desc="ChEMBL34 holdout", unit="target")):
            if (uniprot_id,) in processed:
                continue

            # Step 1: map UniProt → ChEMBL target ID
            if uniprot_id not in target_id_cache:
                try:
                    target_chembl_id = _get_chembl_target_id(uniprot_id, session, limiter, logger)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Target lookup failed for %s: %s", uniprot_id, exc)
                    target_chembl_id = None
                target_id_cache[uniprot_id] = target_chembl_id

            target_chembl_id = target_id_cache[uniprot_id]
            if target_chembl_id is None:
                processed.add((uniprot_id,))
                continue

            # Step 2: fetch post-cutoff activities
            try:
                activities = _fetch_activities_for_target(target_chembl_id, session, limiter, logger)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Activity fetch failed for %s (%s): %s", uniprot_id, target_chembl_id, exc)
                activities = []

            seq = id_to_seq.get(uniprot_id, "")
            for act in activities:
                act["uniprot_id"] = uniprot_id
                act["sequence"] = seq
                buffer.append(act)

            processed.add((uniprot_id,))
            logger.info(
                "%s → %s: %d activities (total buffer: %d)",
                uniprot_id, target_chembl_id, len(activities), len(buffer),
            )

            if len(buffer) >= 500:
                write_batch(checkpoints_dir, phase, batch_idx, buffer, sort_cols=["uniprot_id", "smiles"])
                logger.info("Checkpoint batch %d: %d rows.", batch_idx, len(buffer))
                status.update(phase, "checkpoint", progress=(i + 1) / total)
                batch_idx += 1
                buffer = []

            if (i + 1) % 10 == 0:
                status.update(phase, "running", progress=(i + 1) / total)

        if buffer:
            write_batch(checkpoints_dir, phase, batch_idx, buffer, sort_cols=["uniprot_id", "smiles"])
            logger.info("Final checkpoint batch %d: %d rows.", batch_idx, len(buffer))

        out_path = PROJECT_ROOT / "data" / "raw" / "chembl34_holdout_raw.parquet"
        concat_batches_to_parquet(
            checkpoints_dir, phase, out_path,
            keep=["uniprot_id", "target_chembl_id", "compound_chembl_id",
                  "smiles", "pchembl_value", "standard_type", "document_year", "sequence"],
        )
        logger.info("Wrote %s.", out_path)

        n_rows = len(pd.read_parquet(out_path))
        n_targets = uniprot_df["uniprot_id"].nunique()
        logger.info(
            "ChEMBL34 holdout: %d activities across %d targets (document_year >= %d).",
            n_rows, n_targets, DOCUMENT_YEAR_GTE,
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
