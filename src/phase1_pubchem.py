"""
Gets PubChem deposit dates for ChemBERTa contamination analysis.

Requires:
  - phase1_chembl sentinel

Outputs:
  - data/processed/pubchem_timestamps.parquet
    [smiles, pubchem_cid, create_date]

Sentinels:
  - writes checkpoints/phase1_pubchem.done
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd
import pubchempy as pcp
import requests
from tqdm import tqdm

from utils.checkpointing import (
    concat_batches_to_parquet,
    load_processed_keys,
    next_batch_index,
    write_batch,
)
from utils.http_utils import RateLimiter, request_with_backoff
from utils.logging_utils import setup_logging
from utils.sentinel import require_sentinel, write_sentinel
from utils.smiles import append_invalid_smiles, validate_smiles
from utils.status import StatusUpdater


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get_pubchem_create_date(cid: int, session: requests.Session, limiter: RateLimiter) -> Optional[str]:
    limiter.wait()
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/dates/JSON"
    resp = request_with_backoff(
        "GET",
        url,
        session=session,
        timeout=60,
        max_retries=10,
        backoff_base_s=2.0,
        backoff_max_s=120.0,
        retry_statuses=(429,),
        headers={"Accept": "application/json"},
    )
    resp.raise_for_status()
    data = resp.json()
    try:
        info = data["InformationList"]["Information"][0]
        return info.get("CreateDate")
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    phase = "phase1_pubchem"
    prereq = "phase1_chembl"
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
        in_path = PROJECT_ROOT / "data" / "processed" / "chembl_timestamps.parquet"
        df = pd.read_parquet(in_path)
        smiles_list = [str(s) for s in df["smiles"].dropna().unique().tolist()]

        limiter = RateLimiter(min_interval_s=0.2)
        session = requests.Session()

        checkpoints_dir = PROJECT_ROOT / "checkpoints"
        processed = load_processed_keys(checkpoints_dir, phase, key_cols=["smiles"])
        batch_idx = next_batch_index(checkpoints_dir, phase)
        buffer: list[dict] = []

        total = len(smiles_list)
        logger.info("Loaded %d unique SMILES from %s.", total, in_path)
        done = 0
        for smiles in tqdm(smiles_list, desc="PubChem dates", unit="cmpd"):
            is_valid, canon = validate_smiles(smiles)
            if not is_valid or canon is None:
                append_invalid_smiles(PROJECT_ROOT, source=phase, key="chembl_timestamps", smiles=smiles, reason="rdkit_invalid")
                done += 1
                continue

            if (canon,) in processed:
                done += 1
                continue

            limiter.wait()
            try:
                compounds = pcp.get_compounds(canon, "smiles")
                cid = int(compounds[0].cid) if compounds and compounds[0].cid is not None else None
            except Exception as exc:  # noqa: BLE001
                logger.warning("PubChem lookup failed for SMILES (skipping): %s — %s", canon[:80], exc)
                append_invalid_smiles(PROJECT_ROOT, source=phase, key="pubchem_lookup", smiles=smiles, reason=f"pubchem_error:{type(exc).__name__}")
                done += 1
                continue

            create_date = None
            if cid is not None:
                try:
                    create_date = _get_pubchem_create_date(cid, session=session, limiter=limiter)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("PubChem date fetch failed for CID %s (continuing): %s", cid, exc)

            buffer.append({"smiles": canon, "pubchem_cid": cid, "create_date": create_date})
            processed.add((canon,))
            done += 1

            if len(buffer) >= 200:
                write_batch(checkpoints_dir, phase, batch_idx, buffer, sort_cols=["smiles"])
                logger.info("Wrote checkpoint batch %d with %d rows.", batch_idx, len(buffer))
                status.update(phase, "checkpoint", progress=done / total)
                batch_idx += 1
                buffer = []

            if done % 50 == 0:
                status.update(phase, "running", progress=done / total)

        if buffer:
            write_batch(checkpoints_dir, phase, batch_idx, buffer, sort_cols=["smiles"])
            logger.info("Wrote final checkpoint batch %d with %d rows.", batch_idx, len(buffer))
            status.update(phase, "checkpoint", progress=1.0)

        out_path = PROJECT_ROOT / "data" / "processed" / "pubchem_timestamps.parquet"
        concat_batches_to_parquet(checkpoints_dir, phase, out_path, keep=["smiles", "pubchem_cid", "create_date"])
        logger.info("Wrote %s.", out_path)

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
