"""
Gets UniProt entry creation dates for benchmark proteins.

Outputs:
  - data/processed/uniprot_timestamps.parquet
    [uniprot_id, sequence, first_public_date, source_dataset]

Sentinels:
  - writes checkpoints/phase1_uniprot.done
"""

from __future__ import annotations

import ast
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import pandas as pd
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
from utils.sentinel import write_sentinel
from utils.status import StatusUpdater


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _download_if_missing(url: str, out_path: Path, timeout: float = 60) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        tmp.replace(out_path)


def _load_dict_literal(path: Path) -> dict[str, Any]:
    text = path.read_text()
    data = ast.literal_eval(text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict literal in {path}, got {type(data)}")
    return data


def _search_uniprot_by_gene(gene_name: str, session: requests.Session, limiter: RateLimiter) -> tuple[Optional[str], Optional[str]]:
    """Look up UniProt accession + firstPublicDate by human gene name (two-step)."""
    base_search = "https://rest.uniprot.org/uniprotkb/search"

    def _get_accession(query: str) -> Optional[str]:
        limiter.wait()
        r = request_with_backoff(
            "GET", base_search, session=session, timeout=60,
            max_retries=8, backoff_base_s=2.0, backoff_max_s=120.0,
            retry_statuses=(429, 503),
            params={"query": query, "fields": "accession", "format": "json", "size": "1"},
            headers={"Accept": "application/json"},
        )
        r.raise_for_status()
        results = r.json().get("results") or []
        if not results:
            return None
        return results[0].get("primaryAccession")

    # Try reviewed human entry first, fall back to any human entry
    acc = _get_accession(f"gene_exact:{gene_name} AND organism_id:9606 AND reviewed:true")
    if not acc:
        acc = _get_accession(f"gene_exact:{gene_name} AND organism_id:9606")
    if not acc:
        return None, None

    # Fetch full entry to get entryAudit.firstPublicDate
    limiter.wait()
    r2 = request_with_backoff(
        "GET", f"https://rest.uniprot.org/uniprotkb/{acc}.json",
        session=session, timeout=60, max_retries=6,
        backoff_base_s=2.0, backoff_max_s=120.0, retry_statuses=(429, 503),
    )
    r2.raise_for_status()
    audit = r2.json().get("entryAudit") or {}
    return acc, audit.get("firstPublicDate")


def main() -> int:
    phase = "phase1_uniprot"
    status = StatusUpdater(PROJECT_ROOT)
    logger = setup_logging(PROJECT_ROOT, phase)
    status.update(phase, "running", progress=0.0)

    try:
        raw_dir = PROJECT_ROOT / "data" / "raw"
        davis_path = raw_dir / "davis_proteins.txt"
        kiba_path = raw_dir / "kiba_proteins.txt"

        _download_if_missing(
            "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/davis/proteins.txt",
            davis_path,
        )
        _download_if_missing(
            "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/kiba/proteins.txt",
            kiba_path,
        )

        davis = _load_dict_literal(davis_path)
        kiba = _load_dict_literal(kiba_path)

        proteins: list[dict[str, str]] = []
        for ds_name, d in [("davis", davis), ("kiba", kiba)]:
            for key, seq in d.items():
                proteins.append({"source_dataset": ds_name, "key": str(key), "sequence": str(seq)})

        checkpoints_dir = PROJECT_ROOT / "checkpoints"
        processed = load_processed_keys(checkpoints_dir, phase, key_cols=["source_dataset", "sequence"])
        batch_idx = next_batch_index(checkpoints_dir, phase)
        buffer: list[dict] = []

        limiter = RateLimiter(min_interval_s=1.0)
        session = requests.Session()

        total = len(proteins)
        logger.info("Loaded %d proteins (davis=%d, kiba=%d).", total, len(davis), len(kiba))
        done = 0
        for row in tqdm(proteins, desc="UniProt dates", unit="prot"):
            ds = row["source_dataset"]
            key = row["key"]
            seq = row["sequence"]

            proc_key = (ds, seq)
            if proc_key in processed:
                done += 1
                continue

            uniprot_id, first_public_date = _search_uniprot_by_gene(key, session=session, limiter=limiter)
            buffer.append(
                {
                    "uniprot_id": uniprot_id,
                    "sequence": seq,
                    "first_public_date": first_public_date,
                    "source_dataset": ds,
                }
            )
            processed.add(proc_key)
            done += 1

            if len(buffer) >= 50:
                write_batch(checkpoints_dir, phase, batch_idx, buffer, sort_cols=["source_dataset", "uniprot_id"])
                logger.info("Wrote checkpoint batch %d with %d rows.", batch_idx, len(buffer))
                status.update(phase, "checkpoint", progress=done / total)
                batch_idx += 1
                buffer = []

            if done % 25 == 0:
                status.update(phase, "running", progress=done / total)

        if buffer:
            write_batch(checkpoints_dir, phase, batch_idx, buffer, sort_cols=["source_dataset", "uniprot_id"])
            logger.info("Wrote final checkpoint batch %d with %d rows.", batch_idx, len(buffer))
            status.update(phase, "checkpoint", progress=1.0)

        out_path = PROJECT_ROOT / "data" / "processed" / "uniprot_timestamps.parquet"
        concat_batches_to_parquet(
            checkpoints_dir,
            phase,
            out_path,
            keep=["uniprot_id", "sequence", "first_public_date", "source_dataset"],
        )
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

