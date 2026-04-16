"""
Maps Davis and KIBA benchmark compounds to ChEMBL IDs and retrieves first-deposit timestamps.

Outputs:
  - data/processed/chembl_timestamps.parquet
    [smiles, chembl_id, first_known_date, source_dataset]

Sentinels:
  - writes checkpoints/phase1_chembl.done
"""

from __future__ import annotations

import ast
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

from utils.chembl_client import ChEMBLClient
from utils.checkpointing import (
    concat_batches_to_parquet,
    load_processed_keys,
    next_batch_index,
    write_batch,
)
from utils.logging_utils import setup_logging
from utils.sentinel import write_sentinel
from utils.smiles import append_invalid_smiles, validate_smiles
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


def main() -> int:
    phase = "phase1_chembl"
    status = StatusUpdater(PROJECT_ROOT)
    logger = setup_logging(PROJECT_ROOT, phase)
    status.update(phase, "running", progress=0.0)

    try:
        raw_dir = PROJECT_ROOT / "data" / "raw"
        davis_path = raw_dir / "davis_ligands.txt"
        kiba_path = raw_dir / "kiba_ligands.txt"

        _download_if_missing(
            "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/davis/ligands_can.txt",
            davis_path,
        )
        _download_if_missing(
            "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/kiba/ligands_can.txt",
            kiba_path,
        )

        davis = _load_dict_literal(davis_path)
        kiba = _load_dict_literal(kiba_path)

        compounds: list[dict[str, str]] = []
        for ds_name, d in [("davis", davis), ("kiba", kiba)]:
            for key, smi in d.items():
                compounds.append({"source_dataset": ds_name, "key": str(key), "smiles": str(smi)})

        client = ChEMBLClient(min_interval_s=0.1, cache_dir=PROJECT_ROOT / "checkpoints" / "chembl_cache")
        checkpoints_dir = PROJECT_ROOT / "checkpoints"
        processed = load_processed_keys(checkpoints_dir, phase, key_cols=["source_dataset", "smiles"])
        batch_idx = next_batch_index(checkpoints_dir, phase)
        buffer: list[dict] = []

        total = len(compounds)
        logger.info("Loaded %d compounds (davis=%d, kiba=%d).", total, len(davis), len(kiba))
        pbar = tqdm(compounds, desc="ChEMBL mapping", unit="cmpd")
        done = 0

        for row in pbar:
            ds = row["source_dataset"]
            key = row["key"]
            smiles = row["smiles"]

            is_valid, canon = validate_smiles(smiles)
            if not is_valid or canon is None:
                append_invalid_smiles(PROJECT_ROOT, source=phase, key=f"{ds}:{key}", smiles=smiles, reason="rdkit_invalid")
                done += 1
                continue

            proc_key = (ds, canon)
            if proc_key in processed:
                done += 1
                continue

            chembl_ids = client.search_by_smiles(canon)
            chembl_id = chembl_ids[0] if chembl_ids else None
            first_date = client.get_first_known_date(chembl_id) if chembl_id else None

            buffer.append(
                {
                    "smiles": canon,
                    "chembl_id": chembl_id,
                    "first_known_date": first_date,
                    "source_dataset": ds,
                }
            )
            processed.add(proc_key)
            done += 1

            if len(buffer) >= 100:
                write_batch(checkpoints_dir, phase, batch_idx, buffer, sort_cols=["source_dataset", "smiles"])
                logger.info("Wrote checkpoint batch %d with %d rows.", batch_idx, len(buffer))
                status.update(phase, "checkpoint", progress=done / total)
                batch_idx += 1
                buffer = []

            if done % 50 == 0:
                status.update(phase, "running", progress=done / total)

        if buffer:
            write_batch(checkpoints_dir, phase, batch_idx, buffer, sort_cols=["source_dataset", "smiles"])
            logger.info("Wrote final checkpoint batch %d with %d rows.", batch_idx, len(buffer))
            status.update(phase, "checkpoint", progress=1.0)

        out_path = PROJECT_ROOT / "data" / "processed" / "chembl_timestamps.parquet"
        concat_batches_to_parquet(
            checkpoints_dir,
            phase,
            out_path,
            keep=["smiles", "chembl_id", "first_known_date", "source_dataset"],
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

