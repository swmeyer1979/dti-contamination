"""
Computes max Tanimoto similarity from benchmark compounds to ChEMBL27 corpus.

Requires:
  - phase1_chembl

Outputs:
  - data/processed/compound_contamination.parquet
    [smiles, max_tanimoto, nearest_neighbor_id, contamination_class, cutoff_used]

Sentinels:
  - writes checkpoints/phase2_fpsim2.done
"""

from __future__ import annotations

import gzip
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

from utils.logging_utils import setup_logging
from utils.sentinel import require_sentinel, write_sentinel
from utils.smiles import append_invalid_smiles, validate_smiles
from utils.status import StatusUpdater


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _download_stream(url: str, out_path: Path, chunk_size: int = 8192) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0") or "0")
        with tmp.open("wb") as f, tqdm(
            total=total if total > 0 else None,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {out_path.name}",
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))
    try:
        tmp.replace(out_path)
    except FileNotFoundError:
        # another process beat us to the rename — if out_path now exists, we're done
        if out_path.exists() and out_path.stat().st_size > 0:
            return
        raise


def _fpsim2_create_db(smiles_list: list[str], db_path: Path, logger) -> None:
    try:
        from FPSim2.io import create_db_file  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("FPSim2.io.create_db_file import failed.") from e
    logger.info("Building FPSim2 DB at %s (%d SMILES, this can take a while).", db_path, len(smiles_list))
    # mols_source: iterable of (smiles_str, int_id) tuples; mol_format must be "smiles"
    smiles_pairs = [(smi, i) for i, smi in enumerate(smiles_list)]
    create_db_file(
        smiles_pairs,
        str(db_path),
        mol_format="smiles",
        fp_type="Morgan",
        fp_params={"radius": 2, "fpSize": 2048},
    )
    logger.info("Built FPSim2 DB.")


def _build_smiles_corpus(raw_gz_path: Path, logger) -> list[str]:
    smiles: list[str] = []
    with gzip.open(raw_gz_path, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            smi_idx = header.index("canonical_smiles")
        except ValueError:
            try:
                smi_idx = header.index("canonical_smiles\t")  # defensive
            except ValueError as e:
                raise RuntimeError(
                    f"canonical_smiles column not found in {raw_gz_path}. Header columns: {header[:20]}"
                ) from e

        for line in tqdm(f, desc="Parsing ChEMBL27 chemreps", unit="lines"):
            parts = line.rstrip("\n").split("\t")
            if smi_idx >= len(parts):
                continue
            smi = parts[smi_idx]
            ok, canon = validate_smiles(smi)
            if not ok or canon is None:
                append_invalid_smiles(PROJECT_ROOT, source="phase2_fpsim2", key="chembl27_chemreps", smiles=smi, reason="rdkit_invalid")
                continue
            smiles.append(canon)

    logger.info("Validated %d canonical SMILES from ChEMBL27 chemreps.", len(smiles))
    return smiles


def _classify_tanimoto(sim: float) -> str:
    if sim >= 0.9:
        return "highly_contaminated"
    if sim >= 0.6:
        return "likely_contaminated"
    return "clean"


def main() -> int:
    phase = "phase2_fpsim2"
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

        cutoffs = json.loads((PROJECT_ROOT / "config" / "cutoffs.json").read_text())
        cutoff_used = 0.6

        raw_dir = PROJECT_ROOT / "data" / "raw"
        chemreps_gz = raw_dir / "chembl27_chemreps.txt.gz"
        if not chemreps_gz.exists():
            _download_stream(
                "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_27/chembl_27_chemreps.txt.gz",
                chemreps_gz,
            )

        db_path = raw_dir / "chembl27_fps.h5"
        if not db_path.exists():
            smiles_corpus = _build_smiles_corpus(chemreps_gz, logger=logger)
            _fpsim2_create_db(smiles_corpus, db_path, logger=logger)

        try:
            from FPSim2 import FPSim2Engine  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("FPSim2Engine import failed. Check FPSim2 installation.") from e

        engine = FPSim2Engine(str(db_path))

        bench = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "chembl_timestamps.parquet")
        bench_smiles = [str(s) for s in bench["smiles"].dropna().unique().tolist()]

        rows: list[dict] = []
        for smi in tqdm(bench_smiles, desc="FPSim2 similarity", unit="cmpd"):
            ok, canon = validate_smiles(smi)
            if not ok or canon is None:
                append_invalid_smiles(PROJECT_ROOT, source=phase, key="bench", smiles=smi, reason="rdkit_invalid")
                continue

            results = engine.similarity(canon, 0.0, n_workers=1)
            if len(results) > 0:
                idx = int(results["coeff"].argmax())
                nearest_neighbor_id = int(results[idx][0])
                max_tanimoto = float(results[idx][1])
            else:
                nearest_neighbor_id = None
                max_tanimoto = 0.0

            rows.append(
                {
                    "smiles": canon,
                    "max_tanimoto": max_tanimoto,
                    "nearest_neighbor_id": nearest_neighbor_id,
                    "contamination_class": _classify_tanimoto(max_tanimoto),
                    "cutoff_used": cutoff_used,
                }
            )

        out_path = PROJECT_ROOT / "data" / "processed" / "compound_contamination.parquet"
        pd.DataFrame(rows).to_parquet(out_path, index=False)
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

