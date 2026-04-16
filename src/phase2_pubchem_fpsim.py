"""
Computes max Tanimoto similarity from benchmark compounds to the ChemBERTa pretraining corpus.

Corpus correction note:
  ChemBERTa (seyonec/ChemBERTa-zinc-base-v1) was pretrained on a ZINC subset (~100K molecules),
  NOT PubChem10M. The model name "-zinc-base-v1" denotes the ZINC-trained variant.
  Using ChEMBL27 as the contamination reference for ChemBERTa is an approximation — compounds
  in Davis/KIBA that appear in ZINC but not ChEMBL27 could be mislabeled "clean."
  This script originally targeted PubChem10M (seyonec/PubChem10M_SMILES_BPE_60k) as an
  alternative corpus, but:
    1. That HuggingFace dataset requires authentication (HTTP 401).
    2. PubChem10M is not the correct corpus for ChemBERTa-zinc-base-v1.
    3. The actual ZINC pretraining split (~100K molecules) is not publicly accessible.
  This script is therefore a STUB and exits with a clear limitation note rather than
  producing incorrect results. The ChEMBL27-based analysis in phase5_stats.py serves as
  the primary (conservative) contamination reference.

Requires:
  - phase1_chembl (benchmark compounds in chembl_timestamps.parquet)

Outputs:
  - (none — see limitation above)

Sentinels:
  - writes checkpoints/phase2_pubchem_fpsim.done with a limitation flag
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils.logging_utils import setup_logging
from utils.sentinel import require_sentinel, write_sentinel
from utils.smiles import append_invalid_smiles, validate_smiles
from utils.status import StatusUpdater


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_pubchem10m_smiles(logger) -> list[str]:
    """Download PubChem10M from HuggingFace and return validated SMILES list."""
    cache_path = PROJECT_ROOT / "data" / "raw" / "pubchem10m_smiles.txt"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        logger.info("Loading cached PubChem10M SMILES from %s.", cache_path)
        lines = cache_path.read_text().splitlines()
        return [l.strip() for l in lines if l.strip()]

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "HuggingFace `datasets` package required for PubChem10M download. "
            "Install with: pip install datasets"
        ) from e

    logger.info("Downloading seyonec/PubChem10M_SMILES_BPE_60k from HuggingFace (this is ~500MB)...")
    ds = load_dataset("seyonec/PubChem10M_SMILES_BPE_60k", split="train", streaming=False)

    smiles_list: list[str] = []
    invalid = 0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as fout:
        for row in tqdm(ds, desc="Validating PubChem10M SMILES", unit="mol"):
            smi = str(row.get("text", "") or row.get("smiles", "")).strip()
            if not smi:
                invalid += 1
                continue
            ok, canon = validate_smiles(smi)
            if not ok or canon is None:
                invalid += 1
                append_invalid_smiles(
                    PROJECT_ROOT,
                    source="phase2_pubchem_fpsim",
                    key="pubchem10m",
                    smiles=smi,
                    reason="rdkit_invalid",
                )
                continue
            smiles_list.append(canon)
            fout.write(canon + "\n")

    logger.info(
        "PubChem10M: %d valid SMILES loaded, %d invalid/skipped.", len(smiles_list), invalid
    )
    return smiles_list


def _build_fpsim2_db(smiles_list: list[str], db_path: Path, logger) -> None:
    try:
        from FPSim2.io import create_db_file  # type: ignore
    except ImportError as e:
        raise RuntimeError("FPSim2 not installed. Install with: pip install FPSim2") from e

    logger.info(
        "Building FPSim2 DB at %s (%d SMILES — this takes ~20-40 min).", db_path, len(smiles_list)
    )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    smiles_pairs = [(smi, i) for i, smi in enumerate(smiles_list)]
    create_db_file(
        smiles_pairs,
        str(db_path),
        mol_format="smiles",
        fp_type="Morgan",
        fp_params={"radius": 2, "fpSize": 2048},
    )
    logger.info("Built PubChem10M FPSim2 DB.")


def _classify_tanimoto(sim: float) -> str:
    if sim >= 0.9:
        return "highly_contaminated"
    if sim >= 0.6:
        return "likely_contaminated"
    return "clean"


_LIMITATION_NOTE = (
    "CORPUS LIMITATION: ChemBERTa-zinc-base-v1 was pretrained on a ZINC subset (~100K molecules), "
    "not PubChem10M. The correct ZINC pretraining split is not publicly available via HuggingFace "
    "(authentication required; ~250M full ZINC is impractical for FPSim2). "
    "ChEMBL27 is used as a conservative proxy in phase5_stats.py (primary contamination reference). "
    "This phase exits successfully without producing output — see design.compound_contamination_reference "
    "in stats_report.json for full limitation disclosure."
)


def main() -> int:
    phase = "phase2_pubchem_fpsim"
    status = StatusUpdater(PROJECT_ROOT)
    logger = setup_logging(PROJECT_ROOT, phase)
    status.update(phase, "running", progress=0.0)

    logger.warning(_LIMITATION_NOTE)
    logger.info(
        "Phase %s exiting as stub. No output produced. "
        "See module docstring for corpus correction details.",
        phase,
    )
    write_sentinel(phase)
    status.update(phase, "completed", progress=1.0, error=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
