"""
Computes sequence identity between benchmark proteins and pre-cutoff UniProt sequences.

Requires:
  - phase1_uniprot

Outputs:
  - data/processed/protein_contamination.parquet
    [uniprot_id, max_pident, contamination_class]

Sentinels:
  - writes checkpoints/phase2_mmseqs2.done
"""

from __future__ import annotations

import shutil
import subprocess
import traceback
from pathlib import Path

import pandas as pd

from utils.logging_utils import setup_logging
from utils.sentinel import require_sentinel, write_sentinel
from utils.status import StatusUpdater


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _write_fasta(path: Path, ids: list[str], seqs: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for pid, seq in zip(ids, seqs):
            f.write(f">{pid}\n")
            for j in range(0, len(seq), 60):
                f.write(seq[j : j + 60] + "\n")


def _classify(pident: float) -> str:
    if pident >= 90.0:
        return "contaminated"
    if pident >= 40.0:
        return "partial"
    return "clean"


def main() -> int:
    phase = "phase2_mmseqs2"
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

        if shutil.which("mmseqs") is None:
            msg = "mmseqs2 not found on PATH. Install with: brew install mmseqs2"
            logger.error(msg)
            status.update(phase, "error", error=msg)
            return 1

        df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "uniprot_timestamps.parquet")
        df = df.copy()
        df["uniprot_id"] = df["uniprot_id"].fillna("").astype(str)
        df["sequence"] = df["sequence"].fillna("").astype(str)
        df = df[df["sequence"].str.len() > 0]

        # Assign stable IDs for missing accessions
        ids = []
        for i, acc in enumerate(df["uniprot_id"].tolist()):
            ids.append(acc if acc else f"UNK_{i}")

        cutoff = pd.Timestamp("2021-04-01", tz="UTC")
        dates = pd.to_datetime(df["first_public_date"], errors="coerce", utc=True)
        is_target = (dates.notna()) & (dates < cutoff)

        query_fasta = PROJECT_ROOT / "data" / "raw" / "query_proteins.fasta"
        target_fasta = PROJECT_ROOT / "data" / "raw" / "target_proteins.fasta"
        _write_fasta(query_fasta, ids, df["sequence"].tolist())
        _write_fasta(target_fasta, [ids[i] for i in range(len(ids)) if bool(is_target.iloc[i])], df["sequence"][is_target].tolist())

        results_tsv = PROJECT_ROOT / "data" / "raw" / "mmseqs_results.tsv"
        tmp_dir = PROJECT_ROOT / "data" / "raw" / "mmseqs_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "mmseqs",
            "easy-search",
            str(query_fasta),
            str(target_fasta),
            str(results_tsv),
            str(tmp_dir),
            "--min-seq-id",
            "0.4",
            "--format-output",
            "query,target,pident,evalue",
            "-v",
            "1",
        ]
        logger.info("Running: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)

        res = pd.read_csv(results_tsv, sep="\t", header=None, names=["query", "target", "pident", "evalue"])
        res["pident"] = pd.to_numeric(res["pident"], errors="coerce")
        max_pident = res.groupby("query")["pident"].max().to_dict()

        out_rows = []
        for q in ids:
            p = float(max_pident.get(q, 0.0) or 0.0)
            out_rows.append({"uniprot_id": q, "max_pident": p, "contamination_class": _classify(p)})

        out_path = PROJECT_ROOT / "data" / "processed" / "protein_contamination.parquet"
        pd.DataFrame(out_rows).to_parquet(out_path, index=False)
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
