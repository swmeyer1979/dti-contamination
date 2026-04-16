from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


_BATCH_RE = re.compile(r"^(?P<name>.+)_batch_(?P<idx>\d+)\.parquet$")


def find_existing_batches(checkpoints_dir: Path, name: str) -> list[Path]:
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    matches: list[tuple[int, Path]] = []
    for p in checkpoints_dir.glob(f"{name}_batch_*.parquet"):
        m = _BATCH_RE.match(p.name)
        if not m:
            continue
        if m.group("name") != name:
            continue
        matches.append((int(m.group("idx")), p))
    return [p for _, p in sorted(matches, key=lambda x: x[0])]


def next_batch_index(checkpoints_dir: Path, name: str) -> int:
    batches = find_existing_batches(checkpoints_dir, name)
    if not batches:
        return 1
    m = _BATCH_RE.match(batches[-1].name)
    assert m is not None
    return int(m.group("idx")) + 1


def load_processed_keys(checkpoints_dir: Path, name: str, key_cols: list[str]) -> set[tuple]:
    processed: set[tuple] = set()
    for p in find_existing_batches(checkpoints_dir, name):
        try:
            df = pd.read_parquet(p)
        except Exception:  # noqa: BLE001
            continue
        if df.empty:
            continue
        missing = [c for c in key_cols if c not in df.columns]
        if missing:
            continue
        for row in df[key_cols].itertuples(index=False, name=None):
            processed.add(tuple(row))
    return processed


def write_batch(
    checkpoints_dir: Path,
    name: str,
    batch_idx: int,
    records: list[dict],
    *,
    sort_cols: Optional[list[str]] = None,
) -> Path:
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    out = checkpoints_dir / f"{name}_batch_{batch_idx}.parquet"
    df = pd.DataFrame.from_records(records)
    if sort_cols:
        existing = [c for c in sort_cols if c in df.columns]
        if existing:
            df = df.sort_values(existing, kind="mergesort")
    df.to_parquet(out, index=False)
    return out


def concat_batches_to_parquet(
    checkpoints_dir: Path,
    name: str,
    output_path: Path,
    *,
    keep: Optional[Iterable[str]] = None,
) -> None:
    dfs: list[pd.DataFrame] = []
    for p in find_existing_batches(checkpoints_dir, name):
        df = pd.read_parquet(p)
        if keep is not None:
            cols = [c for c in keep if c in df.columns]
            df = df[cols]
        dfs.append(df)
    out_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)

