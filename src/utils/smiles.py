from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from rdkit import Chem


def validate_smiles(smiles: str) -> Tuple[bool, Optional[str]]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, None
        canon = Chem.MolToSmiles(mol, canonical=True)
        return True, canon
    except Exception:  # noqa: BLE001
        return False, None


def append_invalid_smiles(project_root: Path, source: str, key: str, smiles: str, reason: str) -> None:
    out_path = project_root / "data" / "processed" / "invalid_smiles.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{"source": source, "key": key, "smiles": smiles, "reason": reason}])
    header = not out_path.exists()
    df.to_csv(out_path, mode="a", header=header, index=False)

