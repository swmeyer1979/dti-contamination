"""
Aggregate random-transformer predictions across multiple seeds.

Loads per-seed prediction parquets for seeds 7, 42, 99, averages y_pred
for each (smiles, protein_key, dataset) triplet, then reports per-seed and
multi-seed macro-avg Pearson r on holdout (n_min=10 pairs per target).

Outputs:
  - results/random_transformer_multiseed_predictions.parquet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
SEEDS = [7, 42, 99]


def _per_target_macro_r(df: pd.DataFrame, min_n: int = 10) -> float:
    """Macro-averaged per-target Pearson r on the holdout set."""
    h = df[df["dataset"] == "holdout"].copy()
    rs = []
    for prot, g in h.groupby("protein_key"):
        if len(g) < min_n:
            continue
        if g["affinity"].std() < 1e-8 or g["y_pred"].std() < 1e-8:
            continue
        r, _ = pearsonr(g["affinity"], g["y_pred"])
        rs.append(r)
    if not rs:
        return float("nan")
    return float(np.mean(rs))


def main() -> None:
    frames: dict[int, pd.DataFrame] = {}
    for seed in SEEDS:
        if seed == 7:
            path = RESULTS / "random_transformer_predictions.parquet"
        else:
            path = RESULTS / f"random_transformer_seed{seed}_predictions.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        df = pd.read_parquet(path)
        df["seed"] = seed
        frames[seed] = df
        macro = _per_target_macro_r(df)
        print(f"  seed={seed}: holdout macro-avg r (n_min=10) = {macro:+.4f}  (n_holdout={len(df[df.dataset=='holdout'])})")

    # Align on identity columns — use seed=7 as template for metadata
    template = frames[7].drop(columns=["y_pred", "seed"]).copy()

    # Stack y_pred columns
    y_stack = np.stack([frames[s]["y_pred"].to_numpy(dtype=np.float64) for s in SEEDS], axis=1)
    avg_pred = y_stack.mean(axis=1)

    multi = template.copy()
    multi["y_pred"] = avg_pred
    multi["seeds_averaged"] = str(SEEDS)

    out_path = RESULTS / "random_transformer_multiseed_predictions.parquet"
    multi.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path} ({len(multi)} rows)")

    macro_multi = _per_target_macro_r(multi)
    print(f"  multi-seed holdout macro-avg r (n_min=10) = {macro_multi:+.4f}")


if __name__ == "__main__":
    main()
