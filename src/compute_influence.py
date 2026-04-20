"""
Influence decomposition: per-target contribution to pooled Pearson r.

For each model on holdout:
  influence_t = pooled_r - pooled_r_without_t

where pooled_r is computed over all holdout pairs and pooled_r_without_t
excludes all pairs belonging to target t.

Outputs:
  - results/influence_decomposition.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
MODELS = ["esm2_probe", "deepdta", "random_probe", "random_transformer"]
TARGET_OF_INTEREST = "Q92918"


def _pooled_r(df: pd.DataFrame) -> float:
    if len(df) < 3:
        return float("nan")
    a = df["affinity"].to_numpy(dtype=float)
    p = df["y_pred"].to_numpy(dtype=float)
    if np.std(a) < 1e-8 or np.std(p) < 1e-8:
        return float("nan")
    r, _ = pearsonr(a, p)
    return float(r)


def _influence_decomposition(holdout: pd.DataFrame) -> dict:
    targets = holdout["protein_key"].unique().tolist()
    base_r = _pooled_r(holdout)

    influences = []
    for t in targets:
        without = holdout[holdout["protein_key"] != t]
        r_without = _pooled_r(without)
        inf = base_r - r_without
        influences.append({
            "protein": t,
            "n": int((holdout["protein_key"] == t).sum()),
            "influence": inf,
            "pooled_r_without": r_without,
        })

    influences.sort(key=lambda x: abs(x["influence"]), reverse=True)
    return {"pooled_r": base_r, "n_total": len(holdout), "by_target": influences}


def main() -> None:
    results: dict = {}

    for model in MODELS:
        if model == "random_transformer":
            path = RESULTS / "random_transformer_predictions.parquet"
        else:
            path = RESULTS / f"{model}_predictions.parquet"

        if not path.exists():
            print(f"  SKIP {model}: {path} not found")
            continue

        df = pd.read_parquet(path)
        holdout = df[df["dataset"] == "holdout"].copy()
        if len(holdout) == 0:
            print(f"  SKIP {model}: no holdout rows")
            continue

        decomp = _influence_decomposition(holdout)
        results[model] = decomp

        print(f"\n=== {model} (pooled_r={decomp['pooled_r']:+.4f}, n={decomp['n_total']}) ===")
        print("  Top-5 most influential targets (by |influence|):")
        for row in decomp["by_target"][:5]:
            print(f"    {row['protein']:12s}  n={row['n']:4d}  "
                  f"influence={row['influence']:+.6f}  "
                  f"r_without={row['pooled_r_without']:+.6f}")

        # Q92918 specifically
        q_rows = [r for r in decomp["by_target"] if r["protein"] == TARGET_OF_INTEREST]
        if q_rows:
            q = q_rows[0]
            pct = (q["influence"] / decomp["pooled_r"] * 100) if abs(decomp["pooled_r"]) > 1e-8 else float("nan")
            print(f"\n  {TARGET_OF_INTEREST}: influence={q['influence']:+.6f}  "
                  f"({pct:+.2f}% of pooled r)")
        else:
            print(f"\n  {TARGET_OF_INTEREST}: not in holdout targets")

    # Key number: Q92918 explains X% of pooled r variance for DeepDTA
    print("\n=== KEY NUMBER: Q92918 on DeepDTA ===")
    if "deepdta" in results:
        d = results["deepdta"]
        q_rows = [r for r in d["by_target"] if r["protein"] == TARGET_OF_INTEREST]
        if q_rows:
            q = q_rows[0]
            pct = (q["influence"] / d["pooled_r"] * 100) if abs(d["pooled_r"]) > 1e-8 else float("nan")
            print(f"  DeepDTA pooled r           = {d['pooled_r']:+.6f}")
            print(f"  DeepDTA r without Q92918   = {q['pooled_r_without']:+.6f}")
            print(f"  Influence (drop in r)      = {q['influence']:+.6f}")
            print(f"  Q92918 explains {pct:+.2f}% of pooled r")
        else:
            print(f"  {TARGET_OF_INTEREST} not in DeepDTA holdout targets")
    else:
        print("  DeepDTA results not available")

    out_path = RESULTS / "influence_decomposition.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
