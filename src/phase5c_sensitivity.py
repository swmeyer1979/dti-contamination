"""
Sensitivity analysis for per-target r metrics on the 1028-pair unified holdout.

Three checks (statistician's non-negotiables):
  1. Cutoff sensitivity: compute macro-avg and median per-target r at n_min in
     {5, 10, 20, 30}. Report how model rankings change.
  2. Leave-one-target-out: for each target, drop it, recompute macro-avg; report
     range and per-target contribution. Identifies single-target dominance at
     the per-target level.
  3. Target-level bootstrap: resample targets with replacement (not pairs),
     compute macro-avg per bootstrap; report 95% CI and p-value for
     "random-init transformer beats ESM-2 on macro-avg."

Outputs:
  - results/sensitivity.json
  - results/stats_figures/sensitivity_ranking.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _per_target_r(df: pd.DataFrame, min_n: int) -> pd.DataFrame:
    rows = []
    for prot, g in df.groupby("protein_key"):
        if len(g) < min_n:
            continue
        if g.affinity.std() < 1e-8 or g.y_pred.std() < 1e-8:
            continue
        r, _ = pearsonr(g.affinity, g.y_pred)
        rows.append({
            "protein": prot,
            "n": len(g),
            "r": r,
            "source": g["source"].iloc[0] if "source" in g.columns else "unknown",
        })
    return pd.DataFrame(rows)


def _macro(per_target_r: pd.DataFrame) -> float:
    if len(per_target_r) == 0:
        return float("nan")
    return float(per_target_r["r"].mean())


def _median(per_target_r: pd.DataFrame) -> float:
    if len(per_target_r) == 0:
        return float("nan")
    return float(per_target_r["r"].median())


def run_cutoff_sensitivity(models: dict[str, pd.DataFrame], cutoffs: list[int]) -> dict:
    out = {}
    for cutoff in cutoffs:
        row = {}
        for model, df in models.items():
            ptr = _per_target_r(df, cutoff)
            row[model] = {
                "n_targets": int(len(ptr)),
                "macro_r": _macro(ptr),
                "median_r": _median(ptr),
                "n_pairs_covered": int(ptr["n"].sum()) if len(ptr) > 0 else 0,
            }
        out[f"n_min_{cutoff}"] = row
    return out


def run_leave_one_target_out(models: dict[str, pd.DataFrame], min_n: int = 10) -> dict:
    out = {}
    for model, df in models.items():
        ptr = _per_target_r(df, min_n)
        if len(ptr) < 3:
            out[model] = {"skipped": True, "reason": "too few targets"}
            continue
        base = ptr["r"].mean()
        loos = []
        for _, dropped in ptr.iterrows():
            remaining = ptr[ptr.protein != dropped["protein"]]
            loos.append({
                "dropped_protein": dropped["protein"],
                "dropped_n": int(dropped["n"]),
                "dropped_r": float(dropped["r"]),
                "macro_without": float(remaining["r"].mean()),
                "delta": float(remaining["r"].mean() - base),
            })
        loos_sorted = sorted(loos, key=lambda x: abs(x["delta"]), reverse=True)
        out[model] = {
            "base_macro": float(base),
            "n_targets": int(len(ptr)),
            "most_influential": loos_sorted[:5],
            "range_of_macro_without": [
                min(l["macro_without"] for l in loos),
                max(l["macro_without"] for l in loos),
            ],
        }
    return out


def _wild_cluster_boot_macro_r(
    models: dict[str, pd.DataFrame],
    min_n: int = 10,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict:
    """Rademacher wild bootstrap for macro-averaged per-target r.

    For each bootstrap iteration:
      1. Draw Rademacher ε_t ∈ {-1, +1} independently for each target.
      2. Compute perturbed r*_t = r_t + ε_t * (r_t - r̄)  where r̄ = macro-avg.
      3. Macro-avg under wild perturbation = mean(r*_t).

    This preserves the variance structure of target-level r values while
    recentering under the null, giving CI estimates robust to heteroscedasticity
    across targets. Compare to standard target-level bootstrap (run_target_bootstrap).
    """
    rng = np.random.default_rng(seed)
    out: dict = {}

    for model, df in models.items():
        h = df[df["dataset"] == "holdout"].copy()
        ptr_rows = []
        for prot, g in h.groupby("protein_key"):
            if len(g) < min_n:
                continue
            if g["affinity"].std() < 1e-8 or g["y_pred"].std() < 1e-8:
                continue
            r, _ = pearsonr(g["affinity"], g["y_pred"])
            ptr_rows.append(r)

        r_vals = np.array(ptr_rows, dtype=float)
        n_t = len(r_vals)
        if n_t < 3:
            out[model] = {"skipped": True, "reason": "too few targets", "n_targets": n_t}
            continue

        r_bar = float(r_vals.mean())

        # Wild Rademacher perturbation: ε ~ {-1, +1} per target
        # r*_t = r_t + ε_t * (r_t - r̄)  =>  macro* = r̄ + ε̄ * (Var of r vals)
        wild_macros = np.empty(n_boot)
        for i in range(n_boot):
            eps = rng.choice([-1.0, 1.0], size=n_t)
            r_star = r_vals + eps * (r_vals - r_bar)
            wild_macros[i] = r_star.mean()

        ci_lo = float(np.percentile(wild_macros, 2.5))
        ci_hi = float(np.percentile(wild_macros, 97.5))

        out[model] = {
            "n_targets": n_t,
            "macro": r_bar,
            "ci95_wild_cluster": [ci_lo, ci_hi],
            "method": "Rademacher wild bootstrap on target-level r (ε_t ∈ {-1,+1} per target)",
        }

    return out


def run_target_bootstrap(
    models: dict[str, pd.DataFrame],
    min_n: int = 10,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict:
    """Resample targets (with replacement) and recompute macro-avg per bootstrap."""
    rng = np.random.default_rng(seed)
    out = {}

    # Build per-target-r tables once
    ptr_by_model = {m: _per_target_r(df, min_n) for m, df in models.items()}

    # Align target sets: target-level bootstrap uses the UNION of targets
    # (each model can have different subset depending on min_n criterion)
    # For pairwise comparison (model A vs model B), we need common targets.
    # Start with: per-model bootstrap on its own target set
    for model, ptr in ptr_by_model.items():
        if len(ptr) < 3:
            out[model] = {"skipped": True, "reason": "too few targets"}
            continue
        n_t = len(ptr)
        r_vals = ptr["r"].to_numpy()
        boot_macros = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n_t, size=n_t)
            boot_macros[i] = r_vals[idx].mean()
        out[model] = {
            "n_targets": n_t,
            "macro": float(r_vals.mean()),
            "ci95": [float(np.percentile(boot_macros, 2.5)),
                     float(np.percentile(boot_macros, 97.5))],
        }

    # Pairwise delta bootstrap: model M_treat - model M_ctrl on common targets
    # This is the statistician-requested test for "random transformer beats ESM-2"
    pairs_to_test = [
        ("random_transformer", "esm2_probe"),
        ("random_transformer", "deepdta"),
        ("esm2_probe", "deepdta"),
        ("random_probe", "random_transformer"),
    ]
    delta_results = {}
    for treat, ctrl in pairs_to_test:
        ptr_t = ptr_by_model[treat].set_index("protein")
        ptr_c = ptr_by_model[ctrl].set_index("protein")
        common = ptr_t.index.intersection(ptr_c.index)
        if len(common) < 3:
            delta_results[f"{treat}_minus_{ctrl}"] = {"skipped": True, "n_common": len(common)}
            continue
        r_t = ptr_t.loc[common, "r"].to_numpy()
        r_c = ptr_c.loc[common, "r"].to_numpy()
        observed = float((r_t - r_c).mean())
        n_t = len(common)
        boot_deltas = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n_t, size=n_t)
            boot_deltas[i] = (r_t[idx] - r_c[idx]).mean()
        delta_results[f"{treat}_minus_{ctrl}"] = {
            "n_common_targets": n_t,
            "observed_delta": observed,
            "ci95": [float(np.percentile(boot_deltas, 2.5)),
                     float(np.percentile(boot_deltas, 97.5))],
            "p_treat_beats_ctrl": float((boot_deltas > 0).mean()),
        }
    out["_pairwise_deltas"] = delta_results
    return out


def make_cutoff_figure(cutoff_results: dict, out_path: Path):
    cutoffs = sorted([int(k.split("_")[-1]) for k in cutoff_results.keys()])
    models = list(next(iter(cutoff_results.values())).keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for model in models:
        macros = [cutoff_results[f"n_min_{c}"][model]["macro_r"] for c in cutoffs]
        n_targets = [cutoff_results[f"n_min_{c}"][model]["n_targets"] for c in cutoffs]
        ax1.plot(cutoffs, macros, marker="o", label=model)
        for c, m, n in zip(cutoffs, macros, n_targets):
            ax1.annotate(f"n={n}", (c, m), textcoords="offset points",
                         xytext=(5, 5), fontsize=7)

    ax1.set_xlabel("Minimum pairs per target (cutoff)")
    ax1.set_ylabel("Macro-averaged per-target Pearson r")
    ax1.set_title("Macro-avg r vs cutoff")
    ax1.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    for model in models:
        medians = [cutoff_results[f"n_min_{c}"][model]["median_r"] for c in cutoffs]
        ax2.plot(cutoffs, medians, marker="s", label=model)
    ax2.set_xlabel("Minimum pairs per target (cutoff)")
    ax2.set_ylabel("Median per-target Pearson r")
    ax2.set_title("Median per-target r vs cutoff")
    ax2.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Per-target metric sensitivity to minimum-n cutoff", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    models = {}
    for m in ["esm2_probe", "deepdta", "random_probe", "random_transformer"]:
        p = pd.read_parquet(PROJECT_ROOT / "results" / f"{m}_predictions.parquet")
        h = p[p.dataset == "holdout"].copy()
        models[m] = h

    print(f"Loaded {len(models)} models. Holdout rows per model:", {m: len(d) for m, d in models.items()})

    print("\n=== 1. Cutoff sensitivity ===")
    cutoff_res = run_cutoff_sensitivity(models, cutoffs=[5, 10, 20, 30])
    for cut_key, row in cutoff_res.items():
        print(f"\n{cut_key}:")
        for model, stats in row.items():
            print(f"  {model:25s}  n_targets={stats['n_targets']:3d}  "
                  f"macro={stats['macro_r']:+.4f}  median={stats['median_r']:+.4f}  "
                  f"covered_pairs={stats['n_pairs_covered']}")

    print("\n=== 2. Leave-one-target-out (min_n=10) ===")
    loo_res = run_leave_one_target_out(models, min_n=10)
    for model, stats in loo_res.items():
        if stats.get("skipped"):
            print(f"  {model}: SKIPPED ({stats['reason']})")
            continue
        print(f"  {model}: base macro={stats['base_macro']:+.4f}, "
              f"range if one target dropped [{stats['range_of_macro_without'][0]:+.4f}, "
              f"{stats['range_of_macro_without'][1]:+.4f}]")
        print(f"    most influential drops:")
        for inf in stats["most_influential"][:3]:
            print(f"      drop {inf['dropped_protein']} (n={inf['dropped_n']}, r={inf['dropped_r']:+.3f}): "
                  f"macro → {inf['macro_without']:+.4f} (Δ={inf['delta']:+.4f})")

    print("\n=== 3. Target-level bootstrap (resample targets, 10k iters, min_n=10) ===")
    tb_res = run_target_bootstrap(models, min_n=10, n_boot=10_000)
    for model, stats in tb_res.items():
        if model.startswith("_"):
            continue
        if stats.get("skipped"):
            print(f"  {model}: SKIPPED")
            continue
        print(f"  {model}: macro={stats['macro']:+.4f}, "
              f"95% CI=[{stats['ci95'][0]:+.4f}, {stats['ci95'][1]:+.4f}] (n={stats['n_targets']} targets)")

    print("\n=== Pairwise delta (resample common-target set) ===")
    for comp_key, stats in tb_res["_pairwise_deltas"].items():
        if stats.get("skipped"):
            print(f"  {comp_key}: SKIPPED (n_common={stats.get('n_common', 0)})")
            continue
        print(f"  {comp_key}: Δ={stats['observed_delta']:+.4f}, "
              f"95% CI=[{stats['ci95'][0]:+.4f}, {stats['ci95'][1]:+.4f}], "
              f"P(treat>ctrl)={stats['p_treat_beats_ctrl']:.3f}  (n_common={stats['n_common_targets']})")

    print("\n=== 4. Wild cluster bootstrap (Rademacher, per-target ε, 10k iters, min_n=10) ===")
    wild_res = _wild_cluster_boot_macro_r(models, min_n=10, n_boot=10_000)
    for model, stats in wild_res.items():
        if stats.get("skipped"):
            print(f"  {model}: SKIPPED ({stats['reason']})")
            continue
        std_ci = tb_res.get(model, {}).get("ci95")
        std_str = (f" | standard target bootstrap CI=[{std_ci[0]:+.4f}, {std_ci[1]:+.4f}]"
                   if std_ci else "")
        print(f"  {model}: macro={stats['macro']:+.4f}, "
              f"wild CI=[{stats['ci95_wild_cluster'][0]:+.4f}, {stats['ci95_wild_cluster'][1]:+.4f}]"
              f"{std_str}  (n={stats['n_targets']} targets)")

    # Save
    out = {
        "cutoff_sensitivity": cutoff_res,
        "leave_one_out": loo_res,
        "target_bootstrap": tb_res,
        "wild_cluster_bootstrap": wild_res,
    }
    out_path = PROJECT_ROOT / "results" / "sensitivity.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")

    make_cutoff_figure(cutoff_res, PROJECT_ROOT / "results" / "stats_figures" / "sensitivity_ranking.png")
    print(f"Wrote sensitivity_ranking.png")


if __name__ == "__main__":
    main()
