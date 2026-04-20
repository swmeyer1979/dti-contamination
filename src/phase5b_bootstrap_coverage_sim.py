"""
Bootstrap coverage simulation for clustered DTI-style test panels.

Generates synthetic panels where true Pearson r(y_true, y_pred) is known, applies
each of four bootstrap procedures (pair / protein-cluster / compound-cluster /
two-way), and measures empirical 95% CI coverage across many replications.

Purpose: defensibility appendix for the benchmark paper — shows that pair-level
bootstrap under-covers (CIs too narrow) while the two-way cluster bootstrap
recovers nominal coverage under realistic DTI cluster structure.

Outputs:
  - results/bootstrap_coverage.json   (per-method coverage + CI width)
  - results/stats_figures/bootstrap_coverage.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _weighted_pearson_r(x, y, w):
    mask = w > 0
    if mask.sum() < 3:
        return float("nan")
    w = w.astype(float)
    sw = float(w.sum())
    if sw <= 0:
        return float("nan")
    mx = float((w * x).sum() / sw)
    my = float((w * y).sum() / sw)
    dx = x - mx
    dy = y - my
    cov = float((w * dx * dy).sum() / sw)
    vx = float((w * dx * dx).sum() / sw)
    vy = float((w * dy * dy).sum() / sw)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return cov / np.sqrt(vx * vy)


def generate_panel(n_proteins, n_compounds_per_protein, sigma_p, sigma_c, sigma_noise, signal, rng):
    """Generate a clustered DTI-style panel.

    true_affinity[i,j] = protein_effect[i] + compound_effect[j] + noise
    predictions[i,j] = signal * true_affinity[i,j] + independent_noise

    Protein effects and compound effects are drawn per-cluster, creating true
    within-cluster correlation structure. `signal` controls the model's quality
    (higher signal → higher Pearson r).
    """
    proteins = np.arange(n_proteins)
    prot_effect = rng.normal(0, sigma_p, size=n_proteins)

    # compound_effect drawn per unique compound; compounds shared across proteins
    # to create two-way structure
    n_unique_compounds = max(10, n_compounds_per_protein * 3)
    comp_effect = rng.normal(0, sigma_c, size=n_unique_compounds)

    rows = []
    for i in range(n_proteins):
        comp_ids = rng.choice(n_unique_compounds, size=n_compounds_per_protein, replace=False)
        for c in comp_ids:
            y_true = prot_effect[i] + comp_effect[c] + rng.normal(0, sigma_noise)
            y_pred = signal * y_true + rng.normal(0, sigma_noise)
            rows.append((i, c, y_true, y_pred))

    prots = np.array([r[0] for r in rows])
    comps = np.array([r[1] for r in rows])
    yt = np.array([r[2] for r in rows])
    yp = np.array([r[3] for r in rows])
    return prots, comps, yt, yp


def pair_bootstrap_ci(yt, yp, n_boot, rng):
    n = len(yt)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(idx) >= 3:
            r, _ = pearsonr(yt[idx], yp[idx])
            boot[b] = r
        else:
            boot[b] = np.nan
    return float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))


def one_way_cluster_bootstrap_ci(yt, yp, clusters, n_boot, rng):
    unique_c, inv = np.unique(clusters, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    inv_sorted = inv[order]
    splits = np.searchsorted(inv_sorted, np.arange(len(unique_c) + 1))
    groups = [order[splits[k]:splits[k + 1]] for k in range(len(unique_c))]
    k = len(unique_c)

    boot = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.integers(0, k, size=k)
        idx = np.concatenate([groups[j] for j in sampled])
        if len(idx) >= 3:
            r, _ = pearsonr(yt[idx], yp[idx])
            boot[b] = r
        else:
            boot[b] = np.nan
    return float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))


def two_way_bootstrap_ci(yt, yp, prots, comps, n_boot, rng):
    up, inv_p = np.unique(prots, return_inverse=True)
    uc, inv_c = np.unique(comps, return_inverse=True)
    k_p, k_c = len(up), len(uc)

    boot = np.empty(n_boot)
    for b in range(n_boot):
        pc = rng.multinomial(k_p, np.full(k_p, 1.0 / k_p))
        cc = rng.multinomial(k_c, np.full(k_c, 1.0 / k_c))
        w = (pc[inv_p] * cc[inv_c]).astype(np.float64)
        boot[b] = _weighted_pearson_r(yt, yp, w)
    return float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))


def simulate_coverage(
    n_replications=300,
    n_boot=500,
    n_proteins=80,
    n_compounds_per_protein=100,
    sigma_p=1.0,
    sigma_c=0.6,
    sigma_noise=0.4,
    signal=0.8,
    seed=42,
):
    rng = np.random.default_rng(seed)

    # Estimate "true" Pearson r for this DGP via a large reference sample
    ref_rng = np.random.default_rng(seed + 10_000)
    ref_prots, ref_comps, ref_yt, ref_yp = generate_panel(
        200, 200, sigma_p, sigma_c, sigma_noise, signal, ref_rng
    )
    r_true, _ = pearsonr(ref_yt, ref_yp)
    print(f"Reference (large-sample) Pearson r for DGP: {r_true:.4f}")

    results = {method: {"covers": 0, "widths": []} for method in
               ["pair", "cluster_protein", "cluster_compound", "two_way"]}

    for rep in range(n_replications):
        rep_rng = np.random.default_rng(seed + rep)
        prots, comps, yt, yp = generate_panel(
            n_proteins, n_compounds_per_protein, sigma_p, sigma_c, sigma_noise, signal, rep_rng
        )

        lo, hi = pair_bootstrap_ci(yt, yp, n_boot, rep_rng)
        results["pair"]["covers"] += 1 if (lo <= r_true <= hi) else 0
        results["pair"]["widths"].append(hi - lo)

        lo, hi = one_way_cluster_bootstrap_ci(yt, yp, prots, n_boot, rep_rng)
        results["cluster_protein"]["covers"] += 1 if (lo <= r_true <= hi) else 0
        results["cluster_protein"]["widths"].append(hi - lo)

        lo, hi = one_way_cluster_bootstrap_ci(yt, yp, comps, n_boot, rep_rng)
        results["cluster_compound"]["covers"] += 1 if (lo <= r_true <= hi) else 0
        results["cluster_compound"]["widths"].append(hi - lo)

        lo, hi = two_way_bootstrap_ci(yt, yp, prots, comps, n_boot, rep_rng)
        results["two_way"]["covers"] += 1 if (lo <= r_true <= hi) else 0
        results["two_way"]["widths"].append(hi - lo)

        if (rep + 1) % 50 == 0:
            print(f"  rep {rep + 1}/{n_replications}")

    summary = {"r_true": r_true, "n_replications": n_replications, "n_boot_per_rep": n_boot,
               "methods": {}}
    for method, data in results.items():
        cov = data["covers"] / n_replications
        widths = np.array(data["widths"])
        summary["methods"][method] = {
            "coverage": cov,
            "mean_ci_width": float(widths.mean()),
            "median_ci_width": float(np.median(widths)),
        }
    return summary


def make_figure(summary, out_path):
    methods = ["pair", "cluster_compound", "cluster_protein", "two_way"]
    labels = ["Pair-level", "Compound cluster", "Protein cluster", "Two-way cluster"]
    coverage = [summary["methods"][m]["coverage"] for m in methods]
    widths = [summary["methods"][m]["mean_ci_width"] for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    bars1 = ax1.bar(range(len(methods)), coverage, color=["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"])
    ax1.axhline(0.95, color="black", linestyle="--", linewidth=0.8, label="Nominal 0.95")
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylabel("Empirical coverage")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("95% CI coverage under clustered DGP")
    ax1.legend(loc="lower right", fontsize=8)
    for b, v in zip(bars1, coverage):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    bars2 = ax2.bar(range(len(methods)), widths, color=["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"])
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylabel("Mean CI width")
    ax2.set_title("Mean 95% CI width")
    for b, v in zip(bars2, widths):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    fig.suptitle(
        f"Bootstrap coverage under two-way clustered DGP "
        f"(true r = {summary['r_true']:.3f}, n_rep = {summary['n_replications']})",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Running bootstrap coverage simulation (n_rep=300, n_boot=500 per rep)...")
    summary = simulate_coverage()

    print("\nResults:")
    for method, s in summary["methods"].items():
        print(f"  {method:20s}  coverage={s['coverage']:.3f}  mean_width={s['mean_ci_width']:.3f}")

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "bootstrap_coverage.json").write_text(json.dumps(summary, indent=2))
    make_figure(summary, out_dir / "stats_figures" / "bootstrap_coverage.png")
    print(f"\nWrote results/bootstrap_coverage.json and results/stats_figures/bootstrap_coverage.png")


if __name__ == "__main__":
    main()
