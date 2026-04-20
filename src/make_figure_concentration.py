"""
Generate the Q92918-concentration figure for the paper: histogram of per-target
pair counts in the 1,028-pair holdout, with the top target flagged.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    hd = pd.read_parquet(PROJECT_ROOT / "data" / "splits" / "holdout_temporal_splits.parquet")
    counts = hd.groupby("protein_key").size().sort_values(ascending=False)

    # Panel A: rank-ordered bar chart of top 25 targets
    top_n = 25
    top_counts = counts.head(top_n).values
    top_labels = counts.head(top_n).index.tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Color Q92918 red, others blue
    colors = ["#c62828" if label == "Q92918" else "#1f77b4" for label in top_labels]
    bars = ax1.bar(range(len(top_counts)), top_counts, color=colors)
    ax1.set_xlabel("Target rank")
    ax1.set_ylabel("Number of pairs in holdout")
    ax1.set_title(f"Top {top_n} targets by pair count (of 85 total)")
    ax1.set_xticks([0, 4, 9, 14, 19, 24])
    ax1.set_xticklabels(["1", "5", "10", "15", "20", "25"])
    # Annotate Q92918 specifically
    q_rank = top_labels.index("Q92918")
    q_count = top_counts[q_rank]
    ax1.annotate(
        f"Q92918 (MAP3K1)\n{q_count} pairs ({100*q_count/len(hd):.1f}% of holdout)",
        xy=(q_rank, q_count),
        xytext=(5, q_count * 0.85),
        arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.2),
        fontsize=9,
        color="#c62828",
        ha="left",
    )
    ax1.grid(True, axis="y", alpha=0.3)

    # Panel B: cumulative share of pairs as more targets are included
    cum_share = np.cumsum(counts.values) / counts.sum()
    ax2.plot(range(1, len(cum_share) + 1), cum_share * 100, linewidth=1.8, color="#1f77b4")
    ax2.axhline(50, color="#666", linestyle=":", linewidth=0.8)
    ax2.axhline(80, color="#666", linestyle=":", linewidth=0.8)
    # Find rank at which cumulative share crosses 50% and 80%
    r50 = int(np.searchsorted(cum_share, 0.50) + 1)
    r80 = int(np.searchsorted(cum_share, 0.80) + 1)
    ax2.axvline(1, color="#c62828", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.annotate(
        f"1 target = {100*cum_share[0]:.1f}%",
        xy=(1, cum_share[0] * 100),
        xytext=(10, cum_share[0] * 100 - 5),
        fontsize=9,
        color="#c62828",
    )
    ax2.annotate(
        f"{r50} targets → 50%",
        xy=(r50, 50),
        xytext=(r50 + 3, 45),
        fontsize=9,
        color="#666",
    )
    ax2.annotate(
        f"{r80} targets → 80%",
        xy=(r80, 80),
        xytext=(r80 + 3, 75),
        fontsize=9,
        color="#666",
    )
    ax2.set_xlabel("Number of targets included (rank-ordered by pair count)")
    ax2.set_ylabel("Cumulative % of holdout pairs")
    ax2.set_title("Cumulative pair share")
    ax2.set_xlim(0, 85)
    ax2.set_ylim(0, 102)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Per-target pair count concentration in the 1,028-pair holdout (85 targets)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()

    out_path = PROJECT_ROOT / "paper" / "figures" / "target_concentration.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
