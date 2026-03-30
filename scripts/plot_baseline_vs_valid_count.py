"""Plot baseline safety score vs valid lever count per image.

Reads:
  - Baseline scores from the JSON cache
  - Per-image auxiliary CSV for n_valid counts

Produces:
  - PNG scatter plot for the paper
  - Prints Spearman correlation
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(
    scores_path: str = "data/specs_paper_n50_scores.json",
    per_image_csv: str = "data/03_eval_results/specs_paper_n50_per_image_auxiliary.csv",
    output_path: str = "paper/figures/figure_4_baseline_vs_valid_count.png",
) -> None:
    with open(scores_path) as f:
        baseline_scores: dict[str, float] = json.load(f)

    n_valid_map: dict[str, int] = {}
    with open(per_image_csv) as f:
        for row in csv.DictReader(f):
            n_valid_map[row["image_id"]] = int(row["n_valid"])

    baselines = []
    counts = []
    for image_id, score in baseline_scores.items():
        baselines.append(float(score))
        counts.append(n_valid_map.get(image_id, 0))

    baselines = np.array(baselines)
    counts = np.array(counts)

    from scipy.stats import spearmanr
    rho, pval = spearmanr(baselines, counts)
    print(f"N = {len(baselines)}")
    print(f"Spearman rho = {rho:.3f}, p = {pval:.3f}")

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(counts))
    max_count = int(max(counts)) if len(counts) else 0

    ax.scatter(
        baselines,
        counts + jitter,
        s=48,
        c=counts,
        cmap="RdYlGn",
        edgecolors="0.3",
        linewidths=0.5,
        vmin=0,
        vmax=max(3, max_count),
        zorder=3,
    )

    ax.set_xlabel("Baseline safety score ($f_a(x)$)", fontsize=10)
    ax.set_ylabel("Valid lever count $|V(x)|$", fontsize=10)
    ax.set_yticks(list(range(0, max_count + 1)))
    ax.set_xlim(0, 5.5)
    ax.set_ylim(-0.4, max(3.6, max_count + 0.6))

    ax.annotate(
        f"$\\rho$ = {rho:.2f}, p = {pval:.2f}",
        xy=(0.97, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved → {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main(*sys.argv[1:])
