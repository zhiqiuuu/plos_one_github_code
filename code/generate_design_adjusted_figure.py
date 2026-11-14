"""Render the final association-path figure without a causal mediation claim."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "results" / "output_final" / "B01" / "main_paths_brr.csv"
OUTPUT_DIR = ROOT / "figures"


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=Path, default=INPUT_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    paths = pd.read_csv(args.input_file).set_index("parameter")
    keys = ["a", "c", "b", "c_prime"]
    labels = ["a: Training -> utility", "c: Training -> AI use", "b: Utility -> AI use", "c': Training -> AI use | utility"]
    colors = ["#0F766E", "#A16207", "#475569", "#7C3AED"]
    estimates = paths.loc[keys, "estimate"].to_numpy()
    lower = paths.loc[keys, "ci_lower_95"].to_numpy()
    upper = paths.loc[keys, "ci_upper_95"].to_numpy()
    positions = np.arange(len(keys))[::-1]

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(4.3, 2.6))
    ax.axvline(0, color="#94A3B8", linewidth=0.8, zorder=0)
    for position, estimate, lo, hi, color in zip(positions, estimates, lower, upper, colors):
        ax.errorbar(
            estimate,
            position,
            xerr=[[estimate - lo], [hi - estimate]],
            fmt="none",
            color=color,
            linewidth=1.4,
            capsize=2.5,
            zorder=2,
        )
        ax.scatter(estimate, position, s=34, color=color, edgecolor="white", linewidth=0.6, zorder=3)

    ax.set_yticks(positions, labels)
    ax.set_xlabel("Coefficient (a: linear scale; b, c, c': log odds)")
    ax.set_xlim(min(-0.15, lower.min() - 0.08), upper.max() + 0.12)
    ax.set_ylim(-0.65, 3.65)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.6, zorder=0)
    fig.text(
        0.39,
        0.04,
        "Association paths; 95% Fay-BRR CI; no mediation estimate",
        ha="left",
        va="top",
        fontsize=5.8,
        color="#334155",
    )
    fig.subplots_adjust(left=0.39, right=0.98, top=0.96, bottom=0.22)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "Fig1_design_adjusted_association_paths"
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    main()
