#!/usr/bin/env python
"""Generate the optimization-docs figure.

Output (light-theme WebP matching the README figures):

- ``docs/_static/figures/ess_x_scale.webp`` — the Exponential Spectral
  Scaling trust-region weights ``x_scale = exp(-alpha*max(|m|,|n|)) /
  exp(-alpha)`` per harmonic level, for the example alpha = 0.7 and the
  default alpha = 1.2 (pure formula, no solves).

Run from the repo root::

    python docs/_static/figures/sources/make_optimization_docs_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1]

BLUE = "#2a78d6"       # categorical slot 1 (after / vmex)
AQUA = "#1baf7a"        # categorical slot 2
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3e0"

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "svg.fonttype": "none",
})


def ess_figure() -> None:
    levels = np.arange(1, 7)
    fig, ax = plt.subplots(figsize=(5.4, 3.0), dpi=150)
    for alpha, color, label_y in ((0.7, BLUE, None), (1.2, AQUA, None)):
        scale = np.exp(-alpha * levels) / np.exp(-alpha)
        ax.plot(levels, scale, "-o", color=color, lw=2, ms=5,
                markeredgecolor="white", markeredgewidth=1, zorder=3)
        ax.annotate(rf"$\alpha = {alpha}$", (levels[-1], scale[-1]),
                    textcoords="offset points", xytext=(8, 0),
                    va="center", color=color, fontsize=9)
    ax.set_yscale("log")
    ax.set_xlim(0.7, 7.0)
    ax.set_xticks(levels)
    ax.set_xlabel(r"harmonic level  $\max(|m|,\, |n|)$")
    ax.set_ylabel("trust-region scale (log)")
    ax.set_title("Exponential Spectral Scaling: per-dof trust radius",
                 fontsize=10, loc="left")
    ax.grid(True, axis="y", color=GRID, lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate("level-1 dofs keep the full step", (1.55, 0.93),
                color=INK2, fontsize=8, va="top")
    ax.annotate("max_mode-6 dofs:\n~1/400 of the step\n"
                r"($\alpha = 1.2$)",
                (6, np.exp(-1.2 * 6) / np.exp(-1.2)),
                textcoords="offset points", xytext=(-86, 6),
                color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "ess_x_scale.webp", pil_kwargs={"lossless": True})
    plt.close(fig)


if __name__ == "__main__":
    ess_figure()
    print("ess_x_scale.webp", (OUT / "ess_x_scale.webp").stat().st_size, "bytes")
