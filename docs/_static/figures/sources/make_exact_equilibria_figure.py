#!/usr/bin/env python3
"""Draw ``readme_exact_equilibria.webp`` from ``benchmarks/exact_equilibria.json``.

The record is taken from the exact-equilibrium benchmark
(rogeriojorge/vmex-benchmark-analytical, commit named in ``source``): the
boundaries of Landreman's integer-transform family A and the sheared family B
(arXiv:2609.26742) at ``tau = 0, +-tau`` (Issan et al., arXiv:2610.07304), and,
for VMEX seeded from the exact geometry at ``ftol = 1e-11``, the relative L2
error of J (native field and WOUT) and the flux-label error on each surface.

Usage::

    python docs/_static/figures/sources/make_exact_equilibria_figure.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
RECORD = REPO / "benchmarks" / "exact_equilibria.json"
OUT = REPO / "docs" / "_static" / "figures" / "readme_exact_equilibria.webp"
SYM, ASYM = "#6b6b68", "#2a78d6"


def main():
    data = json.loads(RECORD.read_text())
    runs = data["runs"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4), gridspec_kw={"width_ratios": (1, 1.2, 1.2)})
    for key, b in data["boundaries_phi0"].items():
        shift = 0.0 if key.startswith("integer") else 0.75
        color = {"-1": "#eb6834", "+0": SYM, "+1": ASYM}[key[-2:]]
        axes[0].plot(np.asarray(b["R"]) + shift, b["Z"], color=color, lw=1.3)
    axes[0].set(aspect="equal", ylim=(-0.55, 0.4), xlabel="R", ylabel="Z",
                title="exact boundaries, $\\tau=0$ and $\\pm\\tau$")
    axes[0].text(0.95, -0.47, "family A", ha="center", fontsize=8)
    axes[0].text(1.7, -0.47, "family B", ha="center", fontsize=8)
    for key, color, name in (("integer_3d_ns129", SYM, "symmetric"), ("integer_3d_tau_ns129", ASYM, "$\\tau_0=0.5$")):
        r = runs[key]["routes"]
        s = [x["s"] for x in r]
        axes[1].semilogy(s, [x["J_wout_relative_l2"] for x in r], "s--", color=color, ms=4, lw=1)
        axes[1].semilogy(s, [x["J_native_relative_l2"] for x in r], "o-", color=color, ms=4, lw=1.4, label=name)
    axes[1].plot([], [], "s--", color="k", ms=4, label="from WOUT")
    axes[1].set(xlabel="s", ylabel="J error", title="native field (solid) vs WOUT, ns 129")
    for key, color, ls, label in (("integer_3d_ns65", SYM, "--", "symmetric, ns 65"),
                                  ("integer_3d_ns129", SYM, "-", "symmetric, ns 129"),
                                  ("integer_3d_tau_ns65", ASYM, "--", "$\\tau_0=0.5$, ns 65"),
                                  ("integer_3d_tau_ns129", ASYM, "-", "$\\tau_0=0.5$, ns 129")):
        s, err = np.asarray(runs[key]["surfaces"]).T
        axes[2].semilogy(s, err, ls, color=color, lw=1.4, label=label)
    axes[2].set(xlabel="s", ylabel="flux-label error", title="surfaces: resonant offset at $\\tau\\neq0$")
    for ax in axes:
        ax.grid(True, color="#e4e4e1", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1:]:
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.canvas.draw()
    Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).quantize(64).save(OUT, lossless=True, method=6)


if __name__ == "__main__":
    main()
