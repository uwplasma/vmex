#!/usr/bin/env python3
"""Draw ``docs/_static/figures/readme_extender_islands.webp``.

Reads only ``benchmarks/extender_islands_sections.npz``: phi = 0 Poincare
sections recorded by two external benchmarks (github.com/rogeriojorge/neutral-beam-hsx
and github.com/rogeriojorge/vmex-hint-benchmark, case Q0). Both equilibria are
at zero beta, so the extended field (plasma part plus coils) equals the coil field.

* HSX (QHS): field lines launched on the VMEX surfaces s = 0.25 to 1, and 1 to
  4 cm outside the LCFS, followed for 200 field periods (an outside line is
  drawn until it first leaves the frame), over the WOUT surfaces.
* Landreman-Paul QA: the same seeds traced through the coil field and through
  HINT's relaxed field (128^2 grid), 300 transits, with the VMEX surfaces of
  the free-boundary solve matched to the traced surface just inside the
  iota = 2/5 island chain.

Usage::

    python docs/_static/figures/sources/make_extender_islands_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
RECORD = REPO / "benchmarks" / "extender_islands_sections.npz"
OUT = REPO / "docs" / "_static" / "figures" / "readme_extender_islands.webp"
INSIDE, OUTSIDE, HINT, VMEX = "#2a78d6", "#eb6834", "#eb6834", "#1b1b1b"


def main() -> None:
    d = np.load(RECORD)
    plt.rcParams.update({"font.size": 9, "svg.hashsalt": "vmex"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 4.4), gridspec_kw={"width_ratios": [0.7, 1]})

    P, s0 = d["hsx_poincare"], d["hsx_s"]
    n = len(s0)
    a.scatter(*P[:n].reshape(-1, 2).T, s=0.3, color=INSIDE, lw=0, label="from VMEX surfaces")
    out = P[n:]  # a line from outside the LCFS is kept until it first leaves the frame
    keep = np.cumprod((np.abs(out[..., 0] - 1.31) < 0.31) & (np.abs(out[..., 1]) < 0.3), 1).astype(bool)
    a.scatter(*out[keep].T, s=1.5, color=OUTSIDE, lw=0, label="from 1 to 4 cm outside (open)")
    for i, (r, z) in enumerate(zip(d["hsx_wout_R"], d["hsx_wout_Z"])):
        a.plot(r, z, color=VMEX, lw=0.5 + 0.7 * (i == len(d["hsx_wout_R"]) - 1),
               label="VMEX s = 0.25, 0.6, 1" if i == 0 else None)
    a.set(xlim=(1.3, 1.68), ylim=(-0.3, 0.3), title="HSX (QHS), $\\phi$ = 0")

    closed = d["lpqa_closed"]
    for key, color, size, label in (("coils", "#9a9a96", 1.2, "coils"),
                                    ("hint", HINT, 0.4, "HINT field, same seeds")):
        R, Z = d[f"lpqa_{key}_R"], d[f"lpqa_{key}_Z"]
        b.scatter(R[closed].ravel(), Z[closed].ravel(), s=size, color=color, lw=0, label=label)
        b.scatter(R[~closed].ravel(), Z[~closed].ravel(), s=size * 2, color=color, lw=0)
    for i, (r, z) in enumerate(zip(d["lpqa_vmex_R"], d["lpqa_vmex_Z"])):
        b.plot(r, z, color=VMEX, lw=0.5 + 0.7 * (i == len(d["lpqa_vmex_R"]) - 1),
               label="VMEX surfaces" if i == 0 else None)
    b.set(title="Landreman-Paul QA vs HINT, $\\phi$ = 0")

    for ax, loc in ((a, "upper right"), (b, "lower right")):
        ax.set(aspect="equal", xlabel="R [m]", ylabel="Z [m]")
        ax.grid(alpha=0.2, lw=0.5)
        ax.legend(loc=loc, fontsize=6.5, markerscale=6, framealpha=0.9, edgecolor="none")
    fig.tight_layout()
    fig.savefig(OUT, dpi=110, pil_kwargs={"quality": 80, "method": 6})
    plt.close(fig)


if __name__ == "__main__":
    main()
