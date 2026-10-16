#!/usr/bin/env python3
"""Draw ``desc_protocol.webp`` from ``benchmarks/desc_protocol.json``.

The record follows the comparison protocol of Panici et al. (2023),
J. Plasma Phys. 89, 955890303: every code's output is loaded into DESC
(``desc.vmec.VMECIO``) and scored with one metric, the volume-averaged force
error normalised by the pressure gradient, ``<|F|>/<|grad p|>``, taken over
0.2 <= rho <= 0.9.  Left: that error against wall time on DSHAPE as the
poloidal resolution is raised.  Right: near-axis rotational transform of the
Landreman-Paul QA (beta = 2.5 %, bootstrap current) against radial resolution.

Usage::

    python benchmarks/make_desc_protocol_figure.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RECORD = REPO / "benchmarks" / "desc_protocol.json"
FIGURE = REPO / "docs" / "_static" / "figures" / "desc_protocol.webp"
STYLE = {"vmex": ("VMEX", "#2a6fdb", "o"), "cpp_vmec": ("C++ VMEC", "#e0803a", "s"),
         "vmec2000": ("VMEC2000", "#3a9e5c", "^"), "desc": ("DESC", "#c94f6d", "D")}


def main():
    rec = json.loads(RECORD.read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.5), constrained_layout=True)
    for code, rows in rec["force_vs_time"]["DSHAPE"].items():
        label, color, marker = STYLE[code]
        a.loglog([r["wall_s"] for r in rows], [r["f_norm_core"] for r in rows], marker=marker, color=color,
                 label=label, ms=4, lw=1, alpha=0.85, ls="-" if code == "desc" else "none")
    a.set(xlabel="wall time [s] (warm)", ylabel=r"$\langle|F|\rangle / \langle|\nabla p|\rangle$",
          title="DSHAPE, mpol 4 to 12, ns 51")
    a.legend(fontsize=7, frameon=False)
    iota = rec["near_axis_iota"]
    for code, r in iota["codes"].items():
        label, color, marker = STYLE[code]
        b.plot(r["ns"], r["iota_0_1"], marker=marker, color=color, label=label, ms=4, lw=1, alpha=0.85)
    b.axhline(iota["desc_reference"], color=STYLE["desc"][1], ls="--", lw=1, label="DESC")
    b.set_xscale("log", base=2)
    b.set_xticks([65, 129, 257], ["65", "129", "257"])
    b.set(xlabel="ns", ylabel=r"$\iota(\rho = 0.1)$", title=r"QA, $\beta$ = 2.5 %, bootstrap")
    b.legend(fontsize=7, frameon=False)
    fig.canvas.draw()
    Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).quantize(8).save(
        FIGURE, lossless=True, method=6)


if __name__ == "__main__":
    main()
