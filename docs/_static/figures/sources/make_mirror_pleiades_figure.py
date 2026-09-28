#!/usr/bin/env python3
"""Draw ``docs/_static/figures/readme_mirror_pleiades.webp``.

The two-coil free-boundary mirror of ``examples/data/input.mirror_two_coil_free_boundary``
against the independent Pleiades reference
``examples/data/pleiades_two_coil_beta_reference.csv``: the on-axis midplane
field ratio ``B(beta)/B_vac`` at 0, 1, 3 and 10 % central beta, and its
difference from the finest Pleiades grid at two VMEX resolutions.

``--solve`` re-solves VMEX (about 3 h for both rungs on a loaded 36-core CPU) and rewrites
``benchmarks/pleiades_two_coil_mirror.json``; without it the committed record
is plotted.

Usage::

    python docs/_static/figures/sources/make_mirror_pleiades_figure.py [--solve]
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
DATA = REPO / "examples" / "data"
RECORD = REPO / "benchmarks" / "pleiades_two_coil_mirror.json"
OUT = REPO / "docs" / "_static" / "figures" / "readme_mirror_pleiades.webp"
BETAS = [0.0, 0.01, 0.03, 0.10]
RUNGS = [(7, 7, 13), (11, 9, 17)]
B_VAC = 0.0836001422205
VMEX, PLEIADES, MUTED = ("#2a78d6", "#8fb8e8"), "#eb6834", "#6b6b68"


def solve() -> None:
    from vmex.mirror import MirrorInput, solve_mirror_beta_scan

    inp = MirrorInput.from_file(DATA / "input.mirror_two_coil_free_boundary")
    rows = []
    for ns, elements, nxi in RUNGS:
        t = time.perf_counter()
        sols = solve_mirror_beta_scan(replace(inp, ns=ns, elements=elements, nxi=nxi), BETAS)
        ratio = [float(s.summary()["axis_field_center"]) / B_VAC for s in sols]
        rows.append({"ns": ns, "elements": elements, "nxi": nxi, "field_ratio": ratio,
                     "wall_s": round(time.perf_counter() - t, 1)})
        print(rows[-1], flush=True)
    RECORD.write_text(json.dumps({
        "case": "examples/data/input.mirror_two_coil_free_boundary",
        "reference": "examples/data/pleiades_two_coil_beta_reference.csv",
        "betas": BETAS, "b_vac": B_VAC, "rungs": rows,
        "date": time.strftime("%Y-%m-%d"),
    }, indent=1) + "\n")


def main() -> None:
    if "--solve" in sys.argv or not RECORD.exists():
        solve()
    rec = json.loads(RECORD.read_text())
    ref = np.loadtxt(DATA / "pleiades_two_coil_beta_reference.csv", delimiter=",", skiprows=5)
    plt.rcParams.update({"font.size": 9})
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 3.2))
    grids = sorted({(int(r[0]), int(r[1])) for r in ref})
    for k, (nr, nz) in enumerate(grids):
        sel = (ref[:, 0] == nr) & (ref[:, 1] == nz)
        a.plot(np.r_[0, 100 * ref[sel, 2]], np.r_[1, ref[sel, 7]], "o", mfc="none", ms=5 + 3 * k,
               color=PLEIADES, alpha=0.4 + 0.3 * k, label=f"Pleiades {nr}x{nz}")
    best = ref[(ref[:, 0] == grids[-1][0]) & (ref[:, 1] == grids[-1][1])]
    fine = np.r_[1.0, best[:, 7]]
    beta = 100 * np.asarray(rec["betas"])
    for color, row in zip(VMEX, rec["rungs"]):
        label = f"VMEX ns={row['ns']}, {row['elements']} elements"
        a.plot(beta, row["field_ratio"], "-", color=color, lw=1.5, label=label)
        b.semilogy(beta, np.abs(np.asarray(row["field_ratio"]) - fine), "s-", color=color, ms=4,
                   label=label)
    coarse = ref[(ref[:, 0] == grids[0][0]) & (ref[:, 1] == grids[0][1])]
    b.semilogy(100 * coarse[:, 2], np.abs(coarse[:, 7] - best[:, 7]), "o--", color=PLEIADES, ms=4,
               mfc="none", label=f"Pleiades {grids[0][0]}x{grids[0][1]} (its own grid error)")
    a.set(xlabel="central beta [%]", ylabel="$B_0(\\beta)/B_{vac}$", title="On-axis midplane field")
    b.set(xlabel="central beta [%]", ylabel="|VMEX - Pleiades 51x101|",
          title="Difference halves under refinement")
    for ax in (a, b):
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=7, framealpha=0.9, edgecolor="none")
    fig.tight_layout()
    fig.savefig(OUT, dpi=110, pil_kwargs={"quality": 85, "method": 6})
    plt.close(fig)


if __name__ == "__main__":
    main()
