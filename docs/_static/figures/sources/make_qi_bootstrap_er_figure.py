#!/usr/bin/env python3
"""Regenerate ``readme_qi_bootstrap_er.webp``, the README QI bootstrap-at-ambipolar-E_r figure.

Runs ``examples/optimization/QI_optimization_bootstrap_er.py`` at its default
settings in a scratch directory, then draws from the objects the script leaves
behind: <j.B> on the kinetic surfaces from DKX, the Redl formula and the VMEX
equilibrium before (the self-consistent Redl seed) and after the optimization,
the ambipolar E_r root before and after, and the objective history.  Output is
a 1100 px WebP.  Needs ``pip install "vmex[optimizers,neoclassical]"``.

Usage::

    python docs/_static/figures/sources/make_qi_bootstrap_er_figure.py
"""

from __future__ import annotations

import io
import os
import runpy
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from vmex.core.bootstrap import j_dot_B_redl, redl_geometry_from_state  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
EXAMPLE = REPO / "examples" / "optimization" / "QI_optimization_bootstrap_er.py"
OUT = REPO / "docs" / "_static" / "figures" / "readme_qi_bootstrap_er.webp"

os.environ.pop("VMEX_EXAMPLES_CI", None)
cwd = Path.cwd()
with tempfile.TemporaryDirectory() as tmp:
    os.chdir(tmp)
    try:
        ns = runpy.run_path(str(EXAMPLE))
    finally:
        os.chdir(cwd)

kinetic, profiles = ns["kinetic"], ns["profiles"]


def currents(equilibrium):
    """(s, VMEX, Redl, DKX) <j.B> in MA T/m^2 and the ambipolar E_r in kV/m."""
    s, j_vmex, j_dkx = kinetic.current_profiles(equilibrium.state, equilibrium.runtime)
    geometry = redl_geometry_from_state(equilibrium.state, equilibrium.runtime,
                                        surfaces=np.asarray(s), n_lambda=ns["REDL_N_LAMBDA"])
    j_redl = j_dot_B_redl(profiles, geometry, 0)[0]
    return (np.asarray(s), *(np.asarray(j) / 1e6 for j in (j_vmex, j_redl, j_dkx)),
            np.asarray(kinetic.electric_field(), dtype=float))


before = currents(ns["picard"].equilibrium)
after = currents(ns["final_equilibrium"])
monitor = ns["monitor"]
iteration = np.asarray([r.iteration for r in monitor.records])
total = np.asarray(monitor.history["total"])

plt.rcParams.update({"font.size": 12, "axes.titlesize": 12, "legend.fontsize": 10,
                     "axes.grid": True, "grid.alpha": 0.3})
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
for k, (label, d, marker) in enumerate((("seed", before, "o"), ("optimized", after, "s"))):
    for j, (name, color) in enumerate((("VMEX equilibrium", "C0"), ("Redl", "C2"), ("DKX at $E_r$", "C3"))):
        ax[0].plot(d[0], d[1 + j], marker + ("--" if k == 0 else "-"), color=color,
                   mfc="none" if k == 0 else color, label=f"{name}, {label}")
    ax[1].plot(d[0], d[4], marker + "-", label=label)
ax[0].set(xlabel=r"$s$", ylabel=r"$\langle j\cdot B\rangle$ [MA T/m$^2$]",
          title=r"bootstrap $\langle j\cdot B\rangle$")
ax[0].set_ylim(top=1.45 * max(d[2].max() for d in (before, after)))
ax[0].legend(ncol=2, fontsize=8, loc="upper center")
ax[1].set(xlabel=r"$s$", ylabel=r"$E_r$ [kV/m]", title=r"ambipolar ion root $E_r$")
ax[1].legend()
ax[2].plot(iteration, total, "o-")
ax[2].set(xlabel="accepted evaluation", ylabel="objective (monitor total)", yscale="log",
          title="objective history", xticks=iteration)
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=100)
Image.open(buf).convert("RGB").resize((1100, round(1100 * 4.2 / 14)), Image.LANCZOS).save(
    OUT, "WEBP", quality=40, method=6)
print(f"wrote {OUT}")
