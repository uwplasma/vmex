#!/usr/bin/env python
"""Find the PHIEDGE whose free-boundary plasma reaches a given outboard point.

A free-boundary run takes the enclosed toroidal flux PHIEDGE as input and
returns the last closed flux surface.  Codes without a flux input (HINT, for
example) are instead pinned by a geometric point, so comparing them on the
same coils needs the inverse map: the PHIEDGE whose LCFS passes through a
target outboard-midplane radius.  ``vj.solve_phiedge`` finds it with a
bracketed secant, one warm-started free-boundary solve per step.

The coils are the Landreman & Paul (2021) precise-QA set optimized in ESSOS,
tabulated once into an in-memory ``MgridField``; the plasma is a vacuum
(zero pressure, zero current) so the target is purely a flux-surface one.
The figure shows each iterate: its metric value against PHIEDGE, and its LCFS.
"""

import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import vmex as vj
from vmex.core.freeboundary import _phiedge_metric

# The ESSOS coil set, and the plasma deck that seeds the initial guess:
DATA_DIR = Path(__file__).resolve().parent / "data"
COILS_FILE = DATA_DIR / "ESSOS_biot_savart_LandremanPaulQA.json"
INPUT_FILE = DATA_DIR / "input.LandremanPaul2021_QA_lowres"

# What the LCFS must match.  METRIC is "r_outboard" (LCFS R at theta = phi = 0,
# in m), "volume" (plasma volume, m^3), or any callable (inp, result) -> float,
# for example the edge rotational transform:
#     METRIC = lambda inp, result: float(vj.wout_from_result(inp, result).iotaf[-1])
# Reachable r_outboard on these coils at NS = 16 is roughly 1.277-1.308 m.
METRIC, TARGET = "r_outboard", 1.285

# First PHIEDGE guess [Wb] and root-solve controls:
PHIEDGE0 = -0.025
RTOL, MAX_SOLVES = 1e-4, 12

# Cylindrical grid the coil field is tabulated onto, and the plasma resolution
# (NZETA must divide the grid's toroidal plane count kp):
GRID = dict(rmin=0.45, rmax=1.55, zmin=-0.6, zmax=0.6, ir=96, jz=96, kp=32)
MPOL, NTOR, NZETA = 5, 5, 16
NS, NITER, FTOL = 31, 20000, 1e-10

# Directory that receives the wout, and the figure (the docs copy, lossless WebP):
OUTPUT_DIR = Path("output_free_boundary_phiedge")
FIGURE = Path(__file__).resolve().parents[1] / "docs/_static/figures/free_boundary_phiedge.webp"
MAKE_PLOTS = True

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    GRID = dict(GRID, ir=48, jz=48)
    NS, NITER, FTOL, RTOL = 16, 4000, 1e-8, 1e-3
    MAKE_PLOTS = False

###############################################################################
# End of input parameters.
###############################################################################

### Set up the coils and the deck #############################################

from essos.coils import Coils  # noqa: E402 (optional heavy import)

coil_field = vj.MgridField.from_coils(Coils.from_json(str(COILS_FILE)), **GRID)

inp = vj.VmecInput.from_file(INPUT_FILE)
k = inp.ntor - NTOR
inp = replace(
    inp, lfreeb=True, mgrid_file="essos_coils(direct)", mpol=MPOL, ntor=NTOR,
    nzeta=NZETA,
    rbc=inp.rbc[k:k + 2 * NTOR + 1, :MPOL], zbs=inp.zbs[k:k + 2 * NTOR + 1, :MPOL],
    rbs=inp.rbs[k:k + 2 * NTOR + 1, :MPOL], zbc=inp.zbc[k:k + 2 * NTOR + 1, :MPOL],
    raxis_c=inp.raxis_c[:NTOR + 1], zaxis_s=inp.zaxis_s[:NTOR + 1],
    raxis_s=inp.raxis_s[:NTOR + 1], zaxis_c=inp.zaxis_c[:NTOR + 1],
    ns_array=[NS], niter_array=[NITER], ftol_array=[FTOL], pres_scale=0.0)
print(f"seed boundary: R(theta=0, phi=0) = {float(np.sum(inp.rbc)):.4f} m; "
      f"target {TARGET:.4f} m")

### Solve for PHIEDGE #########################################################

# Wrap the metric to record every iterate (PHIEDGE, value, wout) for the figure:
evaluate = _phiedge_metric(METRIC)
iterates = []


def recorded_metric(trial, result):
    value = evaluate(trial, result)
    iterates.append((trial.phiedge, value, vj.wout_from_result(trial, result)))
    print(f"  PHIEDGE = {trial.phiedge:.6f} Wb  ->  {value:.6f}")
    return value


start = time.perf_counter()
solved, result = vj.solve_phiedge(
    inp, coil_field, TARGET, metric=recorded_metric,
    phiedge0=PHIEDGE0, rtol=RTOL, max_iter=MAX_SOLVES)
elapsed = time.perf_counter() - start

### Print, plot and save ######################################################

wout = vj.wout_from_result(solved, result)
r_out = float(np.sum(np.asarray(wout.rmnc)[-1]))
print(f"PHIEDGE = {solved.phiedge:.6f} Wb  ->  R_outboard = {r_out:.5f} m "
      f"(|error| {abs(r_out - TARGET):.1e} m), volume = "
      f"{float(wout.volume_p):.4f} m^3, {elapsed:.1f} s")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
wout_path = vj.write_wout(OUTPUT_DIR / "wout_LandremanPaul_QA_phiedge.nc", wout)
print(f"Wrote {wout_path}")

if MAKE_PLOTS:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vmex.core.plotting import surface_rz

    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(iterates)))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.0, 4.0), dpi=100, width_ratios=(1.6, 1.0))
    phi, value = np.array([(p, v) for p, v, _ in iterates]).T
    ax0.axhline(TARGET, color="#c0392b", lw=1.0, ls="--", label="target")
    ax0.plot(phi, value, color="#9a9a9a", lw=0.8)
    for n, (p, v, wout) in enumerate(iterates):
        ax0.plot(p, v, "o", color=colors[n])
        ax0.annotate(str(n + 1), (p, v), textcoords="offset points", xytext=(5, 4), fontsize=8)
        R, Z = surface_rz(wout, s_index=-1, theta=theta, phi=np.array([0.0]))
        ax1.plot(R[:, 0], Z[:, 0], color=colors[n], lw=1.0, label=f"{n + 1}: {p:.4f} Wb")
    label = {"r_outboard": "LCFS R at theta = phi = 0 [m]", "volume": "volume [m^3]"}
    ax0.set(xlabel="PHIEDGE [Wb]", ylabel=label.get(METRIC, "metric"),
            title="Iterates of the PHIEDGE root solve")
    ax0.legend(frameon=False, fontsize=8)
    if METRIC == "r_outboard":
        ax1.plot(TARGET, 0.0, "x", ms=10, mew=2, color="#c0392b", label="target point")
    ax1.set(xlabel="R [m]", ylabel="Z [m]", title="LCFS at $\\phi = 0$")
    ax1.set_aspect("equal")
    ax1.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7, frameon=False)
    for ax in (ax0, ax1):
        ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    from PIL import Image  # 16-color lossless WebP keeps the docs copy near 12 KB

    fig.canvas.draw()
    Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB").quantize(16).save(
        FIGURE, lossless=True, method=6)
    print(f"Wrote {FIGURE}")
