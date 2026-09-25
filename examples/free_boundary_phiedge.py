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
The figure compares the seed boundary with the LCFS that meets the target.
"""

import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import vmex as vj

# The ESSOS coil set, and the plasma deck that seeds the initial guess:
DATA_DIR = Path(__file__).resolve().parent / "data"
COILS_FILE = DATA_DIR / "ESSOS_biot_savart_LandremanPaulQA.json"
INPUT_FILE = DATA_DIR / "input.LandremanPaul2021_QA_lowres"

# Target LCFS radius at the outboard midplane (theta = 0, phi = 0) [m]:
R_OUTBOARD_TARGET = 1.285

# First PHIEDGE guess [Wb] and root-solve controls:
PHIEDGE0 = -0.025
RTOL, MAX_SOLVES = 1e-4, 12

# Cylindrical grid the coil field is tabulated onto, and the plasma resolution
# (NZETA must divide the grid's toroidal plane count kp):
GRID = dict(rmin=0.45, rmax=1.55, zmin=-0.6, zmax=0.6, ir=96, jz=96, kp=32)
MPOL, NTOR, NZETA = 5, 5, 16
NS, NITER, FTOL = 31, 20000, 1e-10

# Directory that receives every output file, and whether to draw the figure:
OUTPUT_DIR = Path("output_free_boundary_phiedge")
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
      f"target {R_OUTBOARD_TARGET:.4f} m")

### Solve for PHIEDGE #########################################################

start = time.perf_counter()
solved, result = vj.solve_phiedge(
    inp, coil_field, R_OUTBOARD_TARGET, metric="r_outboard",
    phiedge0=PHIEDGE0, rtol=RTOL, max_iter=MAX_SOLVES)
elapsed = time.perf_counter() - start

### Print, plot and save ######################################################

wout = vj.wout_from_result(solved, result)
r_out = float(np.sum(np.asarray(wout.rmnc)[-1]))
print(f"PHIEDGE = {solved.phiedge:.6f} Wb  ->  R_outboard = {r_out:.5f} m "
      f"(|error| {abs(r_out - R_OUTBOARD_TARGET):.1e} m), volume = "
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
    m = np.arange(inp.rbc.shape[1])
    fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=110)
    ax.plot(np.cos(np.outer(theta, m)) @ inp.rbc.sum(axis=0),
            np.sin(np.outer(theta, m)) @ inp.zbs.sum(axis=0),
            "--", color="#9a9a9a", label="seed boundary")
    R, Z = surface_rz(wout, s_index=-1, theta=theta, phi=np.array([0.0]))
    ax.plot(R[:, 0], Z[:, 0], color="#2e6da4", lw=2.0,
            label=f"LCFS, PHIEDGE = {solved.phiedge:.4f} Wb")
    ax.plot(R_OUTBOARD_TARGET, 0.0, "x", ms=10, mew=2, color="#c0392b", label="target point")
    ax.set(xlabel="R [m]", ylabel="Z [m]", title="Free-boundary LP-QA at $\\phi = 0$")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "phiedge_target.png")
    print(f"Wrote {OUTPUT_DIR / 'phiedge_target.png'}")
