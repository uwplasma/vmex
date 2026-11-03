#!/usr/bin/env python
"""Bounce-averaged drift surfaces of trapped particles in a solved equilibrium.

A trapped particle's bounce centre drifts on a surface of constant second
adiabatic invariant J. This script solves a quasi-axisymmetric vacuum
equilibrium, takes its Boozer spectrum on a radial grid, and integrates the
bounce-centre motion of one pitch from several launch radii with
``drift_surfaces_from_boozer``. It prints the largest radius ``s_max`` of each
orbit (an orbit reaching s = 1 is lost) and the conservation of J along it,
then differentiates ``s_max`` with ``jax.grad`` and plots the J contours with
the orbits in x = s cos(alpha), y = s sin(alpha).
"""

import os
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import vmex as vj
from vmex import optimize as opt
from vmex.core.maxj import drift_surfaces_from_boozer
from vmex.core.omnigenity import boozer_spectrum_state

# Input deck:
INPUT_FILE = Path(__file__).resolve().parent / "data" / "input.LandremanPaul2021_QA_lowres"

# Boozer surfaces the action J(s, alpha) is sampled on, and Boozer resolution:
SURFACES = np.linspace(0.1, 1.0, 10)
MBOOZ, NBOOZ = 12, 12

# Pitch lambda = 1 / B*, with B* this fraction of the way through the range of
# |B| shared by every surface (0 = deepest wells only, 1 = barely trapped):
PITCH_FRACTION = 0.5

# Launch radii s0 of the orbits, all at alpha = 0:
LAUNCH_RADII = np.array([0.2, 0.4, 0.6, 0.8])

# Field lines per surface and Runge-Kutta steps per orbit:
NALPHA = 32
NSTEPS = 256

# Figure written to the working directory:
FIGURE_PATH = Path("drift_surfaces.png")

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    SURFACES, NALPHA, NSTEPS = np.linspace(0.2, 1.0, 6), 16, 64

###############################################################################
# End of input parameters.
###############################################################################

### Solve the equilibrium and transform to Boozer coordinates #################

equilibrium = opt.solve_equilibrium(vj.VmecInput.from_file(INPUT_FILE))
booz = boozer_spectrum_state(equilibrium.solution, equilibrium.solver_context,
                             surfaces=SURFACES, mboz=MBOOZ, nboz=NBOOZ)
s_b = booz["psi_b"] / booz["psi_edge"]

# |B| range on the first and last surface brackets the shared trapped interval.
theta, zeta = np.meshgrid(np.linspace(0, 2 * np.pi, 41),
                          np.linspace(0, 2 * np.pi / booz["nfp"], 41))
angle = (np.asarray(booz["xm_b"])[:, None, None] * theta
         - np.asarray(booz["xn_b"])[:, None, None] * zeta)
b_all = np.einsum("km,mij->kij", np.asarray(booz["bmnc_b"]), np.cos(angle))
b_low, b_high = b_all.min(axis=(1, 2)).max(), b_all.max(axis=(1, 2)).min()
pitch = 1.0 / (b_low + PITCH_FRACTION * (b_high - b_low))

### Integrate the drift surfaces ##############################################

fields = ("iota_b", "G_b", "I_b")
def drift(bmnc_b, launch=LAUNCH_RADII):
    return drift_surfaces_from_boozer(
        bmnc_b=bmnc_b, xm_b=booz["xm_b"], xn_b=booz["xn_b"], nfp=booz["nfp"],
        s_b=s_b, pitch=pitch, s0=launch, nalpha=NALPHA, nsteps=NSTEPS,
        **{name: booz[name] for name in fields})

out = drift(booz["bmnc_b"])
for s0, s_max, action in zip(LAUNCH_RADII, out["s_max"], out["action"]):
    spread = (action.max() - action.min()) / abs(action.mean())
    print(f"s0 = {s0:.2f}: s_max = {s_max:.4f}, "
          f"J conserved to {spread:.1e} (relative), lost: {bool(s_max >= 1.0)}")

### Sensitivity of the largest radius #########################################

# d s_max / d bmnc_b: how each Boozer |B| harmonic on each surface moves the
# outermost excursion of the orbit launched at LAUNCH_RADII[-1].
grad = jax.grad(lambda b: drift(b, LAUNCH_RADII[-1:])["s_max"][0])(booz["bmnc_b"])
k, m = np.unravel_index(np.argmax(np.abs(grad)), grad.shape)
print(f"largest d s_max / d bmnc_b = {float(grad[k, m]):+.3e} "
      f"at s = {float(s_b[k]):.2f}, (m, n) = ({int(booz['xm_b'][m])}, {int(booz['xn_b'][m])})")

### Plot ######################################################################

alpha = np.append(out["alpha_grid"], 2 * np.pi)
radius = np.asarray(out["s_grid"])[:, None]
j_map = np.asarray(out["j_map"])
j_map = np.concatenate([j_map, j_map[:, :1]], axis=1)

fig, ax = plt.subplots(figsize=(6, 5.5))
contours = ax.contour(radius * np.cos(alpha), radius * np.sin(alpha), j_map, levels=14, cmap="viridis")
fig.colorbar(contours, ax=ax, label=r"$J / v$")
for k, (s_orbit, a_orbit) in enumerate(zip(np.asarray(out["s"]), np.asarray(out["alpha"]))):
    ax.plot(s_orbit * np.cos(a_orbit), s_orbit * np.sin(a_orbit), "k--", lw=1.2,
            label="drift surface" if k == 0 else None)
ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="gray"))
ax.set_aspect("equal")
ax.set_xlabel(r"$s\cos\alpha$")
ax.set_ylabel(r"$s\sin\alpha$")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(FIGURE_PATH, dpi=200)
print(f"Wrote {FIGURE_PATH}")
