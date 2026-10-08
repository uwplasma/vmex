#!/usr/bin/env python
"""Guiding centres from a VMEX equilibrium, through the LCFS, to a wall.

VMEX solves the vacuum Landreman-Paul QA equilibrium; ``vj.essos_tracing_fields``
turns it and the ESSOS coils into everything ``essos.dynamics.Tracing`` needs:
the VMEC field inside (handed over in memory), the exterior field from the
LCFS to a wall 3 cm outside it (``VmecExtender`` on the coils, tabulated by
ESSOS ``InterpolatedField``) and the wall.  The same orbits are traced through
the direct exterior field (``n=None``) to measure what the tricubic table
costs in accuracy and saves in time.

Requires ESSOS with ``InterpolatedField.around`` (uwplasma/ESSOS#135, #159).
"""

from dataclasses import replace
import os
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import vmex as vj
from vmex import optimize as opt

import essos.fields
from essos.coils import Coils
from essos.dynamics import Particles, Tracing

if not hasattr(getattr(essos.fields, "InterpolatedField", None), "around"):
    raise SystemExit("This example needs essos.fields.InterpolatedField.around: install an ESSOS "
                     "release that includes uwplasma/ESSOS#135 and #159 (newer than 0.20.1)")

DATA = Path(__file__).resolve().parent / "data"
NPARTICLES, ENERGY_EV, TMAX, GRID, NS = 16, 5.0e3, 2.0e-5, 48, 51
if os.environ.get("VMEX_EXAMPLES_CI") == "1":
    NPARTICLES, TMAX, GRID, NS = 8, 2.0e-6, 32, 31

inp = vj.VmecInput.from_file(DATA / "input.LandremanPaul2021_QA_lowres")
inp = replace(inp, phiedge=-0.025, ns_array=np.array([NS]), ftol_array=np.array([1e-12]),
              niter_array=np.array([8000]))
wout = opt.solve_equilibrium(inp, verbose=False).wout
coils = Coils.from_json(str(DATA / "ESSOS_biot_savart_LandremanPaulQA.json"))

key = jax.random.split(jax.random.key(0), 3)
births = jnp.stack([jax.random.uniform(key[0], (NPARTICLES,), minval=0.8, maxval=0.95),  # (s, theta, phi)
                    jax.random.uniform(key[1], (NPARTICLES,), maxval=2 * jnp.pi),
                    jnp.zeros(NPARTICLES)], 1)
particles = Particles(initial_xyz=births, energy=ENERGY_EV * 1.602176634e-19,
                      initial_vparallel_over_v=jax.random.uniform(key[2], (NPARTICLES,), minval=-1, maxval=1))

results = {}
for name, n in (("interpolated", GRID), ("direct", None)):
    start = perf_counter()
    setup = vj.essos_tracing_fields(wout, coils, wall=0.03, n=n, plasma="vacuum")
    trace = Tracing(**setup, model="GuidingCenterAdaptative", particles=particles, maxtime=TMAX,
                    times_to_trace=200, atol=1e-9, rtol=1e-9)
    trace.trajectories.block_until_ready()
    results[name] = trace
    print(f"{name:12s}: {perf_counter() - start:7.2f} s, crossed the LCFS {np.isfinite(trace.lcfs_times).mean():.3f}, "
          f"struck the wall {trace.wall_hits.mean():.3f}")

direct, interp = results["direct"], results["interpolated"]
both = direct.wall_hits & interp.wall_hits
print(f"Wall strikes in both fields: {both.sum()}; strike-time difference "
      f"max {np.abs(direct.wall_times - interp.wall_times)[both].max(initial=0.0):.2e} s, strike-position "
      f"difference max {np.linalg.norm(direct.wall_positions - interp.wall_positions, axis=1)[both].max(initial=0.0):.2e} m")

wall = setup["wall"]
figure = plt.figure(figsize=(6, 5))
axis = figure.add_subplot(projection="3d")
wall.plot(ax=axis, show=False, alpha=0.15)
for xyz in np.asarray(interp.trajectories_xyz):
    axis.plot(*xyz.T, lw=0.6)
axis.scatter(*interp.wall_positions[interp.wall_hits].T, color="k", s=10, label="wall strikes")
axis.legend()
figure.tight_layout(); figure.savefig("vmex_interpolated_particle_tracing.png", dpi=200); plt.close(figure)
print("Wrote vmex_interpolated_particle_tracing.png")
