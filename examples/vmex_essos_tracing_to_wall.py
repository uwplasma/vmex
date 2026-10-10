#!/usr/bin/env python
"""Guiding centres from a VMEX equilibrium, through the LCFS, to a wall.

VMEX solves the vacuum Landreman-Paul QA equilibrium of the ESSOS coils.
``vj.essos_tracing_fields`` hands ESSOS the three things its ``Tracing`` needs:
the VMEC field inside (in memory), the field from the LCFS out to the wall
(``VmecExtender`` on the coils, tabulated by ESSOS ``InterpolatedField``) and
the wall, 3 cm outside the LCFS.  Each orbit then stays inside, crosses the
LCFS and comes back, or strikes the wall.  The same births traced through the
direct (untabulated) exterior field check the table.
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
from PIL import Image

import vmex as vj
from vmex import optimize as opt
from essos.coils import Coils
from essos.constants import ONE_EV, PROTON_MASS, ELEMENTARY_CHARGE
from essos.dynamics import Particles, Tracing
from essos.surfaces import SurfaceRZFourier

DATA = Path(__file__).resolve().parent / "data"
NPARTICLES, ENERGY_EV, TMAX, WALL, GRID, NS = 64, 3e3, 4e-5, 0.03, 48, 51
FIGURE = Path("vmex_essos_tracing_to_wall.webp")
if os.environ.get("VMEX_EXAMPLES_CI") == "1":  # smoke pass of the test suite
    NPARTICLES, TMAX, GRID, NS = 8, 2e-6, 24, 31

# 1. Equilibrium and coils
inp = vj.VmecInput.from_file(DATA / "input.LandremanPaul2021_QA_lowres")
inp = replace(inp, phiedge=-0.025, ns_array=np.array([NS]), ftol_array=np.array([1e-12]),
              niter_array=np.array([8000]))
wout = opt.solve_equilibrium(inp, verbose=False).wout
coils = Coils.from_json(str(DATA / "ESSOS_biot_savart_LandremanPaulQA.json"))

# 2. Protons born between s = 0.7 and 0.95 with random pitch
k = jax.random.split(jax.random.key(0), 4)
births = jnp.stack([jax.random.uniform(k[0], (NPARTICLES,), minval=0.7, maxval=0.95),  # (s, theta, phi)
                    jax.random.uniform(k[1], (NPARTICLES,), maxval=2 * jnp.pi),
                    jax.random.uniform(k[2], (NPARTICLES,), maxval=2 * jnp.pi)], 1)
particles = Particles(initial_xyz=births, mass=PROTON_MASS, charge=ELEMENTARY_CHARGE, energy=ENERGY_EV * ONE_EV,
                      initial_vparallel_over_v=jax.random.uniform(k[3], (NPARTICLES,), minval=-1, maxval=1))

# 3. Trace from the axis to the wall, through the tabulated and the direct exterior field
traces = {}
for name, n in (("interpolated", GRID), ("direct", None)):
    start = perf_counter()
    setup = vj.essos_tracing_fields(wout, coils, wall=WALL, n=n, plasma="vacuum")
    traces[name] = Tracing(**setup, model="GuidingCenterAdaptative", particles=particles, maxtime=TMAX,
                           times_to_trace=1000, atol=1e-9, rtol=1e-9)
    traces[name].trajectories.block_until_ready()
    print(f"{name:12s} {perf_counter() - start:6.1f} s, struck the wall {traces[name].wall_hits.mean():.3f}")
trace, direct = traces["interpolated"], traces["direct"]

crossed, struck = np.isfinite(trace.lcfs_times), np.asarray(trace.wall_hits)
fates = {"stay inside": ~crossed, "cross and return": crossed & ~struck & (trace.returns > 0),
         "strike the wall": struck}
for label, mask in fates.items():
    print(f"{label:17s} {mask.sum():3d} of {NPARTICLES}")
both = struck & direct.wall_hits
gap = np.linalg.norm(trace.wall_positions[both] - direct.wall_positions[both], axis=1)
print(f"Strikes in both fields: {both.sum()}, largest strike-point distance {gap.max(initial=0.0):.1e} m")

# 4. Orbits in 3D, and their points near phi = 0 in the R, Z plane
colors = {"stay inside": "#3b6fb6", "cross and return": "#e08a1e", "strike the wall": "#c0392b"}
vmec, wall = setup["field"], setup["wall"]
lcfs = SurfaceRZFourier.from_vmec(vmec, ntheta=48, nphi=96)
xyz = np.asarray(trace.trajectories_xyz)
fig = plt.figure(figsize=(10.0, 4.6))
grid_spec = fig.add_gridspec(1, 2, width_ratios=(1.5, 1.0), wspace=0.02)
ax = fig.add_subplot(grid_spec[0], projection="3d")
g = np.asarray(lcfs.gamma)
ax.plot_surface(*np.moveaxis(g, -1, 0), color="0.75", alpha=0.25, linewidth=0)
for label, mask in fates.items():
    for i in np.flatnonzero(mask)[:5]:
        ax.plot(*xyz[i].T, lw=1.0, color=colors[label])
ax.scatter(*trace.wall_positions[struck].T, color="k", marker="x", s=22, depthshade=False, label="wall strikes")
lim = np.abs(g[..., :2]).max()
ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-0.5 * lim, 0.5 * lim))
ax.set_box_aspect((1, 1, 0.5), zoom=1.45)
ax.set_axis_off()
ax.view_init(elev=38, azim=-60)
ax.legend(loc="upper left", fontsize=8, frameon=False)

ax = fig.add_subplot(grid_spec[1])
for surface, style, label in ((lcfs, "-", "LCFS"), (wall, "--", f"wall, {WALL * 100:.0f} cm out")):
    c = np.asarray(surface.gamma)[0]
    c = np.vstack([c, c[:1]])
    ax.plot(np.hypot(c[:, 0], c[:, 1]), c[:, 2], "k" + style, lw=1, label=label)
period = 2 * np.pi / vmec.nfp
near = np.abs(np.mod(np.arctan2(xyz[..., 1], xyz[..., 0]) + period / 2, period) - period / 2) < 0.03
for label, mask in fates.items():
    sel = near & mask[:, None]
    ax.scatter(np.hypot(xyz[..., 0], xyz[..., 1])[sel], xyz[..., 2][sel], s=1.5, color=colors[label],
               label=f"{label} ({mask.sum()})")
ax.set(xlabel="R [m]", ylabel="Z [m]", aspect="equal")
ax.set_title(f"{NPARTICLES} protons, {ENERGY_EV / 1e3:.0f} keV, {TMAX * 1e6:.0f} $\\mu$s; points near $\\phi = 0$",
             fontsize=9)
ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), markerscale=5, frameon=False)
fig.subplots_adjust(left=0.0, right=0.84, top=0.92, bottom=0.11)
fig.canvas.draw()  # 64-color lossless WebP keeps the docs copy small
Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB").quantize(64).save(
    FIGURE, lossless=True, method=6)
print(f"Wrote {FIGURE}")
