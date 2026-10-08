#!/usr/bin/env python
"""Trace particles through a VMEX field tabulated by ESSOS ``InterpolatedField``.

VMEX solves the vacuum Landreman-Paul QA equilibrium and supplies the field
through its exterior API (``equilibrium.exterior_field``, here the ESSOS coil
field), the birth surface ``s = 0.25`` and the loss surface (the LCFS).
``essos.fields.InterpolatedField`` samples that field once on a cylindrical
grid around the LCFS and then reads a tricubic spline instead of summing over
every coil segment.  Guiding-centre and full-orbit ensembles are traced
through both fields and compared: final-position deviation, loss fraction,
energy error and wall time.

Requires ESSOS with ``essos.fields.InterpolatedField`` (uwplasma/ESSOS#135).
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
from essos.fields import BiotSavart, MagneticField
from essos.surfaces import SurfaceClassifier, surfacerzfourier_from_boundary

if not hasattr(essos.fields, "InterpolatedField"):
    raise SystemExit("This example needs essos.fields.InterpolatedField: install an ESSOS "
                     "release that includes uwplasma/ESSOS#135 (newer than 0.20.0)")
InterpolatedField = essos.fields.InterpolatedField

DATA = Path(__file__).resolve().parent / "data"
NPARTICLES, ENERGY_EV, TMAX_GC, TMAX_FO, GRID, NS = 8, 5.0e3, 2.0e-5, 1.0e-6, 48, 51
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    NPARTICLES, TMAX_GC, TMAX_FO, GRID, NS = 4, 2.0e-6, 1.0e-7, 32, 31

print("Solving the vacuum QA equilibrium and loading its ESSOS coils...")
inp = vj.VmecInput.from_file(DATA / "input.LandremanPaul2021_QA_lowres")
inp = replace(inp, phiedge=-0.025, ns_array=np.array([NS]), ftol_array=np.array([1e-12]),
              niter_array=np.array([8000]))
equilibrium = opt.solve_equilibrium(inp, verbose=False)
coils = BiotSavart(Coils.from_json(str(DATA / "ESSOS_biot_savart_LandremanPaulQA.json")))
exterior = equilibrium.exterior_field(
    external_field=lambda xyz: jax.vmap(coils.B)(xyz.reshape(-1, 3)).reshape(xyz.shape), plasma="vacuum")


@jax.tree_util.register_pytree_node_class
class VmexField(MagneticField):
    """The VMEX exterior field as a one-point ESSOS field."""

    def B(self, point):
        return exterior.B(point[None])[0]

    def sqrtg(self, point):
        return 1.0

    def to_xyz(self, point):
        return point

    def tree_flatten(self):
        return (), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls()


field = VmexField()
equilibrium.set_points_flux(np.stack([np.ones(256), np.linspace(0, 2 * np.pi, 256), np.repeat(
    np.linspace(0, np.pi / inp.nfp, 16), 16)], 1))
lcfs = np.asarray(equilibrium.field.get_points_cart()); R_lcfs = np.hypot(lcfs[:, 0], lcfs[:, 1])
margin = 0.05  # the spline stencil of the boundary cells reaches this far out
start = perf_counter()
interpolated = InterpolatedField(
    field, R=(R_lcfs.min() - margin, R_lcfs.max() + margin), Z=(-np.abs(lcfs[:, 2]).max() - margin,
    np.abs(lcfs[:, 2]).max() + margin), nr=GRID, nz=GRID, nphi=2 * GRID, nfp=inp.nfp, stellsym=True,
    chunk_size=4096)
interpolated.coefficients.block_until_ready()
print(f"Tabulated {GRID}x{GRID}x{GRID + 1} nodes in {perf_counter() - start:.1f} s")
equilibrium.set_points_flux(np.stack([np.full(64, 0.5), np.linspace(0, 2 * np.pi, 64),
                                      np.linspace(0, 2 * np.pi, 64)], 1))
probe = jnp.asarray(equilibrium.field.get_points_cart())
B_direct, B_interp = jax.vmap(field.B)(probe), jax.vmap(interpolated.B)(probe)
print(f"Interpolation error at s = 0.5: max |dB|/|B| = "
      f"{float(jnp.max(jnp.linalg.norm(B_interp - B_direct, axis=1) / jnp.linalg.norm(B_direct, axis=1))):.2e}")

equilibrium.set_points_flux(np.stack([np.full(NPARTICLES, 0.25), np.linspace(0, 2 * np.pi, NPARTICLES,
                                      endpoint=False), np.zeros(NPARTICLES)], 1))
births = jnp.asarray(equilibrium.field.get_points_cart())
classifier = SurfaceClassifier(surfacerzfourier_from_boundary(inp.rbc, inp.zbs, inp.nfp, nphi=32, ntheta=32), h=0.02)
results = {}
for model, tmax in (("GuidingCenterAdaptative", TMAX_GC), ("FullOrbitAdaptative", TMAX_FO)):
    for name, f in (("direct", field), ("interpolated", interpolated)):
        particles = Particles(initial_xyz=births, energy=ENERGY_EV * 1.602176634e-19,
                              initial_vparallel_over_v=jnp.linspace(-0.9, 0.9, NPARTICLES),
                              field=f if model.startswith("FullOrbit") else None)
        start = perf_counter()
        trace = Tracing(field=f, model=model, particles=particles, maxtime=tmax, times_to_trace=200,
                        atol=1e-9, rtol=1e-9, condition=lambda t, y, args, **kw: classifier.evaluate_xyz(y[:3]))
        trace.trajectories.block_until_ready()
        wall = perf_counter() - start
        lost = np.asarray(trace.loss_fraction_BioSavart(classifier)[2]) > 0
        energy_error = np.abs(np.asarray(trace.energy()) / particles.energy - 1)[~lost].max(initial=0.0)
        results[model, name] = trace, lost
        print(f"{model:24s} {name:12s}: {wall:7.2f} s, loss fraction {lost.mean():.3f}, "
              f"max energy error {energy_error:.1e}")
    (direct, lost0), (interp, lost1) = results[model, "direct"], results[model, "interpolated"]
    kept = ~(lost0 | lost1)
    deviation = np.linalg.norm(np.asarray(direct.trajectories[kept, -1, :3] - interp.trajectories[kept, -1, :3]), axis=1)
    print(f"{model:24s} final-position deviation of {kept.sum()} confined orbits after {tmax:.0e} s: "
          f"max {deviation.max() if kept.any() else np.nan:.2e} m")

figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
for axis, model in zip(axes, ("GuidingCenterAdaptative", "FullOrbitAdaptative")):
    (direct, lost0), (interp, lost1) = results[model, "direct"], results[model, "interpolated"]
    gap = np.linalg.norm(np.asarray(direct.trajectories[..., :3] - interp.trajectories[..., :3]), axis=-1)
    for i in np.flatnonzero(~(lost0 | lost1)):
        axis.semilogy(np.asarray(direct.times)[1:], gap[i, 1:], lw=1.0)
    axis.set(xlabel="t [s]", title=f"{model.replace('Adaptative', '')}: {lost0.mean():.2f} / {lost1.mean():.2f} lost")
axes[0].set_ylabel("|x_direct - x_interpolated| [m]")
figure.tight_layout(); figure.savefig("vmex_interpolated_particle_tracing.png", dpi=200); plt.close(figure)
print("Wrote vmex_interpolated_particle_tracing.png")
