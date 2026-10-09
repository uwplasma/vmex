#!/usr/bin/env python
"""Re-solve a run's final coils at a higher resolution with the three-term free boundary, and export them for DESC.

    COIL_CASE=qa4-beta python benchmarks/three_term_highres.py <run dir> <out dir> [--modes 12 12] [--ns 101]

``<run dir>`` holds input.run, coils.json and wout.nc (a finished run of
``examples/optimization/single_stage_free_boundary_optimization_three_term.py``). The deck's profiles and the run's final
boundary (as the start) are re-solved at ``--modes``/``--ns`` with every interface condition (B.n, pressure balance,
no sheet current). Writes ``<out>/wout_three_term.nc``, ``<out>/report.json`` (interface residuals, QA, iota, aspect,
LCFS distance to the run's own boundary, time, memory) and ``<out>/coils.npz`` (filament points, tangents and
currents of every coil, with the field at check points, for cross-checks with
other codes).
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

os.environ["JAX_ENABLE_X64"] = "1"
p = argparse.ArgumentParser()
p.add_argument("run", type=Path)
p.add_argument("out", type=Path)
p.add_argument("--modes", type=int, nargs=2, default=(12, 12))
p.add_argument("--ns", type=int, default=101)
p.add_argument("--chunk", type=int, default=4)
p.add_argument("--quadrature", type=int, nargs=2, help="default 4 nfp 48 x 96")
p.add_argument("--export-only", action="store_true", help="write coils.npz and stop")
args = p.parse_args()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from essos.coils import Coils, Curves  # noqa: E402
import vmex as vj  # noqa: E402
from vmex.core.freeboundary_vc import ThreeTermFreeBoundaryModel  # noqa: E402
from vmex.core.optimize import axis_iota, solve_equilibrium, unpack_boundary  # noqa: E402


# The examples' COIL_CASE values this script reads: the target and the iota rows.
CASE = os.environ.get("COIL_CASE", "ellipse5")
if CASE not in ("ellipse5", "ellipse5-beta7", "qa3", "qh", "qi", "qa4-beta", "qa4-beta-tok", "qi6-beta", "qi6-beta-tok",
                "qh4-beta", "qh4-beta-tok"):
    raise ValueError(f"unknown COIL_CASE {CASE!r}")
# None: the constructed QI residual
HELICITY = (1, -1) if CASE == "qh" or CASE.startswith("qh4-beta") else None if CASE.startswith("qi") else (1, 0)
IOTA_AXIS = CASE.removesuffix("-tok") in ("qa4-beta", "qi6-beta", "qh4-beta")  # iota rows include opt.axis_iota and edge
QA_SURFACES = tuple(i / 10 for i in range(1, 11))
QI_SURFACES = tuple(i / 5 for i in range(1, 6))
QI_OPTIONS = dict(mboz=12, nboz=12, nphi=61, nalpha=18, n_bounce=21)


def target_residual():
    """Residual vector of the case's target: quasisymmetry of ``HELICITY``, or constructed QI."""
    import numpy as np
    from vmex import optimize as opt
    from vmex.core.qi import ConstructedQIResidual

    if HELICITY is None:
        return ConstructedQIResidual(np.asarray(QI_SURFACES), **QI_OPTIONS)
    return opt.QuasisymmetryRatioResidual(np.asarray(QA_SURFACES), *HELICITY)


def abs_iota(state, runtime):
    """|iota| bounded by the floor and ceiling rows.

    The half-mesh surfaces (axis slot excluded), as ``opt.min_abs_iota``; with
    ``IOTA_AXIS`` also the axis and the edge: ``opt.axis_iota``, the iota
    without its enclosed-current part extrapolated to the axis, where that
    current vanishes (VMEC's iotaf[0] = 1.5 iotas[1] - 0.5 iotas[2] mostly
    extrapolates the steep bootstrap part off the axis and drifts with ns),
    and VMEC's iotaf[-1] = 1.5 iotas[-1] - 0.5 iotas[-2].
    """
    import jax.numpy as jnp
    from vmex import optimize as opt
    from vmex.core.statephysics import _iotas_half  # private: opt exposes only the half-mesh minimum

    half = _iotas_half(state, runtime)[1:]
    if IOTA_AXIS:
        half = jnp.concatenate([opt.axis_iota(state, runtime)[None], half, 1.5 * half[-1:] - 0.5 * half[-2:-1]])
    return jnp.abs(half)


def min_abs_iota(state, runtime):
    """Smallest |iota| of ``abs_iota``: ``opt.min_abs_iota``, with ``IOTA_AXIS`` including the axis and edge."""
    import jax.numpy as jnp

    return jnp.min(abs_iota(state, runtime))


def boundary_from_wout(inp, wout):
    """``inp`` with the boundary of ``wout``'s last surface, truncated to its resolution."""
    import numpy as np

    rbc, zbs = np.zeros_like(np.asarray(inp.rbc)), np.zeros_like(np.asarray(inp.zbs))
    for m, n, r, z in zip(np.asarray(wout.xm, int), np.asarray(wout.xn, int) // int(wout.nfp),
                          np.asarray(wout.rmnc)[-1], np.asarray(wout.zmns)[-1]):
        if m < inp.mpol and abs(n) <= inp.ntor:
            rbc[n + inp.ntor, m], zbs[n + inp.ntor, m] = r, z
    return replace(inp, rbc=rbc, zbs=zbs)


def restart_input(run):
    """A finished run's ``input.run`` with its final WOUT's boundary, PHIEDGE and current (``--restart``)."""
    import vmex as vj

    inp, w = vj.VmecInput.from_file(run / "input.run"), vj.read_wout(run / "wout.nc")
    return current_from_wout(replace(boundary_from_wout(inp, w), phiedge=float(w.phi[-1]), lfreeb=False), w)


def current_from_wout(inp, w):
    """``inp`` with ``w``'s current profile -- its type, knots or coefficients -- and CURTOR, when ``inp``
    prescribes the current (a run that solved the current changes its type and knots: line_segment_ip)."""
    import numpy as np

    if int(inp.ncurr) == 1:
        kind = str(w.pcurr_type).strip()
        if "spline" in kind or "line_segment" in kind:  # VmecInput trims both to the knots before the -1 padding
            inp = replace(inp, pcurr_type=kind, ac_aux_s=np.asarray(w.ac_aux_s, dtype=float),
                          ac_aux_f=np.asarray(w.ac_aux_f, dtype=float), curtor=float(w.ctor))
        else:
            inp = replace(inp, pcurr_type=kind, ac=np.asarray(w.ac, dtype=float)[: np.size(inp.ac)],
                          curtor=float(w.ctor))
    return inp


@dataclass(frozen=True, eq=False)
class DirectCoilField:
    """Biot-Savart field of filament coils, as a differentiable pytree.

    ESSOS' ``BiotSavart`` jit-compiles its methods with the coil object as a
    static ``self``, so every new coil geometry recompiles and no derivative
    flows back to the coil arrays. This pytree carries the arrays as leaves
    instead: one compiled program serves every trial and JAX differentiates
    the field with respect to the coil parameters. The field is the mean over
    each coil's quadrature points of the filament Biot-Savart integrand.
    ``gamma``/``gamma_dash`` are ESSOS' points and tangents, shape
    ``(coils, points, 3)`` in metres; ``currents`` are in amperes.
    """

    gamma: Any
    gamma_dash: Any
    currents: Any

    def b_cyl(self, r, phi, z):
        """Evaluate the filament field in cylindrical coordinates, in tesla."""
        rr, pp, zz = jnp.broadcast_arrays(jnp.asarray(r), jnp.asarray(phi), jnp.asarray(z))
        cosine, sine = jnp.cos(pp), jnp.sin(pp)
        xyz = jnp.stack((rr * cosine, rr * sine, zz), axis=-1)
        displacement = xyz[..., None, None, :] - jnp.asarray(self.gamma)
        radius2 = jnp.sum(displacement * displacement, axis=-1)
        inv_radius3 = jnp.maximum(radius2, 1.0e-30) ** -1.5
        differential = jnp.cross(jnp.asarray(self.gamma_dash), displacement)
        differential = differential * inv_radius3[..., None]
        current_shape = (1,) * (xyz.ndim - 1) + (-1, 1, 1)
        weighted = differential * jnp.reshape(jnp.asarray(self.currents), current_shape)
        bxyz = 1.0e-7 * jnp.mean(jnp.sum(weighted, axis=-3), axis=-2)
        br = cosine * bxyz[..., 0] + sine * bxyz[..., 1]
        bphi = -sine * bxyz[..., 0] + cosine * bxyz[..., 1]
        return br, bphi, bxyz[..., 2]


jax.tree_util.register_dataclass(DirectCoilField, data_fields=["gamma", "gamma_dash", "currents"], meta_fields=[])


def nominal_field(coils):
    """The examples' ``CoilChart(coils, current_dofs=())`` field at its nominal point: the coils rebuilt from their
    unscaled Fourier coefficients and currents, as a differentiable :class:`DirectCoilField`."""
    raw = np.asarray(coils.dofs_curves) / np.asarray(coils.curves.scaling)[None, None, :]
    curves = Curves(jnp.asarray(raw), int(coils.n_segments), int(coils.nfp), bool(coils.stellsym))
    nominal = Coils(curves, jnp.asarray(np.asarray(coils.dofs_currents_raw, dtype=float)))
    return DirectCoilField(jnp.asarray(nominal.gamma), jnp.asarray(nominal.gamma_dash), jnp.asarray(nominal.currents))


args.out.mkdir(parents=True, exist_ok=True)
t0 = time.perf_counter()


def log(msg):
    print(f"[{time.perf_counter() - t0:7.0f}s] {msg}", flush=True)


mpol, ntor = args.modes
inp = restart_input(args.run).change_resolution(mpol=mpol, ntor=ntor, ntheta=2 * mpol + 6, nzeta=2 * ntor + 6)
inp = replace(inp, ns_array=np.array([args.ns]), ftol_array=np.array([1e-13]), niter_array=np.array([100000]))
field = nominal_field(Coils.from_json(str(args.run / "coils.json")))
rng = np.random.default_rng(0)
check = jnp.asarray(np.c_[1.0 + 0.3 * rng.uniform(-1, 1, 64), 0.3 * rng.uniform(-1, 1, 64), 0.3 * rng.uniform(-1, 1, 64)])
from vmex.core import virtual_casing as vc  # noqa: E402
B_check = np.asarray(vc.external_B_cartesian(field, check.T[:, :, None]))[:, :, 0].T
np.savez(args.out / "coils.npz", gamma=np.asarray(field.gamma), gamma_dash=np.asarray(field.gamma_dash),
         currents=np.asarray(field.currents), check_points=np.asarray(check), check_B=B_check)
log(f"deck {mpol}x{ntor} ns {args.ns}; {np.asarray(field.gamma).shape[0]} coils exported")
if args.export_only:
    sys.exit(0)

model = ThreeTermFreeBoundaryModel(inp, chunk=args.chunk, quadrature=args.quadrature or (4 * int(inp.nfp) * 48, 96),
                           trial_ftol=1e-11)
log(f"model: {model.x0.size} boundary coordinates")
out = model.solve_boundary(model.params0, field, verbose=2, max_nfev=80)
seconds = time.perf_counter() - t0
state, _, params, _ = out["aux"]
deck = unpack_boundary(model.fixed, out["x"], model.max_mode, vary_major_radius=True)
eq = solve_equilibrium(deck, initial_state=state)
w = eq.wout
vj.write_wout(str(args.out / "wout_three_term.nc"), w)
res = model.boundary_residual(eq.solution, model.with_boundary(model.params0, jnp.asarray(out["x"])), field)
q = np.asarray(target_residual().residuals_state(eq.solution, eq.solver_context))
th, planes = np.linspace(0, 2 * np.pi, 721), np.linspace(0, np.pi / int(inp.nfp), 4)


def lcfs_mm(wa, wb):
    d = 0.0
    for ph in planes:
        a, b = (np.outer(th, np.asarray(x.xm, float)) - np.asarray(x.xn, float) * ph for x in (wa, wb))
        Ra, Za = np.cos(a) @ np.asarray(wa.rmnc)[-1], np.sin(a) @ np.asarray(wa.zmns)[-1]
        Rb, Zb = np.cos(b) @ np.asarray(wb.rmnc)[-1], np.sin(b) @ np.asarray(wb.zmns)[-1]
        d = max(d, float(np.max(np.min(np.hypot(Ra[:, None] - Rb[None], Za[:, None] - Zb[None]), axis=1))))
    return round(1e3 * d, 3)


report = dict(run=str(args.run), modes=[mpol, ntor], ns=args.ns, boundary_coordinates=int(model.x0.size),
              seconds=round(seconds), nfev=out["nfev"], njev=out["njev"], fsq=float(w.fsqr + w.fsqz + w.fsql),
              normal=res.normal, pressure=res.pressure, sheet_current=res.sheet_current, qa=float(q @ q),
              min_abs_iota=float(min_abs_iota(eq.solution, eq.solver_context)),
              iota_axis=float(abs(axis_iota(eq.solution, eq.solver_context))), iota_edge=float(abs(np.asarray(w.iotaf)[-1])),
              aspect=float(w.aspect), lcfs_mm_vs_run=lcfs_mm(w, vj.read_wout(str(args.run / "wout.nc"))),
              peak_gpu_gb=round((jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 1e9, 2))
(args.out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
log("DONE " + json.dumps(report))
