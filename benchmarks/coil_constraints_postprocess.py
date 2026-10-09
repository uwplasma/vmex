#!/usr/bin/env python
"""Post-process a finished run of either coil-constraint single-stage example.

    python benchmarks/coil_constraints_postprocess.py runs/free                 # free-boundary arm
    python benchmarks/coil_constraints_postprocess.py runs/fixed --match-flux   # fixed-boundary arm
    COIL_CASE=qi6-beta python benchmarks/coil_constraints_postprocess.py runs/free-qi6 --no-dense

The runs are those of ``examples/optimization/single_stage_free_boundary_optimization_coil_constraints.py``
and ``single_stage_optimization_coil_constraints.py``. Run it with the run's
``COIL_CASE``: the limits and targets are the examples' (the case values this
script reads are repeated below). Reads what both optimization scripts write:
``metrics.jsonl``, the initial coils, the ``coils.stepN.json`` /
``wout.stepN.nc`` checkpoints and the final ``coils.json`` / ``wout.nc``.
Writes into ``<run>/postprocess``:

- ``history.csv``, ``loss.png`` (QA and objective) and ``constraints.png``:
  every logged constrained quantity against its limits (iota floor and
  ceiling, aspect, major radius, mirror ratio, bootstrap mismatch, clearance,
  B.n/|B|, edge R B_phi, ...; ``limits``). At beta > 0 the fixed arm's
  normal-field limit is drawn on the total B.n it constrains;
- ``evolution.gif``: coils and LCFS at every checkpoint, coloured by |B| of
  the coil field, and ``initial_final.png``;
- with ``--poincare N``, ``poincare.png`` / ``poincare.json``: field lines of
  the final coils traced for N toroidal transits from 5 points on each of the
  flux surfaces ``POINCARE_SURFACES`` of the run's own WOUT (free boundary for
  the free arm, the prescribed boundary for the fixed arm), with each line's
  distance from its surface; islands and stochastic layers show up here even
  though the nested-surface equilibrium cannot represent them;
- ``wout_dense.nc``, ``dense.json`` and the ``vmex.plot_wout`` figures: the
  final (or latest checkpoint) coils solved free-boundary at ``--ns`` (default
  201), the like-for-like comparison of both arms: QA, min and max |iota|,
  aspect, major radius, the mirror ratio where the case limits it and, for
  ``--bootstrap`` runs (solved at the run's final current), the Redl mismatch
  and for DKX cases the DKX one. A fixed-arm run need not enclose PHIEDGE; with
  ``--match-flux`` its currents are rescaled by the logged coil-flux ratio,
  which in vacuum changes the field strength only.
- with ``--trace``, ``trace/``: ``vmex --trace`` alpha losses of that dense
  equilibrium, scaled in memory to ARIES-CS size (3.52 MeV alphas; 10 ms
  prompt losses by default, ``--trace-args "--collisional --trace-tmax 0.2"``
  for slowing-down losses).
"""

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

# ---- cases ----------------------------------------------------------------------------------------------------------
# COIL_CASE selects the case of the examples; only the values read here are repeated. The values are ellipse5's; the
# blocks after them override them per case.
DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
CASE = os.environ.get("COIL_CASE", "ellipse5")
RESOLUTION = (8, 8, 51)            # MPOL, NTOR of the dense solve; NS of the Redl surfaces
GRID = (64, 64)                    # NTHETA, NZETA
EQUILIBRIUM_FTOL = 1e-15
QA_SURFACES = tuple(i / 10 for i in range(1, 11))
INPUT_FILE = DATA / "input.rotating_ellipse_nfp2"
B0 = 1.0                           # T
# --bootstrap runs: the reactor-like kinetic profiles of the examples and their bootstrap model.
REACTOR_R0, REACTOR_B0, REACTOR_N0, REACTOR_T0 = 8.0, 6.0, 1.5e20, 15.0e3   # m, T, 1/m^3, eV
REDL_SURFACES = None              # None: every VMEC half-grid surface, as simsopt's RedlGeomVmec
REDL_N_LAMBDA, REDL_TOLERANCE = 32, 1e-3
BOOTSTRAP_MODEL = "redl"          # "dkx": the DKX kinetic <j.B> replaces Redl in the self-consistency row
DKX_SURFACES, DKX_COLLISION_OPERATOR = None, 0  # None: every half-mesh surface; 0: momentum-conserving Fokker-Planck
SEED = None                        # (nfp, aspect, b / a_eff): rotating ellipse replacing the deck's boundary
HELICITY = (1, 0)                  # quasisymmetry (M, N); None: the constructed QI residual
TARGET_NAME = "QA"
QI_SURFACES = tuple(i / 5 for i in range(1, 6))
QI_OPTIONS = dict(mboz=12, nboz=12, nphi=61, nalpha=18, n_bounce=21)  # examples/optimization/QI_optimization.py
MIRROR_LIMIT = None                # upper limit on the edge mirror ratio
IOTA_FLOOR = 0.41
IOTA_CEILING = None                # upper limit on max |iota|
IOTA_AXIS = False                  # True: floor and ceiling also bound the axis iota (opt.axis_iota) and edge iotaf
ASPECT_RANGE = (4.9, 5.1)
RADIUS_TARGET, RADIUS_TOLERANCE, RADIUS_MARGIN = 1.0, 0.01, 0.001
COIL_ORDER, N_SEGMENTS = 16, 256
COIL_SURFACE_DISTANCE_LIMIT = 0.20 # m
NORMAL_FIELD_CONSTRAINT = 0.008   # fixed arm: limit on the area-weighted RMS B.n/|B| (total B at beta > 0)
FIELD_STRENGTH_TOLERANCE = 0.005  # fixed arm, beta > 0: relative band on edge R B_phi around the coils' mu0 I / 2 pi

if CASE == "ellipse5-beta7":
    IOTA_FLOOR = 0.16
elif CASE in ("qa3", "qh", "qi"):
    COIL_ORDER, COIL_SURFACE_DISTANCE_LIMIT = 8, 0.15
    if CASE == "qa3":
        SEED, ASPECT_RANGE = (3, 6.0, 0.5), (5.9, 6.1)
    elif CASE == "qh":
        SEED, HELICITY, TARGET_NAME, ASPECT_RANGE, IOTA_FLOOR = (4, 6.0, 0.9), (1, -1), "QH", (5.9, 6.1), 1.1
    else:
        SEED, HELICITY, TARGET_NAME, ASPECT_RANGE, MIRROR_LIMIT = (4, 8.0, 0.5), None, "QI", (7.9, 8.1), 0.21
        IOTA_FLOOR = 0.51
elif CASE.removesuffix("-tok") in ("qa4-beta", "qi6-beta", "qh4-beta"):
    COIL_ORDER = 12
    IOTA_AXIS, RADIUS_TOLERANCE, RADIUS_MARGIN = True, 1e-3, 1e-4
    if CASE.startswith("qa4-beta"):
        SEED, IOTA_FLOOR, ASPECT_RANGE = (2, 4.0, 0.5), 0.27, (3.5, 4.5)
    elif CASE.startswith("qh4-beta"):
        SEED, HELICITY, TARGET_NAME, ASPECT_RANGE, IOTA_FLOOR = (4, 6.0, 0.9), (1, -1), "QH", (5.9, 6.1), 1.1
        COIL_SURFACE_DISTANCE_LIMIT = 0.15
    else:
        SEED, HELICITY, TARGET_NAME, ASPECT_RANGE, MIRROR_LIMIT = (4, 6.0, 0.7), None, "QI", (5.9, 6.1), 0.21
        IOTA_FLOOR, IOTA_CEILING, BOOTSTRAP_MODEL = 0.86, 0.98, "dkx"
        COIL_SURFACE_DISTANCE_LIMIT = 0.15
elif CASE != "ellipse5":
    raise ValueError(f"unknown COIL_CASE {CASE!r}")
if CASE.endswith("-tok"):
    SEED = (SEED[0], SEED[1], 0.01)  # a circular tokamak with a 1% helical ripple
REDL_HELICITY = 0 if HELICITY is None else HELICITY[1]  # Redl's quasisymmetry N (simsopt convention)

POINCARE_SURFACES = (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
POINCARE_POINTS_PER_SURFACE = 5


# ---- the examples' seed, targets and current helpers ----------------------------------------------------------------
def seed_input():
    """The case's fixed-boundary seed at the optimization resolution.

    With ``SEED = (nfp, aspect, ratio)`` the boundary is the rotating ellipse
    R = R0 + a cos(theta) - b cos(theta + nfp phi), Z = a sin(theta) + b sin(theta + nfp phi)
    with a^2 - b^2 = (R0 / aspect)^2, b = ratio R0 / aspect and PHIEDGE for B0 ~ 1 T;
    otherwise it is ``INPUT_FILE``'s. ``finite_beta_input`` then calibrates PHIEDGE.
    """
    import numpy as np
    import vmex as vj

    mpol, ntor, ns = RESOLUTION
    inp = vj.VmecInput.from_file(INPUT_FILE)
    if SEED is not None:
        inp = replace(inp, nfp=SEED[0])
    inp = inp.change_resolution(mpol=mpol, ntor=ntor, ntheta=GRID[0], nzeta=GRID[1])
    if SEED is not None:
        _, aspect, ratio = SEED
        minor = RADIUS_TARGET / aspect
        rbc, zbs = np.zeros_like(inp.rbc), np.zeros_like(inp.zbs)
        rbc[ntor, 0] = RADIUS_TARGET
        rbc[ntor, 1] = zbs[ntor, 1] = minor * np.sqrt(1.0 + ratio**2)
        rbc[ntor - 1, 1], zbs[ntor - 1, 1] = -ratio * minor, ratio * minor
        inp = replace(inp, rbc=rbc, zbs=zbs, phiedge=np.pi * minor**2)
    return replace(inp, ns_array=np.array([ns]), ftol_array=np.array([EQUILIBRIUM_FTOL]), lfreeb=False)


def target_residual():
    """Residual vector of the case's target: quasisymmetry of ``HELICITY``, or constructed QI."""
    import numpy as np
    from vmex import optimize as opt
    from vmex.core.qi import ConstructedQIResidual

    if HELICITY is None:
        return ConstructedQIResidual(np.asarray(QI_SURFACES), **QI_OPTIONS)
    return opt.QuasisymmetryRatioResidual(np.asarray(QA_SURFACES), *HELICITY)


def redl_surfaces():
    """Surfaces of the Redl self-consistency check: ``REDL_SURFACES``, or by default every VMEC
    half-grid surface s = (j - 1/2) / (ns - 1) at the run's ns, as simsopt's ``RedlGeomVmec``."""
    import numpy as np

    if REDL_SURFACES is not None:
        return np.asarray(REDL_SURFACES)
    ns = int(RESOLUTION[2])
    return (np.arange(1, ns) - 0.5) / (ns - 1)


def redl_profiles(inp):
    """The kinetic profiles of ``bootstrap_input`` for a deck's calibrated pressure, and their Redl mismatch."""
    import numpy as np
    from vmex.core.bootstrap import ELEMENTARY_CHARGE, KineticProfiles, RedlBootstrapMismatch

    r0, b0 = float(inp.rbc[inp.ntor, 0]), B0
    t0 = REACTOR_T0 * (b0 / REACTOR_B0) ** (2 / 3) * (r0 / REACTOR_R0) ** (1 / 3)
    n0 = REACTOR_N0 * (b0 / REACTOR_B0) ** (4 / 3) * (REACTOR_R0 / r0) ** (1 / 3)
    scale = float(inp.pres_scale) / (2 * ELEMENTARY_CHARGE * n0 * t0)
    n0, t0 = n0 * scale ** (2 / 3), t0 * scale ** (1 / 3)
    profiles = KineticProfiles(n0 * np.array([1.0, 0, 0, 0, 0, -1.0]), t0 * np.array([1.0, -1.0]),
                               t0 * np.array([1.0, -1.0]))
    return profiles, RedlBootstrapMismatch(profiles, REDL_HELICITY, redl_surfaces(), n_lambda=REDL_N_LAMBDA)


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


def max_abs_iota(state, runtime):
    """Largest |iota| of ``abs_iota``, the counterpart of ``min_abs_iota``."""
    import jax.numpy as jnp

    return jnp.max(abs_iota(state, runtime))


def dkx_mismatch(inp):
    """(state, runtime) -> DKX's bootstrap mismatch for ``inp``'s kinetic profiles (needs the ``dkx`` package)."""
    import numpy as np
    from dkx.bootstrap import KineticBootstrapMismatch

    kinetic = KineticBootstrapMismatch(redl_profiles(inp)[0], surfaces=redl_surfaces() if DKX_SURFACES is None else DKX_SURFACES,
                                       collision_operator=DKX_COLLISION_OPERATOR,
                                       mboz=QI_OPTIONS["mboz"], nboz=QI_OPTIONS["nboz"])

    def mismatch(state, runtime):
        # DKX reads the radial grid on the host. Under jit it is a tracer, but it is
        # always linspace(0, 1, ns), so hand DKX that concrete grid.
        grid = np.linspace(0.0, 1.0, runtime.setup.s_full.shape[0])
        return kinetic.total(state, replace(runtime, setup=replace(runtime.setup, s_full=grid)))

    return mismatch


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


def resize_coils(coils, order, n_segments):
    """Zero-pad higher Fourier modes without changing the curves or currents."""
    import jax.numpy as jnp
    from essos.coils import Coils, Curves

    if order < coils.order:
        raise ValueError(f"cannot truncate order-{coils.order} coils to order {order}")
    old = coils.curves
    raw = jnp.pad(old.dofs / old.scaling, ((0, 0), (0, 0), (0, 2 * (order - coils.order))))
    curves = Curves(raw, n_segments=n_segments, nfp=coils.nfp, stellsym=coils.stellsym,
                    scaling_type=old.scaling_type, scaling_factor=old.scaling_factor, scale_fixed=old.scale_fixed)
    return Coils(curves, coils.dofs_currents_raw, currents_scale=coils.currents_scale)


# ---- post-processing -------------------------------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="run directory written by either example")
    parser.add_argument("--ns", type=int, default=201, help="radial resolution of the dense free-boundary solve")
    parser.add_argument("--no-dense", action="store_true", help="skip the dense free-boundary solve")
    parser.add_argument("--match-flux", action="store_true", help="rescale fixed-arm currents to enclose PHIEDGE")
    parser.add_argument("--seed-wout", type=Path, help="initial state of the dense solve (default: the run's wout.nc)")
    parser.add_argument("--max-iterations", type=int, default=12000, help="iteration cap of the dense solve")
    parser.add_argument("--poincare", type=int, default=0, metavar="N",
                        help="trace the final coils' field lines for N toroidal transits (0: skip)")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--trace", action="store_true", help="vmex --trace alpha losses of the dense equilibrium")
    parser.add_argument("--trace-args", default="", help='extra vmex --trace flags, e.g. "--collisional"')
    return parser.parse_args(argv)


def read_history(run):
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    keys = [k for k in rows[0] if all(isinstance(r.get(k), (int, float)) for r in rows)]
    return rows, keys


def limits(keys):
    """Logged quantities and their bounds, as the two benchmarks impose them (None: no bound)."""
    width = RADIUS_TOLERANCE - RADIUS_MARGIN
    # At beta > 0 the fixed arm limits the total (B_coils + B_plasma).n/|B|, logged as
    # total_normal_field_rms, and logs the coil-only normal_field_rms unconstrained.
    coil_only_limit = None if "total_normal_field_rms" in keys else NORMAL_FIELD_CONSTRAINT
    return {
        "min_abs_iota": ("min |iota|", IOTA_FLOOR, None),
        "max_abs_iota": ("max |iota|", None, IOTA_CEILING),
        "aspect": ("aspect ratio", *ASPECT_RANGE),
        "mirror_ratio": ("mirror ratio", None, MIRROR_LIMIT),
        "major_radius_m": ("major radius [m]", RADIUS_TARGET - width, RADIUS_TARGET + width),
        "redl_mismatch": (f"bootstrap mismatch ({BOOTSTRAP_MODEL})", None, REDL_TOLERANCE),
        "coil_surface_distance_m": ("coil-plasma distance [m]", COIL_SURFACE_DISTANCE_LIMIT, None),
        "coil_minimum_scaled_slack": ("min coil slack", 0.0, None),
        "plasma_minimum_scaled_slack": ("min plasma-row slack", 0.0, None),
        "normal_field_rms": ("rms coil-only B.n/|B|", None, coil_only_limit),
        "total_normal_field_rms": ("rms (B_coils + B_plasma).n/|B|", None, NORMAL_FIELD_CONSTRAINT),
        "rbtor_ratio": ("edge R B_phi / coils' mu0 I / 2 pi", 1 - FIELD_STRENGTH_TOLERANCE,
                        1 + FIELD_STRENGTH_TOLERANCE),
        "flux_ratio": ("coil flux / PHIEDGE", None, None),
        "phiedge_factor": ("PHIEDGE / seed PHIEDGE", None, None),
    }


def plot_history(rows, keys, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (out / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    step = [r["step"] for r in rows]
    figure, axis = plt.subplots(figsize=(7, 4))
    for key, label in (("qa", TARGET_NAME), ("objective", "objective")):
        if key in keys:
            axis.semilogy(step, [max(r[key], 1e-16) for r in rows], label=label)
    axis.set_xlabel("accepted step"), axis.grid(True, alpha=0.3), axis.legend()
    figure.tight_layout(), figure.savefig(out / "loss.png", dpi=200), plt.close(figure)

    panels = [(k, *spec) for k, spec in limits(keys).items() if k in keys]
    columns = 3
    figure, axes = plt.subplots((len(panels) + columns - 1) // columns, columns,
                                figsize=(4 * columns, 3 * ((len(panels) + columns - 1) // columns)), squeeze=False)
    for axis, (key, label, lower, upper) in zip(axes.flat, panels):
        axis.plot(step, [r[key] for r in rows], lw=1)
        for bound in (lower, upper):
            if bound is not None:
                axis.axhline(bound, color="r", ls="--", lw=0.8)
        axis.set_title(label, fontsize=9), axis.set_xlabel("accepted step", fontsize=8), axis.grid(True, alpha=0.3)
    for axis in list(axes.flat)[len(panels):]:
        axis.set_visible(False)
    figure.tight_layout(), figure.savefig(out / "constraints.png", dpi=200), plt.close(figure)


def checkpoints(run):
    """(label, coils, wout) for every saved step, then the final state."""
    steps = sorted(int(m.group(1)) for p in run.glob("coils.step*.json")
                   if (m := re.fullmatch(r"coils\.step(\d+)\.json", p.name)) and (run / f"wout.step{m.group(1)}.nc").exists())
    frames = [(f"step {s}", run / f"coils.step{s}.json", run / f"wout.step{s}.nc") for s in steps]
    if (run / "coils.json").exists() and (run / "wout.nc").exists():
        frames.append(("final", run / "coils.json", run / "wout.nc"))
    return frames


def plot_evolution(frames, out, final_step=None):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import vmex as vj
    from essos.coils import Coils
    from essos.fields import BiotSavart
    from essos.surfaces import SurfaceRZFourier

    objects = [(SurfaceRZFourier.from_wout_file(str(w), nphi=64, ntheta=32), Coils.from_json(str(c)))
               for _, c, w in frames]

    def modB(x, pair):
        surface, coils = pair
        points = jnp.asarray(surface.gamma).reshape(-1, 3)
        return jnp.linalg.norm(jax.vmap(BiotSavart(coils).B)(points), axis=-1).reshape(surface.gamma.shape[:-1])

    # Title each frame with the optimizer step it was saved at, not its frame index.
    labels = [f"step {final_step}" if label == "final" and final_step is not None else label for label, _, _ in frames]
    vj.plot_optimization_movie(out / "evolution.gif", [np.array([i]) for i in range(len(objects))],
                               lambda x: objects[int(x[0])], color_factory=modB, color_label="|B| [T]",
                               frame_labels=labels)
    vj.plot_optimization_objects(out / "initial_final.png", (labels[0], *objects[0]), (labels[-1], *objects[-1]))


def poincare(frame, transits, out):
    """Poincare section at phi = 0 of the coils' field, seeded on the run's own flux surfaces."""
    from fractions import Fraction

    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import vmex
    from essos.coils import Coils
    from essos.dynamics import Tracing
    from essos.fields import BiotSavart

    label, coil_path, wout_path = frame
    wout = vmex.read_wout(wout_path)
    ns, nfp, xm = int(wout.ns), int(wout.nfp), np.asarray(wout.xm)
    s_full = np.linspace(0.0, 1.0, ns)
    rmnc = np.stack([np.interp(POINCARE_SURFACES, s_full, c) for c in np.asarray(wout.rmnc).T], axis=1)
    zmns = np.stack([np.interp(POINCARE_SURFACES, s_full, c) for c in np.asarray(wout.zmns).T], axis=1)
    theta = np.linspace(0.0, 2.0 * np.pi, 720)
    curves = [(np.cos(np.outer(theta, xm)) @ r, np.sin(np.outer(theta, xm)) @ z) for r, z in zip(rmnc, zmns)]
    seeds = 2.0 * np.pi * (np.arange(POINCARE_POINTS_PER_SURFACE) + 0.5) / POINCARE_POINTS_PER_SURFACE
    points = np.array([(np.cos(xm * t) @ r, 0.0, np.sin(xm * t) @ z)
                       for r, z in zip(rmnc, zmns) for t in seeds])
    field = BiotSavart(Coils.from_json(str(coil_path)))
    r_axis = float(np.asarray(wout.rmnc)[0].sum())
    b_axis = float(jnp.linalg.norm(field.B(jnp.array([r_axis, 0.0, 0.0]))))
    started = time.monotonic()
    # dx/dt = B: a transit is about 2 pi R0 / |B| long, and field lines are longer than 2 pi R0.
    tracing = Tracing(field=field, model="FieldLineAdaptative", initial_conditions=jnp.asarray(points),
                      maxtime=1.6 * transits * 2.0 * np.pi * r_axis / b_axis, times_to_trace=transits * 96,
                      atol=1e-10, rtol=1e-10)
    trajectories = np.asarray(tracing.trajectories)
    seconds = time.monotonic() - started

    iota = np.abs(np.asarray(wout.iotas)[1:])
    s_half = 0.5 * (s_full[1:] + s_full[:-1])
    rationals = sorted({Fraction(nfp * k, m) for m in range(1, 9) for k in range(1, 4)
                        if iota.min() <= nfp * k / m <= iota.max()})
    resonances = {str(q): np.round(s_half[np.flatnonzero(np.diff(np.sign(iota - float(q))))], 3).tolist()
                  for q in rationals}
    lcfs = np.stack(curves[-1], axis=1) if POINCARE_SURFACES[-1] == 1.0 else None
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(POINCARE_SURFACES)))
    figure, axis = plt.subplots(figsize=(7, 7))
    lines, sections = [], {}
    for k, trajectory in enumerate(trajectories):
        j = k // POINCARE_POINTS_PER_SURFACE
        x, y, z = trajectory[:, :3].T
        phase = np.unwrap(np.arctan2(y, x))
        cross = np.flatnonzero(np.diff(np.floor(phase / (2.0 * np.pi))) != 0)
        f = (2.0 * np.pi * np.round(phase[cross + 1] / (2.0 * np.pi)) - phase[cross]) / np.diff(phase)[cross]
        R = np.hypot(x, y)[cross] + f * np.diff(np.hypot(x, y))[cross]
        Z = z[cross] + f * np.diff(z)[cross]
        axis.scatter(R, Z, s=0.4, color=colors[j], zorder=2)
        sections[f"line{k}_R"], sections[f"line{k}_Z"] = R, Z
        surface = np.stack(curves[j], axis=1)
        d = np.min(np.hypot(R[:, None] - surface[None, :, 0], Z[:, None] - surface[None, :, 1]), axis=1) if len(R) else np.array([np.nan])
        outside = bool(lcfs is not None and len(R) and (R.max() > lcfs[:, 0].max() + 0.01 or R.min() < lcfs[:, 0].min() - 0.01
                                                        or np.abs(Z).max() > np.abs(lcfs[:, 1]).max() + 0.01))
        lines.append(dict(s=POINCARE_SURFACES[j], theta=float(seeds[k % POINCARE_POINTS_PER_SURFACE]), crossings=int(len(R)),
                          rms_distance_m=float(np.sqrt(np.nanmean(d**2))), max_distance_m=float(np.nanmax(d)),
                          leaves_lcfs=outside))
    # The run's own flux surfaces on top of the crossings: dashed, the LCFS solid.
    for j, (R, Z) in enumerate(curves):
        edge = POINCARE_SURFACES[j] == 1.0
        axis.plot(R, Z, color="black", lw=1.0 if edge else 0.8, ls="-" if edge else "--", zorder=4)
    axis.scatter([], [], s=12, color=colors[len(colors) // 2], label="field-line crossings, coloured by seed surface")
    axis.plot([], [], color="black", ls="--", lw=0.8,
              label="WOUT flux surfaces s = " + ", ".join(f"{s:g}" for s in POINCARE_SURFACES[:-1]))
    axis.plot([], [], color="black", lw=1.0, label="WOUT LCFS")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), fontsize=7, frameon=False)
    axis.set_title(f"{CASE} {label}: coil-field Poincaré section, φ = 0, ≥{transits} transits")
    axis.set_xlabel("R [m]"), axis.set_ylabel("Z [m]"), axis.set_aspect("equal")
    figure.tight_layout(), figure.savefig(out / "poincare.png", dpi=200), plt.close(figure)
    per_surface = {str(s): dict(max_distance_m=max(line["max_distance_m"] for line in lines if line["s"] == s),
                                leaving_lcfs=sum(line["leaves_lcfs"] for line in lines if line["s"] == s))
                   for s in POINCARE_SURFACES}
    report = dict(frame=label, coils=str(coil_path), wout=str(wout_path), transits=transits, seconds=seconds,
                  iota_axis=float(iota[0]), iota_edge=float(iota[-1]), resonances=resonances,
                  surfaces=per_surface, lines=lines)
    (out / "poincare.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(out / "poincare.npz", surfaces=np.asarray(POINCARE_SURFACES), **sections)
    return dict(frame=label, transits=transits, seconds=round(seconds), resonances=resonances, surfaces=per_surface)


def dense_solve(frame, args, rows, out):
    import numpy as np
    import vmex
    from vmex import optimize as opt
    # Private: the one free-boundary stage solve that accepts an ESSOS field and a seed state at a new NS.
    from vmex.core.freeboundary import _solve_free_boundary_stage, free_boundary_resolution
    from vmex.core.solver import prepare_runtime
    from essos.coils import Coils
    from essos.fields import BiotSavart

    label, coil_path, wout_path = frame
    mpol, ntor, _ = RESOLUTION
    run = Path(coil_path).parent
    if (run / "input.run").exists():
        inp = vmex.VmecInput.from_file(run / "input.run").change_resolution(
            mpol=mpol, ntor=ntor, ntheta=GRID[0], nzeta=GRID[1])
    else:  # older runs: the case's seed deck
        inp = seed_input()
    row = rows[-1] if label == "final" else next(r for r in rows if f"step {r['step']}" == label)
    bootstrap = "redl_mismatch" in row  # --bootstrap run: input.run holds the seed current, the WOUT the final one
    run_wout = vmex.read_wout(wout_path)
    if bootstrap:
        inp = current_from_wout(inp, run_wout)
    phiedge = float(run_wout.phi[-1])  # a free-PHIEDGE run ends at its own PHIEDGE
    inp = replace(inp, lfreeb=True, mgrid_file="direct ESSOS field", ns_array=np.array([args.ns]), phiedge=phiedge,
                  ftol_array=np.array([EQUILIBRIUM_FTOL]), niter_array=np.array([args.max_iterations]))
    coils = resize_coils(Coils.from_json(str(coil_path)), COIL_ORDER, N_SEGMENTS)
    scale = 1.0 / row["flux_ratio"] if args.match_flux else 1.0
    if scale != 1.0:
        coils = Coils(coils.curves, coils.dofs_currents_raw * scale, currents_scale=coils.currents_scale)
    field = BiotSavart(coils)
    seed = args.seed_wout or wout_path
    state = vmex.state_from_wout(vmex.read_wout(seed), inp=inp, ns=args.ns)
    resolution = free_boundary_resolution(inp, field, ns=args.ns)
    started = time.monotonic()
    stage = _solve_free_boundary_stage(
        inp, external_field=field, resolution=resolution, initial_state=state, ftol=EQUILIBRIUM_FTOL,
        max_iterations=args.max_iterations, include_edge_in_convergence=True, edge_force_tolerance=EQUILIBRIUM_FTOL,
        use_fft=False, error_on_no_convergence=False, jacobian_retries=0, allow_initial_axis_reguess=False)
    result = stage.result
    forces = {k: float(getattr(result, k)) for k in ("fsqr", "fsqz", "fsql", "fedge")}
    converged = bool(result.converged and result.vacuum is not None
                     and all(np.isfinite(v) and 0 <= v <= EQUILIBRIUM_FTOL for v in forces.values()))
    report = dict(frame=label, coils=str(coil_path), ns=args.ns, seed=str(seed), current_scale=scale, currents_A=np.asarray(coils.currents).tolist(),
                  converged=converged, iterations=int(result.iterations), seconds=time.monotonic() - started, **forces)
    if converged:
        wout = vmex.wout_from_state(inp=inp, state=result.state, niter=int(result.iterations), converged=True,
                                    vacuum_output=result.vacuum, **{k: forces[k] for k in ("fsqr", "fsqz", "fsql")})
        path = vmex.write_wout(str(out / "wout_dense.nc"), wout)
        rt = prepare_runtime(inp, resolution)
        residuals = target_residual().residuals_state(result.state, rt)
        report.update(qa=float(np.vdot(residuals, residuals)), min_abs_iota=float(min_abs_iota(result.state, rt)),
                      iota_axis=float(abs(opt.axis_iota(result.state, rt))),
                      aspect=float(opt.aspect_ratio(result.state, rt)), major_radius_m=float(opt.major_radius(result.state, rt)),
                      max_abs_iota=float(max_abs_iota(result.state, rt)))
        if MIRROR_LIMIT:
            report["mirror_ratio"] = float(opt.mirror_ratio(result.state, rt))
        if bootstrap:  # the Redl mismatch, and for a DKX case also the DKX one the run held
            report["redl_mismatch"] = float(redl_profiles(inp)[1].total_state(result.state, rt))
            if BOOTSTRAP_MODEL == "dkx":
                try:
                    report["dkx_mismatch"] = float(dkx_mismatch(inp)(result.state, rt))
                except ImportError:
                    print("dkx is not installed: dense.json has no dkx_mismatch", flush=True)
        report["figures"] = [str(p) for p in vmex.plot_wout(path, out, name="dense").values()]
    else:
        print("dense solve did not converge; seed it closer with --seed-wout", flush=True)
    (out / "dense.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def trace(wout, out, extra):
    """``vmex <wout> --trace`` in a fresh process, so it can give JAX one CPU device per core."""
    command = [sys.executable, "-m", "vmex", str(wout), "--trace", "--outdir", str(out / "trace"), *shlex.split(extra)]
    subprocess.run(command, check=True, env=dict(os.environ, JAX_PLATFORMS="cpu"))


def main(argv=None):
    args = parse_args(argv)
    os.environ["JAX_ENABLE_X64"] = "1"
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu" if args.device == "gpu" else "cpu")
    run = args.run.resolve()
    out = run / "postprocess"
    out.mkdir(exist_ok=True)
    rows, keys = read_history(run)
    if args.match_flux and "flux_ratio" not in keys:
        raise SystemExit("--match-flux needs a fixed-arm run that logs flux_ratio")
    plot_history(rows, keys, out)
    frames = checkpoints(run)
    if frames:
        plot_evolution(frames, out, final_step=rows[-1]["step"])
    if args.poincare and frames:
        print(json.dumps(poincare(frames[-1], args.poincare, out)), flush=True)
    if not args.no_dense and frames:  # the final state, or the latest checkpoint of a running job
        report = dense_solve(frames[-1], args, rows, out)
        print(json.dumps(report), flush=True)
        if args.trace and report["converged"]:
            trace(out / "wout_dense.nc", out, args.trace_args)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
