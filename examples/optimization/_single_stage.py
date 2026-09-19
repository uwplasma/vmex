"""Shared setup and verification for the fixed-boundary single-stage examples.

Three scripts solve the same joint plasma-and-coil problem and differ only in
how the constraints enter: quadratic penalties, an augmented Lagrangian, or
least squares.  The seed, the coil model and the end-of-run verification are
identical in all three, so they live here and each script keeps only the part
worth reading -- its objective and its optimizer.

Nothing here is VMEX API; it is example plumbing.
"""

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

import vmex as vj
from vmex import optimize as opt

from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.surfaces import surfacerzfourier_from_boundary

DATA = Path(__file__).resolve().parents[1] / "data"

# Geometry shared by the three scripts.  Clearance, not taste: at aspect 4 with
# iota 0.43 the cross-section reaches about 0.44 m from the axis, so circular
# coils of radius 0.5 could not keep the 0.20 m coil-surface distance.
NFP = 2
N_COILS, COIL_ORDER, COIL_MAJOR_RADIUS, COIL_MINOR_RADIUS = 3, 5, 1.0, 0.65
COIL_CURRENT, N_SEGMENTS, STELLSYM = 2.7e5, 64, True
# The seed is a rotating ellipse just outside the constraints.  RBC(1,1) and
# ZBS(1,1) carry opposite signs; equal signs give a circle and no transform.
SEED_MINOR_RADIUS, SEED_ELLIPSE = 0.29, 0.15
MAX_MODE, SURFACES = 2, np.linspace(0.05, 1.0, 6)
NPHI, NTHETA = 37, 32  # a grid commensurate with the coil count aliases B.n/B
PARAMETER_BOUND = 3.0


@dataclass(frozen=True)
class Limits:
    """What the run is checked against, and the tighter values it optimizes to.

    The optimizer is given tighter values than the check uses because the check
    re-solves on a finer radial and surface grid.
    """

    iota_floor: float = 0.42
    aspect: float = 4.0
    normal_field: float = 0.01
    coil_surface: float = 0.20
    coil_coil: float = 0.17
    curvature: float = 7.0
    iota_constraint: float = 0.43
    aspect_constraint: float = 3.98
    normal_field_constraint: float = 0.008
    curvature_objective: float = 6.9


def seed_input(ci_smoke: bool):
    """The shaped minimal seed, at the resolution the scripts optimize on."""
    inp = vj.VmecInput.from_file(DATA / f"input.minimal_seed_nfp{NFP}")
    rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
    rbc[inp.ntor, 1] = zbs[inp.ntor, 1] = SEED_MINOR_RADIUS
    rbc[inp.ntor + 1, 1], zbs[inp.ntor + 1, 1] = SEED_ELLIPSE, -SEED_ELLIPSE
    mpol = max(MAX_MODE + 2, 5)
    return replace(inp, rbc=rbc, zbs=zbs, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)


def seed_coils(n_segments: int = N_SEGMENTS, coil_order: int = COIL_ORDER):
    """Equally spaced circular coils; swap in ``Coils.from_simsopt`` to reuse a file."""
    curves = CreateEquallySpacedCurves(
        N_COILS, coil_order, COIL_MAJOR_RADIUS, COIL_MINOR_RADIUS,
        n_segments=n_segments, nfp=NFP, stellsym=STELLSYM)
    return curves, Coils(curves, np.full(N_COILS, COIL_CURRENT))


def normalized_normal_field(coils, surface):
    """``B.n / |B|`` on the boundary: the error the coils leave behind."""
    field = BiotSavart(coils)
    B = jax.vmap(field.B)(surface.gamma.reshape(-1, 3)).reshape(surface.gamma.shape)
    return jnp.sum(B * surface.unitnormal, axis=2) / jnp.linalg.norm(B, axis=2)


def normal_field_rms(coils, surface):
    """Area-weighted RMS of ``B.n/|B|``, so refining the grid does not move it."""
    weights = surface.area_element / jnp.sum(surface.area_element)
    return jnp.sqrt(jnp.sum(weights * normalized_normal_field(coils, surface)**2))


def coil_field(coils):
    field = BiotSavart(coils)
    return lambda points: jax.vmap(field.B)(points.reshape(-1, 3)).reshape(points.shape)


@dataclass
class Case:
    """Everything the shared verification needs, which each script already has."""

    script: str  # script stem; names the summary, which benchmarks/ looks up
    stem: str  # artifact prefix, e.g. "single_stage" -> wout_single_stage_optimized.nc
    inp: Any
    plasma_problem: Any
    objects_from_x: Callable
    x0: np.ndarray
    scales: np.ndarray
    n_boundary: int
    coils0: Any
    rbc0: Any
    zbs0: Any
    limits: Limits
    ci_smoke: bool
    make_movie: bool = True
    movie_surface_color: str | None = "absB"


def finish(case: Case, *, u, monitor, report, headline: str, summary_extra: dict,
           seed_values: dict):
    """Re-solve finer, check every target, save, plot, and exit 1 on a miss.

    A lower objective proves nothing on its own -- a boundary with no transform
    makes the quasisymmetry residual trivially small -- so every target is
    compared explicitly against an independent solve.
    """
    limits, smoke = case.limits, case.ci_smoke
    x_final = case.x0 + case.scales * u
    _, coils = case.objects_from_x(jnp.asarray(x_final))
    equilibrium = case.plasma_problem.equilibrium_from_x(x_final[:case.n_boundary])
    # FTOL 1e-12 rather than tighter: 1e-13 is out of reach even at ns=50 here,
    # and a verification solve that cannot converge verifies nothing.
    final_input = replace(
        case.plasma_problem.input_from_x(x_final[:case.n_boundary]),
        ns_array=np.array([31 if smoke else 101]),
        ftol_array=np.array([1.0e-10 if smoke else 1.0e-12]),
        niter_array=np.array([8000]))
    final_equilibrium = opt.solve_equilibrium(
        final_input, initial_state=equilibrium.solution, verbose=not smoke)

    surface = surfacerzfourier_from_boundary(
        jnp.asarray(final_input.rbc), jnp.asarray(final_input.zbs), case.inp.nfp,
        nphi=61, ntheta=64)
    normal = np.asarray(normalized_normal_field(coils, surface))
    weights = np.asarray(surface.area_element)
    weights = weights / weights.sum()
    measured = {
        "B.n/B RMS": float(np.sqrt(np.sum(weights * normal**2))),
        "B.n/B max": float(np.max(np.abs(normal))),
        "maximum curvature": float(np.max(np.asarray(coils.curvature))),
        "min |iota|": float(opt.min_abs_iota(
            final_equilibrium.state, final_equilibrium.runtime)),
        "aspect": float(opt.aspect_ratio(
            final_equilibrium.state, final_equilibrium.runtime)),
    }
    gamma = np.asarray(coils.gamma)
    points = np.asarray(surface.gamma).reshape(-1, 3)
    measured["coil-surface distance"] = min(
        float(np.linalg.norm(c[:, None] - points[None], axis=2).min()) for c in gamma)
    measured["coil-coil distance"] = min(
        float(np.linalg.norm(gamma[i][:, None] - gamma[j][None], axis=2).min())
        for i in range(len(gamma)) for j in range(i + 1, len(gamma)))
    converged = bool(np.all(np.asarray(final_equilibrium.result.converged)))

    final_values = report("final", final_equilibrium)
    print(f"\n{headline}")
    print(f"Coil lengths = {np.asarray(coils.length[:N_COILS])}")
    print(f"B.n/B: area-weighted RMS = {100 * measured['B.n/B RMS']:.3f}%, "
          f"max = {100 * measured['B.n/B max']:.3f}% (target RMS <= "
          f"{100 * limits.normal_field:.1f}%; the maximum is reported, not optimized)")
    print(f"Minimum coil-surface distance = {measured['coil-surface distance']:.4f} m "
          f"(target >= {limits.coil_surface:.4f} m)")
    print(f"Minimum coil-coil distance = {measured['coil-coil distance']:.4f} m "
          f"(target >= {limits.coil_coil:.4f} m)")
    print(f"Maximum curvature = {measured['maximum curvature']:.4f} 1/m "
          f"(target <= {limits.curvature:.4f} 1/m)")
    print(f"Minimum |iota| = {measured['min |iota|']:.4f} "
          f"(target >= {limits.iota_floor:.4f})")
    print(f"Aspect ratio = {measured['aspect']:.4f} (target <= {limits.aspect:.4f})")

    checks = (
        ("minimum |iota|", "min |iota|", limits.iota_floor, "below"),
        ("aspect ratio", "aspect", limits.aspect, "above"),
        ("B.n/B RMS", "B.n/B RMS", limits.normal_field, "above"),
        ("minimum coil-surface distance", "coil-surface distance", limits.coil_surface, "below"),
        ("minimum coil-coil distance", "coil-coil distance", limits.coil_coil, "below"),
        ("maximum curvature", "maximum curvature", limits.curvature, "above"),
    )
    unmet = [f"{label} {measured[key]:.4g} {side} the {limit:.4g} limit"
             for label, key, limit, side in checks
             if (measured[key] < limit if side == "below" else measured[key] > limit)]
    if not converged:
        unmet.append("the ns=101 verification solve did not converge")
    if unmet:
        print("\nThis run did NOT meet its stated targets: " + "; ".join(unmet) + ".")
        if smoke:
            print("Smoke mode caps the budget far below what the targets need; "
                  "exit status 0.")
    else:
        print("\nAll stated targets met.")

    stem = case.stem
    summary = {
        "example": f"{case.script}.py", "smoke": smoke, **summary_extra,
        "equilibrium_solves": monitor.records[-1].equilibrium_solves,
        "seed": seed_values,
        "final": {**final_values, **measured, "verification solve converged": converged},
        "targets": {"min |iota| >=": limits.iota_floor, "aspect <=": limits.aspect,
                    "B.n/B RMS <=": limits.normal_field,
                    "coil-surface distance >=": limits.coil_surface,
                    "coil-coil distance >=": limits.coil_coil,
                    "maximum curvature <=": limits.curvature},
        "unmet": unmet, "met": not unmet,
    }
    Path(f"{case.script}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    input_path = final_input.to_indata(f"input.{stem}_optimized")
    wout_path = vj.write_wout(f"wout_{stem}_optimized.nc", final_equilibrium.wout)
    coils.to_json(f"coils_{stem}_optimized.json")
    # ESSOS writes |B| and B.n/B on the surface and the filaments for ParaView.
    surface_initial = surfacerzfourier_from_boundary(
        case.rbc0, case.zbs0, case.inp.nfp, nphi=60, ntheta=60)
    surface_initial.to_vtk(f"surface_{stem}_initial", field=BiotSavart(case.coils0))
    case.coils0.to_vtk(f"coils_{stem}_initial")
    surface.to_vtk(f"surface_{stem}_optimized", field=BiotSavart(coils))
    coils.to_vtk(f"coils_{stem}_optimized")
    print(f"Wrote {input_path}\nWrote {wout_path}")
    print(f"Wrote coils_{stem}_optimized.json and {case.script}_summary.json")
    print("Wrote initial and optimized surface/coils VTK files")

    print("Plotting results...")
    vj.plot_optimization_objects(f"{stem}.png", ("Initial", surface_initial, case.coils0),
                                 ("Optimized", surface, coils))
    monitor.save(f"{stem}_objectives.csv")
    monitor.plot(f"{stem}_objectives.png", title=f"{stem} objective terms")
    print(f"Wrote {stem}.png")
    print(f"Wrote {stem}_objectives.csv and {stem}_objectives.png")
    if case.make_movie:
        print("Making movie of accepted iterates...")
        monitor.movie_surface_coils(
            f"{stem}.gif", case.objects_from_x, x0=case.x0, scales=case.scales,
            surface_color=case.movie_surface_color,
            plasma_problem=case.plasma_problem,
            external_field=lambda objects: coil_field(objects[1]),
            nphi=NPHI, ntheta=NTHETA, cmap="jet")
    for path in vj.plot_wout(wout_path, ".").values():
        print(f"Wrote {path}")
    if unmet and not smoke:
        raise SystemExit(1)
