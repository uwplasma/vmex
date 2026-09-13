#!/usr/bin/env python
"""Single-stage fixed-boundary plasma and ESSOS coil optimization.

The stated targets are constraints, not weighted penalties.  The
rotational-transform floor, the aspect-ratio limit and the coil normal-field
limit enter a Powell-Hestenes-Rockafellar augmented Lagrangian around SciPy's
L-BFGS-B; quasisymmetry and the coil regularization are the objective.  A
weighted penalty trades a missed target against a lower objective: the
previous version of this example, seeded at a mean iota of 0.08 with the floor
as a hinge, ended at min |iota| 0.07 against its 0.42 floor.

Every target is checked at the end on an independent, finer solve and surface
grid.  A run that misses one says which and exits with status 1; smoke mode
(``VMEX_EXAMPLES_CI=1``) caps the budget, reports, and exits with status 0.
``benchmarks/single_stage_profile.py`` records the measured cost and the final
values in ``benchmarks/single_stage_profile_m4.json``.

Use the commented ``Coils.from_simsopt`` line to replace the generated coils
with a SIMSOPT coil JSON while keeping the objective and derivative code.
Preview: this script needs ESSOS branch ``rj/vmex-optimization-interfaces``.
"""

from dataclasses import replace
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize

import vmex as vj
from vmex import optimize as opt

import jax
import jax.numpy as jnp

try:
    from essos.coils import Coils, CreateEquallySpacedCurves
    from essos.fields import BiotSavart
    from essos.objective_functions import loss_coil_separation, loss_coil_surface_distance
    from essos.surfaces import surfacerzfourier_from_boundary
except ImportError as error:
    raise ImportError(
        "This example needs ESSOS branch rj/vmex-optimization-interfaces "
        "(uwplasma/ESSOS#58)."
    ) from error

started = time.perf_counter()
nfp = 2  # number of field periods
MAKE_MOVIE = True  # set True for a compact GIF of accepted iterates
# Surface colors: None, "absB", "B.n/B", or a callable ``(x, objects) -> values``.
MOVIE_SURFACE_COLOR = "absB"

SURFACES = np.linspace(0.05, 1.0, 6)
MAX_MODE = 2
VARY_MAJOR_RADIUS = False  # set True to optimize RBC(0,0) instead of fixing it

# The seed is a rotating ellipse close to, but outside, the constraints.  Solved
# at ns=31, mpol=ntor=5 (measured, not assumed), minor radius and ellipse give
# mean iota / min |iota| / aspect of 0.077 / 0.076 / 10.2 for 0.1 and 0.02 (this
# example's former seed), 0.388 / 0.373 / 4.12 for 0.28 and 0.14, and
# 0.408 / 0.391 / 4.03 for the values below.  RBC(1,1) and ZBS(1,1) carry
# opposite signs; equal signs give a circle of varying radius and no transform.
SEED_MINOR_RADIUS = 0.29
SEED_ELLIPSE = 0.15

# Targets, checked after the optimization on an ns=101 solve and a 61x64 grid.
IOTA_FLOOR = 0.42  # minimum |iota| over the profile
ASPECT_LIMIT = 4.0  # maximum aspect ratio
NORMAL_FIELD_LIMIT = 0.01  # area-weighted RMS of B.n/|B| on the boundary
CURVATURE_LIMIT = 7.0
COIL_DISTANCE_LIMIT = 0.15
COIL_SURFACE_DISTANCE_LIMIT = 0.20
# The optimizer enforces slightly tighter values than the ones it is checked
# against, because the check uses a finer radial grid and surface grid.
IOTA_CONSTRAINT = 0.43
ASPECT_CONSTRAINT = 3.98
NORMAL_FIELD_CONSTRAINT = 0.008
CURVATURE_OBJECTIVE_LIMIT = 6.9

N_COILS = 3
COIL_ORDER = 5
COIL_MAJOR_RADIUS = 1.0
# Clearance, not taste: at aspect 4 with iota 0.43 from ellipse rotation alone
# the cross-section reaches about 0.44 m from the axis, so circular coils of
# radius 0.5 could not keep the 0.20 m coil-surface distance.
COIL_MINOR_RADIUS = 0.65
COIL_CURRENT = 2.7e5
N_SEGMENTS = 64
STELLSYM = True

LENGTH_TARGET = 4.1  # close to the seed circumference, 2*pi*0.65 = 4.08 m
LENGTH_WEIGHT = 1.0
CURVATURE_WEIGHT = 10.0
COIL_DISTANCE_WEIGHT = 1.0e3
COIL_SURFACE_DISTANCE_WEIGHT = 1.0e3

# Augmented Lagrangian: each stage is one bounded L-BFGS-B solve at fixed
# multipliers; the multipliers then move and the penalty grows only when the
# violation did not fall by at least a factor of four.
PENALTY_START = 10.0
PENALTY_GROWTH = 10.0
PENALTY_MAX = 1.0e6
CONSTRAINT_TOLERANCE = 1.0e-3  # relative violation of the tightened constraints
MAX_STAGES = 8
STAGE_MAXITER = 25
MAX_TRIALS = 300  # objective evaluations: one equilibrium solve and one adjoint each
PARAMETER_BOUND = 3.0
# Circular coils leave B.n/B near 11% RMS on this seed.  Fitting the coils to
# the frozen seed boundary first costs no equilibrium solve, and it keeps the
# normal-field constraint from dominating the first joint stage.
COIL_FIT_MAXITER = 400

# A toroidal grid commensurate with the coil count can alias narrow B.n/B structure.
NPHI, NTHETA = 37, 32

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_STAGES, STAGE_MAXITER, MAX_TRIALS, COIL_FIT_MAXITER = 1, 1, 4, 2
    N_SEGMENTS, NPHI, NTHETA, COIL_ORDER = 24, 8, 8, 2

DATA = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{nfp}"
inp = vj.VmecInput.from_file(DATA)
# VmecInput is frozen, so copy its arrays before shaping the seed.
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor, 1] = zbs[inp.ntor, 1] = SEED_MINOR_RADIUS
rbc[inp.ntor + 1, 1], zbs[inp.ntor + 1, 1] = SEED_ELLIPSE, -SEED_ELLIPSE
inp = replace(inp, rbc=rbc, zbs=zbs)
mpol = max(MAX_MODE + 2, 5)
inp = replace(inp, delt=0.5).change_resolution(
    mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)

qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
plasma_problem = opt.VmecProblem.from_tuples(
    inp, [(qs.residuals_state, 0.0, 1.0)], max_mode=MAX_MODE,
    vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True, progress=not ci_smoke)


def plasma_constraints(equilibrium_state, solver_context):
    """Transform floor and aspect limit as ``c >= 0``, normalized by their limits.

    Floor the profile minimum, not its average: a mean target is satisfiable
    while an interior surface sits near zero transform.  Normalizing both by
    their limits lets one penalty parameter serve every constraint.
    """
    return jnp.stack([
        opt.min_abs_iota(equilibrium_state, solver_context) / IOTA_CONSTRAINT - 1.0,
        1.0 - opt.aspect_ratio(equilibrium_state, solver_context) / ASPECT_CONSTRAINT,
    ])


def augmented_lagrangian(constraints, multipliers, penalty):
    """Powell-Hestenes-Rockafellar term for constraints ``c >= 0``.

    It equals ``-multiplier * c + penalty * c**2 / 2`` while a constraint is
    violated or active, and is constant once the constraint holds with slack,
    so a satisfied target stops pulling on the design.  Its derivative with
    respect to the multipliers is ``(max(multiplier - penalty * c, 0) -
    multiplier) / penalty``, so the gradient already computed for SciPy also
    yields the next multipliers without another evaluation.
    """
    shifted = jnp.maximum(multipliers - penalty * constraints, 0.0)
    return jnp.sum(shifted**2 - multipliers**2) / (2.0 * penalty)


curves0 = CreateEquallySpacedCurves(
    N_COILS, COIL_ORDER, COIL_MAJOR_RADIUS, COIL_MINOR_RADIUS,
    n_segments=N_SEGMENTS, nfp=inp.nfp, stellsym=STELLSYM)
coils0 = Coils(curves0, np.full(N_COILS, COIL_CURRENT))
# To start from a SIMSOPT coil file instead, use:
# coils0 = Coils.from_simsopt("coils.json", nfp=inp.nfp, stellsym=STELLSYM)
# curves0 = coils0.curves

def normalized_normal_field(coils, surface):
    field = BiotSavart(coils)
    magnetic_field = jax.vmap(field.B)(surface.gamma.reshape(-1, 3)).reshape(surface.gamma.shape)
    return jnp.sum(magnetic_field * surface.unitnormal, axis=2) / jnp.linalg.norm(magnetic_field, axis=2)

def coil_field(coils):
    field = BiotSavart(coils)
    return lambda points: jax.vmap(field.B)(points.reshape(-1, 3)).reshape(points.shape)

def normal_field_rms(coils, surface):
    """Area-weighted RMS of B.n/|B| over the target boundary.

    A flux surface requires B.n = 0 on it, so this is the error the coils leave
    behind, normalized by |B| to make it dimensionless and by the area element
    so refining the grid does not change it.
    """
    weights = surface.area_element / jnp.sum(surface.area_element)
    return jnp.sqrt(jnp.sum(weights * normalized_normal_field(coils, surface)**2))

def coil_lengths(coils, _surface):
    return coils.length[:N_COILS]

def coil_curvature_excess(coils, _surface):
    return jnp.maximum(coils.curvature[:N_COILS] - CURVATURE_OBJECTIVE_LIMIT, 0.0)

coil_terms = [
    (coil_lengths, LENGTH_TARGET, LENGTH_WEIGHT),
    (coil_curvature_excess, 0.0, CURVATURE_WEIGHT),
]
coil_term_names = ("coil length", "coil curvature", "coil separation",
                   "coil-surface separation")

# The public VMEX problem owns the boundary-mode convention and RBC(0,0) choice.
x_boundary0 = plasma_problem.x0
rbc0, zbs0 = plasma_problem.boundary_from_x(x_boundary0)

n_curve_dofs = curves0.dofs.size
x_coils0 = np.asarray(curves0.dofs).ravel()
x0 = np.concatenate([x_boundary0, x_coils0])
dof_names = plasma_problem.dof_names + curves0.dof_names

# SciPy works in dimensionless increments u, with x = x0 + scales*u.
scales = np.concatenate([
    0.02 * plasma_problem.scales, np.full(n_curve_dofs, 0.05)])


def objects_from_x(x):
    x_boundary, x_coils = x[:x_boundary0.size], x[x_boundary0.size:]
    rbc, zbs = plasma_problem.boundary_from_x(x_boundary)
    surface = surfacerzfourier_from_boundary(
        rbc, zbs, inp.nfp, nphi=NPHI, ntheta=NTHETA)

    coils = coils0.with_dofs(jnp.concatenate((x_coils, coils0.dofs_currents)))
    return surface, coils


def plasma_lagrangian(u, multipliers, penalty):
    """Quasisymmetry plus the plasma constraint terms: one solve, one adjoint."""
    x = jnp.asarray(x0) + jnp.asarray(scales) * u
    return plasma_problem.jax_objective_from_state(
        x[:x_boundary0.size],
        lambda equilibrium_state, solver_context: augmented_lagrangian(
            plasma_constraints(equilibrium_state, solver_context),
            multipliers[:2], penalty),
        n_extra_terms=1)

def coil_lagrangian(u, multipliers, penalty):
    """Coil regularization plus the normal-field constraint term."""
    surface, coils = objects_from_x(jnp.asarray(x0) + jnp.asarray(scales) * u)
    rows = [jnp.sqrt(weight) * (jnp.atleast_1d(function(coils, surface)) - target).ravel()
            for function, target, weight in coil_terms]
    costs = jnp.concatenate([jnp.stack([0.5 * jnp.vdot(row, row) for row in rows]), jnp.asarray([
        0.5 * COIL_DISTANCE_WEIGHT * loss_coil_separation(
            coils, COIL_DISTANCE_LIMIT, block_size=32),
        0.5 * COIL_SURFACE_DISTANCE_WEIGHT * loss_coil_surface_distance(
            coils, surface, COIL_SURFACE_DISTANCE_LIMIT, block_size=32),
    ])])
    normal_field_constraint = 1.0 - normal_field_rms(coils, surface) / NORMAL_FIELD_CONSTRAINT
    constraint_cost = augmented_lagrangian(
        normal_field_constraint[None], multipliers[2:], penalty)
    return jnp.sum(costs) + constraint_cost, (costs, constraint_cost)


# The multipliers and penalty are arguments, not captured constants, so moving
# them between stages does not recompile either graph.  The equilibrium graph
# makes one host solve per trial; the coil graph is pure JAX.
plasma_value_and_grad = jax.jit(jax.value_and_grad(plasma_lagrangian, argnums=(0, 1), has_aux=True))
coil_value_and_grad = jax.jit(jax.value_and_grad(coil_lagrangian, argnums=(0, 1), has_aux=True))

monitor = opt.OptimizationMonitor(plasma_problem)
multipliers = np.zeros(3)  # iota floor, aspect limit, normal-field limit
penalty = PENALTY_START
counts = {"trials": 0}
last_evaluation = {}
next_multipliers = {}

# VMEX supplies the exact equilibrium derivative; JAX differentiates the coil
# objective. Their values and gradients add directly for any SciPy optimizer.
def value_and_grad(u):
    u = np.asarray(u, dtype=float)
    key = (u.tobytes(), multipliers.tobytes(), float(penalty))
    if last_evaluation.get("key") == key:
        return last_evaluation["value"], last_evaluation["gradient"].copy()
    counts["trials"] += 1
    arguments = (jnp.asarray(u), jnp.asarray(multipliers), jnp.asarray(penalty))
    (plasma_value, (qs_rows, plasma_constraint_cost)), (plasma_gradient, plasma_dmultipliers) = \
        plasma_value_and_grad(*arguments)
    (coil_value, (coil_cost_values, coil_constraint_cost)), (coil_gradient, coil_dmultipliers) = \
        coil_value_and_grad(*arguments)
    qs_rows = np.asarray(qs_rows)
    terms = {"quasisymmetry": 0.5 * float(qs_rows @ qs_rows),
             "iota and aspect constraints": float(np.asarray(plasma_constraint_cost)[0]),
             "normal-field constraint": float(coil_constraint_cost)}
    terms.update(zip(coil_term_names, map(float, np.asarray(coil_cost_values))))
    # max(multiplier - penalty * c, 0), read from the multiplier gradient.
    next_multipliers[u.tobytes()] = np.maximum(
        multipliers + penalty * (np.asarray(plasma_dmultipliers) + np.asarray(coil_dmultipliers)), 0.0)
    value, gradient = monitor.cache_evaluation(
        u, plasma_value + coil_value, plasma_gradient + coil_gradient, terms)
    last_evaluation.update(key=key, value=value, gradient=gradient)
    return value, gradient.copy()


print("Running single_stage_optimization.py")
print(f"Fixed-boundary VMEX + ESSOS: {x_boundary0.size} boundary and "
      f"{x_coils0.size} coil variables, exact reverse-mode derivatives")
print(f"dof_names = {dof_names}")
report = opt.EquilibriumReporter(
    ("QA total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"), ("min |iota|", opt.min_abs_iota, ".4f"))
seed_values = report("seed", plasma_problem.equilibrium_from_x(x_boundary0))

# Stage 0: coils only, boundary pinned by equal bounds.  With the normal-field
# multiplier equal to the penalty, its term is penalty/2 * (RMS/limit)**2 minus a
# constant: a plain least-squares fit with no slack at the limit.
fit_multipliers = jnp.asarray([0.0, 0.0, PENALTY_START])

def coil_fit_value_and_grad(u):
    (value, _), (gradient, _) = coil_value_and_grad(
        jnp.asarray(u), fit_multipliers, jnp.asarray(PENALTY_START))
    return float(value), np.asarray(gradient)

coil_fit = minimize(
    coil_fit_value_and_grad, np.zeros_like(x0), jac=True, method="L-BFGS-B",
    bounds=[(0.0, 0.0)] * x_boundary0.size + [(-PARAMETER_BOUND, PARAMETER_BOUND)] * n_curve_dofs,
    options={"maxiter": COIL_FIT_MAXITER, "maxcor": 20, "ftol": 1.0e-15, "gtol": 1.0e-10})
surface_seed, coils_fit = objects_from_x(jnp.asarray(x0 + scales * coil_fit.x))
print(f"[coil fit] {coil_fit.nit} L-BFGS-B iterations, no equilibrium solves: "
      f"B.n/B RMS = {100 * float(normal_field_rms(coils_fit, surface_seed)):.3f}% on the seed boundary")

u = coil_fit.x
vj.FunctionProblem.from_functions(u, value_and_grad=value_and_grad).compile_value_and_gradient(
    report_interval=10.0)
initial_value = monitor.records[0].cost
bounds = [(-PARAMETER_BOUND, PARAMETER_BOUND)] * x0.size
previous_violation, stages, iterations, final_value = np.inf, 0, 0, initial_value
for stage in range(MAX_STAGES):
    if counts["trials"] >= MAX_TRIALS:
        break
    result = minimize(value_and_grad, u, jac=True, method="L-BFGS-B", bounds=bounds,
                      callback=monitor, options={
                          "maxiter": STAGE_MAXITER, "maxfun": MAX_TRIALS - counts["trials"],
                          "maxcor": 20, "maxls": 20, "ftol": 1.0e-12, "gtol": 1.0e-8})
    u, final_value = result.x, float(result.fun)
    stages, iterations = stages + 1, iterations + int(result.nit)
    if u.tobytes() not in next_multipliers:
        value_and_grad(u)
    updated = next_multipliers[u.tobytes()]
    # max |min(c, multiplier/penalty)|: zero exactly when every constraint
    # holds and each multiplier is zero wherever its constraint has slack.
    violation = float(np.max(np.abs(updated - multipliers))) / penalty
    print(f"[stage {stage + 1}] {result.nit} L-BFGS-B iterations, {counts['trials']} trials, "
          f"violation = {violation:.3e}, multipliers = {np.array2string(updated, precision=4)}, "
          f"penalty = {penalty:.1e}", flush=True)
    multipliers = updated
    if violation <= CONSTRAINT_TOLERANCE and result.status != 1:
        break
    if violation > 0.25 * previous_violation:
        penalty = min(PENALTY_GROWTH * penalty, PENALTY_MAX)
    previous_violation = violation
optimization_seconds = time.perf_counter() - started

x_final = x0 + scales * u
_, coils_final = objects_from_x(jnp.asarray(x_final))
equilibrium = plasma_problem.equilibrium_from_x(x_final[:x_boundary0.size])
final_input = plasma_problem.input_from_x(x_final[:x_boundary0.size])
# FTOL 1e-12 rather than tighter: 1e-13 is out of reach even at ns=50 on the
# shipped QA deck, and a verification solve that cannot converge verifies nothing.
final_input = replace(final_input,
    ns_array=np.array([31 if ci_smoke else 101]),
    ftol_array=np.array([1.0e-10 if ci_smoke else 1.0e-12]),
    niter_array=np.array([8000]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution, verbose=not ci_smoke)

surface_final = surfacerzfourier_from_boundary(
    jnp.asarray(final_input.rbc), jnp.asarray(final_input.zbs), inp.nfp,
    nphi=61, ntheta=64)
normal_field = np.asarray(normalized_normal_field(coils_final, surface_final))
area_weights = np.asarray(surface_final.area_element); area_weights = area_weights / area_weights.sum()
normal_field_rms_final = float(np.sqrt(np.sum(area_weights * normal_field**2)))
normal_field_max = float(np.max(np.abs(normal_field)))
coil_points, surface_points = np.asarray(coils_final.gamma), np.asarray(surface_final.gamma).reshape(-1, 3)
coil_surface_distance = min(float(np.linalg.norm(points[:, None] - surface_points[None], axis=2).min())
                            for points in coil_points)
coil_pairs = [(i, j) for i in range(len(coil_points)) for j in range(i + 1, len(coil_points))]
coil_distance = min(float(np.linalg.norm(coil_points[i][:, None] - coil_points[j][None], axis=2).min())
                    for i, j in coil_pairs)
maximum_curvature = float(np.max(np.asarray(coils_final.curvature)))
minimum_iota = float(opt.min_abs_iota(final_equilibrium.state, final_equilibrium.runtime))
final_aspect = float(opt.aspect_ratio(final_equilibrium.state, final_equilibrium.runtime))
final_converged = bool(np.all(np.asarray(final_equilibrium.result.converged)))

# Print results
final_values = report("final", final_equilibrium)
print(f"\nObjective: {initial_value:.6e} -> {final_value:.6e} after {stages} augmented-Lagrangian "
      f"stages, {iterations} L-BFGS-B iterations and {counts['trials']} trials")
print(f"Coil lengths = {np.asarray(coils_final.length[:N_COILS])}")
print(f"B.n/B: area-weighted RMS = {100 * normal_field_rms_final:.3f}%, max = {100 * normal_field_max:.3f}% "
      f"(target RMS <= {100 * NORMAL_FIELD_LIMIT:.1f}%)")
print(f"Minimum coil-surface distance = {coil_surface_distance:.4f} m "
      f"(target >= {COIL_SURFACE_DISTANCE_LIMIT:.4f} m)")
print(f"Minimum coil-coil distance = {coil_distance:.4f} m (target >= {COIL_DISTANCE_LIMIT:.4f} m)")
print(f"Maximum curvature = {maximum_curvature:.4f} 1/m (target <= {CURVATURE_LIMIT:.4f} 1/m)")
print(f"Minimum |iota| = {minimum_iota:.4f} (target >= {IOTA_FLOOR:.4f})")
print(f"Aspect ratio = {final_aspect:.4f} (target <= {ASPECT_LIMIT:.4f})")

# A boundary with no transform makes the quasisymmetry residual trivially
# small, so a lower objective proves nothing: compare every target explicitly.
checks = (
    ("minimum |iota|", minimum_iota, IOTA_FLOOR, "below"),
    ("aspect ratio", final_aspect, ASPECT_LIMIT, "above"),
    ("B.n/B RMS", normal_field_rms_final, NORMAL_FIELD_LIMIT, "above"),
    ("minimum coil-surface distance", coil_surface_distance, COIL_SURFACE_DISTANCE_LIMIT, "below"),
    ("minimum coil-coil distance", coil_distance, COIL_DISTANCE_LIMIT, "below"),
    ("maximum curvature", maximum_curvature, CURVATURE_LIMIT, "above"),
)
unmet = [f"{name} {value:.4g} {side} the {limit:.4g} limit" for name, value, limit, side in checks
         if (value < limit if side == "below" else value > limit)]
if not final_converged:
    unmet.append("the ns=101 verification solve did not converge")
if unmet:
    print("\nThis run did NOT meet its stated targets: " + "; ".join(unmet) + ".")
    if ci_smoke:
        print("Smoke mode caps the budget far below what the targets need; exit status 0.")
else:
    print("\nAll stated targets met.")

summary = {
    "example": "single_stage_optimization.py", "smoke": ci_smoke,
    "optimization_seconds": round(optimization_seconds, 1),
    "coil_fit_iterations": int(coil_fit.nit),
    "trials": counts["trials"], "stages": stages, "lbfgsb_iterations": iterations,
    "equilibrium_solves": monitor.records[-1].equilibrium_solves,
    "seed": seed_values, "final": {
        **final_values, "min |iota|": minimum_iota, "aspect": final_aspect,
        "B.n/B RMS": normal_field_rms_final, "B.n/B max": normal_field_max,
        "coil-surface distance": coil_surface_distance, "coil-coil distance": coil_distance,
        "maximum curvature": maximum_curvature, "verification solve converged": final_converged},
    "targets": {"min |iota| >=": IOTA_FLOOR, "aspect <=": ASPECT_LIMIT,
                "B.n/B RMS <=": NORMAL_FIELD_LIMIT,
                "coil-surface distance >=": COIL_SURFACE_DISTANCE_LIMIT,
                "coil-coil distance >=": COIL_DISTANCE_LIMIT, "maximum curvature <=": CURVATURE_LIMIT},
    "unmet": unmet, "met": not unmet,
}
Path("single_stage_optimization_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

# Save results
input_path = final_input.to_indata("input.single_stage_optimized")
wout_path = vj.write_wout("wout_single_stage_optimized.nc", final_equilibrium.wout)
coils_final.to_json("coils_single_stage_optimized.json")
# ESSOS writes |B| and B.n/B on the surface and the coil filaments for ParaView.
surface_initial = surfacerzfourier_from_boundary(
    rbc0, zbs0, inp.nfp, nphi=60, ntheta=60)
surface_initial.to_vtk("surface_single_stage_initial", field=BiotSavart(coils0))
coils0.to_vtk("coils_single_stage_initial")
field_final = BiotSavart(coils_final)
surface_final.to_vtk("surface_single_stage_optimized", field=field_final)
coils_final.to_vtk("coils_single_stage_optimized")
print(f"Wrote {input_path}\nWrote {wout_path}")
print("Wrote coils_single_stage_optimized.json and single_stage_optimization_summary.json")
print("Wrote initial and optimized surface/coils VTK files")

# Plot results
print("Plotting results...")
vj.plot_optimization_objects("single_stage_optimization.png",
    ("Initial", surface_initial, coils0), ("Optimized", surface_final, coils_final))
monitor.save("single_stage_objectives.csv")
monitor.plot("single_stage_objectives.png", title="Single-stage objective terms")
print("Wrote single_stage_optimization.png")
print("Wrote single_stage_objectives.csv and single_stage_objectives.png")
if MAKE_MOVIE:
    print("Making movie of accepted iterates...")
    monitor.movie_surface_coils("single_stage_optimization.gif", objects_from_x,
        x0=x0, scales=scales, surface_color=MOVIE_SURFACE_COLOR, plasma_problem=plasma_problem,
        external_field=lambda objects: coil_field(objects[1]), nphi=NPHI, ntheta=NTHETA, cmap="jet")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
if unmet and not ci_smoke:
    raise SystemExit(1)
