#!/usr/bin/env python
r"""Single-stage fixed-boundary plasma and coil optimization with VMEX + ESSOS.

The plasma boundary and the coils are optimized together, as one vector of
Fourier coefficients. VMEX solves the fixed-boundary equilibrium on every trial
and returns the exact derivative of the converged state through the implicit
function theorem; ESSOS supplies the coils and their Biot-Savart field, which
JAX differentiates. The two gradients add, so any SciPy optimizer can drive it.

The objective is

    J = (1/2) |r_QS|^2                          quasisymmetry
      + (1/2) LENGTH_WEIGHT     |max(L - L_target, 0)|^2
      + (1/2) CURVATURE_WEIGHT  |max(kappa - kappa_max, 0)|^2
      + COIL_DISTANCE_WEIGHT         * separation penalty
      + COIL_SURFACE_DISTANCE_WEIGHT * clearance penalty
      + (1/2) CONSTRAINT_WEIGHT * sum max(-c, 0)^2   for c >= 0

where the last line holds the transform floor, the aspect limit and the coil
normal-field limit. Each is a one-sided quadratic penalty: it pulls only while
the target is missed. A penalty settles just INSIDE whatever threshold it is
given, because its pull vanishes with the violation, so the optimizer is handed
tightened values and the check at the end uses the real limits.

Two companions solve the same problem differently:
``single_stage_optimization_augmented_lagrangian.py`` puts the three limits in an augmented
Lagrangian, and ``single_stage_optimization_least_squares.py`` keeps the terms
as a residual vector for Gauss-Newton. Each is self-contained; read whichever
you intend to modify.

Run it with ``VMEX_EXAMPLES_CI=1`` for a short smoke pass that reports and
exits 0. Otherwise a run that misses a target says which and exits 1.
"""

import json
import os
from dataclasses import replace
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

import vmex as vj
from vmex import optimize as opt

from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.objective_functions import loss_coil_separation, loss_coil_surface_distance
from essos.surfaces import surfacerzfourier_from_boundary

# Number of field periods, and the seed deck the boundary is shaped from:
NFP = 2
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"

# Highest boundary Fourier mode number that is varied:
MAX_MODE = 2

# Flux surfaces the quasisymmetry residual is evaluated on:
SURFACES = np.linspace(0.05, 1.0, 6)

# Seed boundary: a rotating ellipse of aspect ratio about 5.97. RBC(1,1) and
# ZBS(1,1) carry opposite signs; equal signs give a circle and no transform.
SEED_MINOR_RADIUS = 0.195
SEED_ELLIPSE = 0.10

# Targets the finished design is checked against:
IOTA_FLOOR = 0.42                 # minimum |iota| over the profile
ASPECT_LIMIT = 6.0                # maximum aspect ratio
NORMAL_FIELD_LIMIT = 0.01         # area-weighted RMS of B.n/|B| on the boundary
COIL_SURFACE_DISTANCE_LIMIT = 0.15
COIL_DISTANCE_LIMIT = 0.17
CURVATURE_LIMIT = 7.0

# Values the optimizer is given instead. They are tighter than the limits above
# for two reasons: the check re-solves on a finer radial and surface grid, and a
# quadratic penalty settles just inside the threshold it is handed.
IOTA_CONSTRAINT = 0.43
ASPECT_CONSTRAINT = 5.97
NORMAL_FIELD_CONSTRAINT = 0.008
CURVATURE_OBJECTIVE_LIMIT = 6.9
COIL_DISTANCE_CONSTRAINT = 0.19
COIL_SURFACE_DISTANCE_CONSTRAINT = 0.16

# Coils: number of unique shapes, Fourier order, and the circle they start on.
# The aspect-6 seed reaches about 0.3 m from the circle R = 1, so radius 0.5
# starts the coils about 0.2 m out, above the 0.15 m clearance limit.
N_COILS = 3
COIL_ORDER = 4
COIL_MAJOR_RADIUS = 1.0
COIL_MINOR_RADIUS = 0.5
COIL_CURRENT = 2.7e5
N_SEGMENTS = 64
STELLSYM = True

# Weights, and the longest coil the optimizer may build. The length term is
# one-sided: coils shorter than LENGTH_TARGET cost nothing, so it never pulls
# against quasisymmetry or the normal field.
LENGTH_TARGET = 5.5
LENGTH_WEIGHT = 0.5
CURVATURE_WEIGHT = 10.0
COIL_DISTANCE_WEIGHT = 1.0e4
COIL_SURFACE_DISTANCE_WEIGHT = 1.0e4
CONSTRAINT_WEIGHT = 1.0e3

# Bound on each scaled variable. They keep a trial boundary inside the range
# the seed solve converges on; L-BFGS-B rather than BFGS because BFGS cannot
# represent a bound.
PARAMETER_BOUND = 3.0

# Budgets. One trial is one equilibrium solve plus one adjoint. The script's
# own end-of-run check passes at iterations 6, 8, 10, 12, 15 and 20 (measured),
# and 12 iterations took 21 trials. Past that point the run only trades a little
# quasisymmetry, so raise MAXITER to go further. The trial cap only stops a run
# whose line searches go astray.
MAXITER = 12
MAX_TRIALS = 30
COIL_FIT_MAXITER = 200            # coil-only pre-fit, no equilibrium solves

# Surface grid the coil objective uses. A toroidal count commensurate with the
# coil number aliases narrow B.n/B structure, so 37 rather than 36.
NPHI, NTHETA = 37, 32

# A compact GIF of the accepted iterates. Off by default: with "absB" colour
# every frame re-solves its equilibrium, which took longer than the whole
# 12-iteration optimization (measured).
MAKE_MOVIE = False
# Surface colour in that GIF: None, "absB", "B.n/B", or a callable
# ``(x, objects) -> values``.
MOVIE_SURFACE_COLOR = "absB"

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAXITER, MAX_TRIALS, COIL_FIT_MAXITER = 2, 4, 2
    N_SEGMENTS, COIL_ORDER, NPHI, NTHETA = 24, 2, 8, 8

###############################################################################
# End of input parameters.
###############################################################################

started = time.perf_counter()

###############################################################################
### Set up the equilibrium ####################################################
###############################################################################

# VmecInput is frozen, so copy its arrays before shaping the seed boundary.
inp = vj.VmecInput.from_file(INPUT_FILE)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor, 1] = zbs[inp.ntor, 1] = SEED_MINOR_RADIUS
rbc[inp.ntor + 1, 1], zbs[inp.ntor + 1, 1] = SEED_ELLIPSE, -SEED_ELLIPSE
mpol = max(MAX_MODE + 2, 5)
inp = replace(inp, rbc=rbc, zbs=zbs, delt=0.5).change_resolution(
    mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)

# A VmecProblem owns the boundary-mode convention, the dof scaling and the
# equilibrium solve. Quasisymmetry is its objective; the limits are added below.
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
plasma_problem = opt.VmecProblem.from_tuples(
    inp, [(qs.residuals_state, 0.0, 1.0)], max_mode=MAX_MODE,
    use_ess=True, progress=not ci_smoke)

###############################################################################
### Set up the coils ##########################################################
###############################################################################

curves0 = CreateEquallySpacedCurves(
    N_COILS, COIL_ORDER, COIL_MAJOR_RADIUS, COIL_MINOR_RADIUS,
    n_segments=N_SEGMENTS, nfp=NFP, stellsym=STELLSYM)
coils0 = Coils(curves0, np.full(N_COILS, COIL_CURRENT))
# To start from a SIMSOPT coil file instead:
# coils0 = Coils.from_simsopt("coils.json", nfp=NFP, stellsym=STELLSYM)
# curves0 = coils0.curves

# One variable vector holds both halves. SciPy works in dimensionless
# increments u, with x = x0 + scales * u, so every dof moves on a similar scale.
x_boundary0 = plasma_problem.x0
rbc0, zbs0 = plasma_problem.boundary_from_x(x_boundary0)
x_coils0 = np.asarray(curves0.dofs).ravel()
x0 = np.concatenate([x_boundary0, x_coils0])
scales = np.concatenate([0.02 * plasma_problem.scales, np.full(x_coils0.size, 0.05)])
n_boundary = x_boundary0.size


def objects_from_x(x):
    """The boundary surface and the coils implied by one variable vector."""
    rbc, zbs = plasma_problem.boundary_from_x(x[:n_boundary])
    surface = surfacerzfourier_from_boundary(rbc, zbs, NFP, nphi=NPHI, ntheta=NTHETA)
    coils = coils0.with_dofs(jnp.concatenate((x[n_boundary:], coils0.dofs_currents)))
    return surface, coils


###############################################################################
### Set up the objective ######################################################
###############################################################################

def normalized_normal_field(coils, surface):
    """B.n / |B| on the boundary: the error the coils leave behind."""
    field = BiotSavart(coils)
    B = jax.vmap(field.B)(surface.gamma.reshape(-1, 3)).reshape(surface.gamma.shape)
    return jnp.sum(B * surface.unitnormal, axis=2) / jnp.linalg.norm(B, axis=2)


def normal_field_rms(coils, surface):
    """Area-weighted RMS of B.n/|B|, so refining the grid does not move it."""
    weights = surface.area_element / jnp.sum(surface.area_element)
    return jnp.sqrt(jnp.sum(weights * normalized_normal_field(coils, surface)**2))


def coil_field(coils):
    """The coil Biot-Savart field as a batched callable, for tracing and plots."""
    field = BiotSavart(coils)
    return lambda points: jax.vmap(field.B)(points.reshape(-1, 3)).reshape(points.shape)


def hinge(constraints):
    """Quadratic penalty for constraints written as c >= 0, normalized by their limits."""
    return 0.5 * CONSTRAINT_WEIGHT * jnp.sum(jnp.maximum(-constraints, 0.0)**2)


def plasma_objective(u):
    """Quasisymmetry, the transform floor and the aspect limit.

    One equilibrium solve and one adjoint. Floor the profile minimum rather
    than its average: a mean target is satisfiable while an interior surface
    sits near zero transform.
    """
    x = jnp.asarray(x0) + jnp.asarray(scales) * u
    return plasma_problem.jax_objective_from_state(
        x[:n_boundary],
        lambda state, ctx: hinge(jnp.stack([
            opt.min_abs_iota(state, ctx) / IOTA_CONSTRAINT - 1.0,
            1.0 - opt.aspect_ratio(state, ctx) / ASPECT_CONSTRAINT])),
        n_extra_terms=1)


def coil_objective(u):
    """Coil regularization and the normal-field limit. Pure JAX, no solve."""
    surface, coils = objects_from_x(jnp.asarray(x0) + jnp.asarray(scales) * u)
    length = jnp.sqrt(LENGTH_WEIGHT) * jnp.maximum(coils.length[:N_COILS] - LENGTH_TARGET, 0.0)
    curvature = jnp.sqrt(CURVATURE_WEIGHT) * jnp.maximum(
        coils.curvature[:N_COILS] - CURVATURE_OBJECTIVE_LIMIT, 0.0)
    costs = jnp.asarray([
        0.5 * jnp.vdot(length, length),
        0.5 * jnp.vdot(curvature, curvature),
        0.5 * COIL_DISTANCE_WEIGHT * loss_coil_separation(
            coils, COIL_DISTANCE_CONSTRAINT, block_size=32),
        0.5 * COIL_SURFACE_DISTANCE_WEIGHT * loss_coil_surface_distance(
            coils, surface, COIL_SURFACE_DISTANCE_CONSTRAINT, block_size=32),
    ])
    normal_field = 1.0 - normal_field_rms(coils, surface) / NORMAL_FIELD_CONSTRAINT
    return jnp.sum(costs) + hinge(normal_field[None]), costs


plasma_value_and_grad = jax.jit(jax.value_and_grad(plasma_objective, has_aux=True))
coil_value_and_grad = jax.jit(jax.value_and_grad(coil_objective, has_aux=True))

# The monitor prints one row per evaluation and records the history for the
# plot and the movie at the end.
monitor = opt.OptimizationMonitor(plasma_problem)
counts, cached = {"trials": 0}, {}
COIL_TERMS = ("coil length", "coil curvature", "coil separation", "coil-surface separation")


def value_and_grad(u):
    """What SciPy calls: the two halves evaluated once and added."""
    u = np.asarray(u, dtype=float)
    if cached.get("key") == u.tobytes():          # SciPy re-asks at the same point
        return cached["value"], cached["gradient"].copy()
    counts["trials"] += 1
    (plasma_value, (qs_rows, plasma_penalty)), plasma_gradient = \
        plasma_value_and_grad(jnp.asarray(u))
    (coil_value, coil_costs), coil_gradient = coil_value_and_grad(jnp.asarray(u))
    qs_rows = np.asarray(qs_rows)
    terms = {"quasisymmetry": 0.5 * float(qs_rows @ qs_rows),
             "iota and aspect penalty": float(np.asarray(plasma_penalty)[0])}
    terms.update(zip(COIL_TERMS, map(float, np.asarray(coil_costs))))
    value, gradient = monitor.cache_evaluation(
        u, plasma_value + coil_value, plasma_gradient + coil_gradient, terms)
    cached.update(key=u.tobytes(), value=value, gradient=gradient)
    return value, gradient.copy()


###############################################################################
### Run the optimization ######################################################
###############################################################################

print("Running single_stage_optimization.py")
print(f"Fixed-boundary VMEX + ESSOS: {n_boundary} boundary and {x_coils0.size} coil "
      f"variables, exact reverse-mode derivatives")
report = opt.EquilibriumReporter(
    ("QA total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"), ("min |iota|", opt.min_abs_iota, ".4f"))
seed_values = report("seed", plasma_problem.equilibrium_from_x(x_boundary0))


def coil_fit_value_and_grad(u):
    (value, _), gradient = coil_value_and_grad(jnp.asarray(u))
    return float(value), np.asarray(gradient)


# Fit the coils to the frozen seed boundary first, with the boundary pinned by
# equal bounds. Circular coils leave B.n/B near 20% RMS; this costs no
# equilibrium solve and keeps the normal-field term from dominating the joint
# solve that follows.
coil_fit = minimize(
    coil_fit_value_and_grad, np.zeros_like(x0), jac=True, method="L-BFGS-B",
    bounds=[(0.0, 0.0)] * n_boundary
           + [(-PARAMETER_BOUND, PARAMETER_BOUND)] * x_coils0.size,
    options={"maxiter": COIL_FIT_MAXITER, "maxcor": 20, "ftol": 1e-15, "gtol": 1e-10})
surface_seed, coils_fit = objects_from_x(jnp.asarray(x0 + scales * coil_fit.x))
print(f"[coil fit] {coil_fit.nit} L-BFGS-B iterations, no equilibrium solves: B.n/B RMS "
      f"= {100 * float(normal_field_rms(coils_fit, surface_seed)):.3f}% on the seed")

u = coil_fit.x
vj.FunctionProblem.from_functions(u, value_and_grad=value_and_grad).compile_value_and_gradient(
    report_interval=10.0)
initial_value = monitor.records[0].cost
result = minimize(value_and_grad, u, jac=True, method="L-BFGS-B", callback=monitor,
                  bounds=[(-PARAMETER_BOUND, PARAMETER_BOUND)] * x0.size,
                  options={"maxiter": MAXITER, "maxfun": MAX_TRIALS, "maxcor": 20,
                           "maxls": 20, "ftol": 1e-12, "gtol": 1e-8})
u, final_value = result.x, float(result.fun)
print(f"[solve] {result.nit} L-BFGS-B iterations, {counts['trials']} trials, "
      f"status {result.status}: {result.message}", flush=True)
optimization_seconds = time.perf_counter() - started

###############################################################################
### Check the result against the targets ######################################
###############################################################################

# A lower objective proves nothing on its own: a boundary with no transform
# makes the quasisymmetry residual trivially small. Re-solve on a finer radial
# and surface grid and compare every target explicitly.
x_final = x0 + scales * u
_, coils_final = objects_from_x(jnp.asarray(x_final))
equilibrium = plasma_problem.equilibrium_from_x(x_final[:n_boundary])
# FTOL 1e-12 rather than tighter: 1e-13 is out of reach even at ns=50 here, and
# a verification solve that cannot converge verifies nothing.
final_input = replace(plasma_problem.input_from_x(x_final[:n_boundary]),
                      ns_array=np.array([31 if ci_smoke else 101]),
                      ftol_array=np.array([1.0e-10 if ci_smoke else 1.0e-12]),
                      niter_array=np.array([8000]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution, verbose=not ci_smoke)

surface_final = surfacerzfourier_from_boundary(
    jnp.asarray(final_input.rbc), jnp.asarray(final_input.zbs), NFP, nphi=61, ntheta=64)
normal_field = np.asarray(normalized_normal_field(coils_final, surface_final))
area = np.asarray(surface_final.area_element)
area = area / area.sum()
normal_field_rms_final = float(np.sqrt(np.sum(area * normal_field**2)))
normal_field_max = float(np.max(np.abs(normal_field)))
gamma = np.asarray(coils_final.gamma)
points = np.asarray(surface_final.gamma).reshape(-1, 3)
coil_surface_distance = min(
    float(np.linalg.norm(c[:, None] - points[None], axis=2).min()) for c in gamma)
coil_distance = min(
    float(np.linalg.norm(gamma[i][:, None] - gamma[j][None], axis=2).min())
    for i in range(len(gamma)) for j in range(i + 1, len(gamma)))
maximum_curvature = float(np.max(np.asarray(coils_final.curvature)))
minimum_iota = float(opt.min_abs_iota(final_equilibrium.state, final_equilibrium.runtime))
final_aspect = float(opt.aspect_ratio(final_equilibrium.state, final_equilibrium.runtime))
final_converged = bool(np.all(np.asarray(final_equilibrium.result.converged)))

###############################################################################
### Print, plot and save ######################################################
###############################################################################

final_values = report("final", final_equilibrium)
print(f"\nObjective: {initial_value:.6e} -> {final_value:.6e} after "
      f"{result.nit} L-BFGS-B iterations and {counts['trials']} trials")
print(f"Coil lengths = {np.asarray(coils_final.length[:N_COILS])}")
print(f"B.n/B: area-weighted RMS = {100 * normal_field_rms_final:.3f}%, "
      f"max = {100 * normal_field_max:.3f}% (target RMS <= "
      f"{100 * NORMAL_FIELD_LIMIT:.1f}%; the maximum is reported, not optimized)")
print(f"Minimum coil-surface distance = {coil_surface_distance:.4f} m "
      f"(target >= {COIL_SURFACE_DISTANCE_LIMIT:.4f} m)")
print(f"Minimum coil-coil distance = {coil_distance:.4f} m "
      f"(target >= {COIL_DISTANCE_LIMIT:.4f} m)")
print(f"Maximum curvature = {maximum_curvature:.4f} 1/m "
      f"(target <= {CURVATURE_LIMIT:.4f} 1/m)")
print(f"Minimum |iota| = {minimum_iota:.4f} (target >= {IOTA_FLOOR:.4f})")
print(f"Aspect ratio = {final_aspect:.4f} (target <= {ASPECT_LIMIT:.4f})")

checks = (
    ("minimum |iota|", minimum_iota, IOTA_FLOOR, "below"),
    ("aspect ratio", final_aspect, ASPECT_LIMIT, "above"),
    ("B.n/B RMS", normal_field_rms_final, NORMAL_FIELD_LIMIT, "above"),
    ("minimum coil-surface distance", coil_surface_distance, COIL_SURFACE_DISTANCE_LIMIT, "below"),
    ("minimum coil-coil distance", coil_distance, COIL_DISTANCE_LIMIT, "below"),
    ("maximum curvature", maximum_curvature, CURVATURE_LIMIT, "above"),
)
unmet = [f"{name} {value:.4g} {side} the {limit:.4g} limit"
         for name, value, limit, side in checks
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
    "coil_fit_iterations": int(coil_fit.nit), "trials": counts["trials"],
    "lbfgsb_iterations": int(result.nit),
    "equilibrium_solves": monitor.records[-1].equilibrium_solves,
    "seed": seed_values,
    "final": {**final_values, "min |iota|": minimum_iota, "aspect": final_aspect,
              "B.n/B RMS": normal_field_rms_final, "B.n/B max": normal_field_max,
              "coil-surface distance": coil_surface_distance,
              "coil-coil distance": coil_distance,
              "maximum curvature": maximum_curvature,
              "verification solve converged": final_converged},
    "targets": {"min |iota| >=": IOTA_FLOOR, "aspect <=": ASPECT_LIMIT,
                "B.n/B RMS <=": NORMAL_FIELD_LIMIT,
                "coil-surface distance >=": COIL_SURFACE_DISTANCE_LIMIT,
                "coil-coil distance >=": COIL_DISTANCE_LIMIT,
                "maximum curvature <=": CURVATURE_LIMIT},
    "unmet": unmet, "met": not unmet,
}
Path("single_stage_optimization_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

input_path = final_input.to_indata("input.single_stage_optimized")
wout_path = vj.write_wout("wout_single_stage_optimized.nc", final_equilibrium.wout)
coils_final.to_json("coils_single_stage_optimized.json")
# ESSOS writes |B| and B.n/B on the surface and the filaments for ParaView.
surface_initial = surfacerzfourier_from_boundary(rbc0, zbs0, NFP, nphi=60, ntheta=60)
surface_initial.to_vtk("surface_single_stage_initial", field=BiotSavart(coils0))
coils0.to_vtk("coils_single_stage_initial")
surface_final.to_vtk("surface_single_stage_optimized", field=BiotSavart(coils_final))
coils_final.to_vtk("coils_single_stage_optimized")
print(f"Wrote {input_path}\nWrote {wout_path}")
print("Wrote coils_single_stage_optimized.json and single_stage_optimization_summary.json")
print("Wrote initial and optimized surface/coils VTK files")

print("Plotting results...")
vj.plot_optimization_objects("single_stage_optimization.png",
                             ("Initial", surface_initial, coils0),
                             ("Optimized", surface_final, coils_final))
monitor.save("single_stage_objectives.csv")
monitor.plot("single_stage_objectives.png", title="Single-stage objective terms")
print("Wrote single_stage_optimization.png")
print("Wrote single_stage_objectives.csv and single_stage_objectives.png")
if MAKE_MOVIE:
    print("Making movie of accepted iterates...")
    monitor.movie_surface_coils(
        "single_stage_optimization.gif", objects_from_x, x0=x0, scales=scales,
        surface_color=MOVIE_SURFACE_COLOR, plasma_problem=plasma_problem,
        external_field=lambda objects: coil_field(objects[1]), nphi=NPHI, ntheta=NTHETA,
        cmap="jet")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
if unmet and not ci_smoke:
    raise SystemExit(1)
