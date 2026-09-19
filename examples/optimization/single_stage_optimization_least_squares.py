#!/usr/bin/env python
"""Single-stage fixed-boundary optimization as one joint least-squares problem.

``single_stage_optimization.py`` sums its terms into a scalar and hands SciPy a
gradient.  This one keeps them as a residual vector and hands SciPy a Jacobian,
so a Gauss-Newton step uses the curvature that ``0.5 * r.T @ r`` implies instead
of rebuilding it from gradients.

The joint Jacobian is cheaper than it looks.  The plasma residuals depend only
on the boundary dofs and the coil residuals are pure JAX, so the block coupling
plasma rows to coil dofs is identically zero and only the boundary columns need
the implicit equilibrium Jacobian:

    r = [r_plasma(b)]      J = [J_plasma   0      ]
        [r_coil(b, c)]         [dr_c/db    dr_c/dc]

Constraint handling is the same as the penalized file, because least squares
minimizes a sum of squares and a one-sided residual squared IS a quadratic
penalty.  So this file inherits that behaviour -- it needs the same tightened
thresholds -- and differs only in the optimizer.  What changes is the step:
Gauss-Newton curvature usually buys progress per evaluation, at the cost of a
Jacobian (one implicit solve per boundary dof) where the scalar lane pays one
adjoint.  ``examples/README.md`` records which wins here.
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares, minimize

from vmex import optimize as opt

from essos.objective_functions import loss_coil_separation, loss_coil_surface_distance
from essos.surfaces import surfacerzfourier_from_boundary

import _single_stage as ss

started = time.perf_counter()
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
limits = ss.Limits()

LENGTH_TARGET, LENGTH_WEIGHT = 5.3, 1.0
CURVATURE_WEIGHT = 10.0
COIL_DISTANCE_CONSTRAINT, COIL_SURFACE_DISTANCE_CONSTRAINT = 0.19, 0.21
COIL_DISTANCE_WEIGHT = COIL_SURFACE_DISTANCE_WEIGHT = 1.0e4
CONSTRAINT_WEIGHT = 1.0e3
MAX_TRIALS = 300
COIL_FIT_MAXITER = 200
coil_order, n_segments = ss.COIL_ORDER, ss.N_SEGMENTS
if ci_smoke:
    MAX_TRIALS, COIL_FIT_MAXITER = 4, 2
    coil_order, n_segments = 2, 24

inp = ss.seed_input(ci_smoke)
curves0, coils0 = ss.seed_coils(n_segments, coil_order)
qs = opt.QuasisymmetryRatioResidual(ss.SURFACES, helicity_m=1, helicity_n=0)


def iota_shortfall(state, ctx):
    """How far the transform floor is missed, and zero once it is met."""
    return jnp.maximum(limits.iota_constraint - opt.min_abs_iota(state, ctx), 0.0)


def aspect_excess(state, ctx):
    return jnp.maximum(opt.aspect_ratio(state, ctx) - limits.aspect_constraint, 0.0)


# One-sided rows, so a satisfied target contributes nothing rather than pulling
# the design back to the limit the way an equality residual would.
plasma_problem = opt.VmecProblem.from_tuples(
    inp,
    [(qs.residuals_state, 0.0, 1.0),
     (iota_shortfall, 0.0, CONSTRAINT_WEIGHT),
     (aspect_excess, 0.0, CONSTRAINT_WEIGHT)],
    max_mode=ss.MAX_MODE, use_ess=True, progress=not ci_smoke)

x_boundary0 = plasma_problem.x0
rbc0, zbs0 = plasma_problem.boundary_from_x(x_boundary0)
x_coils0 = np.asarray(curves0.dofs).ravel()
x0 = np.concatenate([x_boundary0, x_coils0])
scales = np.concatenate([0.02 * plasma_problem.scales, np.full(x_coils0.size, 0.05)])
n_boundary = x_boundary0.size


def objects_from_x(x):
    rbc, zbs = plasma_problem.boundary_from_x(x[:n_boundary])
    surface = surfacerzfourier_from_boundary(
        rbc, zbs, inp.nfp, nphi=ss.NPHI, ntheta=ss.NTHETA)
    coils = coils0.with_dofs(jnp.concatenate((x[n_boundary:], coils0.dofs_currents)))
    return surface, coils


def safe_sqrt(x):
    """``sqrt`` with a zero derivative at zero instead of an infinite one.

    The two clearance terms are aggregate hinge losses, so a row here is their
    norm and is exactly zero once the clearance is met -- which is where the
    plain derivative of ``sqrt`` is infinite and the first Jacobian would come
    back NaN.  The doubled ``where`` keeps the bad branch out of both the value
    and the derivative.
    """
    positive = x > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, x, 1.0)), 0.0)


def coil_residuals(u):
    """Coil rows: length, curvature, the two clearances, and the normal field."""
    surface, coils = objects_from_x(jnp.asarray(x0) + jnp.asarray(scales) * u)
    normal_field = ss.normal_field_rms(coils, surface) / limits.normal_field_constraint
    return jnp.concatenate([
        jnp.sqrt(LENGTH_WEIGHT) * (coils.length[:ss.N_COILS] - LENGTH_TARGET),
        # curvature is per segment, so this block is N_COILS * n_segments rows
        jnp.sqrt(CURVATURE_WEIGHT) * jnp.maximum(
            coils.curvature[:ss.N_COILS] - limits.curvature_objective, 0.0).ravel(),
        safe_sqrt(COIL_DISTANCE_WEIGHT * loss_coil_separation(
            coils, COIL_DISTANCE_CONSTRAINT, block_size=32))[None],
        safe_sqrt(COIL_SURFACE_DISTANCE_WEIGHT * loss_coil_surface_distance(
            coils, surface, COIL_SURFACE_DISTANCE_CONSTRAINT, block_size=32))[None],
        jnp.sqrt(CONSTRAINT_WEIGHT) * jnp.maximum(normal_field - 1.0, 0.0)[None],
    ])


coil_residuals_jit = jax.jit(coil_residuals)
coil_jacobian_jit = jax.jit(jax.jacfwd(coil_residuals))
monitor = opt.OptimizationMonitor(plasma_problem)
counts, cached = {"trials": 0}, {}


def residual_and_jacobian(u):
    """Stack the two blocks; the plasma-versus-coil block is structurally zero.

    SciPy asks for the residual and the Jacobian in separate calls at the same
    point, and the equilibrium solve behind them is the expensive part, so the
    pair is computed once and handed out twice.
    """
    u = np.asarray(u, dtype=float)
    if cached.get("key") == u.tobytes():
        return cached["residual"], cached["jacobian"]
    counts["trials"] += 1
    x = x0 + scales * u
    plasma_rows, plasma_jac = plasma_problem.residual_and_jac(x[:n_boundary])
    coil_rows = np.asarray(coil_residuals_jit(jnp.asarray(u)))
    coil_jac = np.asarray(coil_jacobian_jit(jnp.asarray(u)))
    # plasma_jac is in x; the optimizer works in u, so scale its columns.
    plasma_jac = np.asarray(plasma_jac) * scales[None, :n_boundary]
    top = np.hstack([plasma_jac, np.zeros((plasma_jac.shape[0], x_coils0.size))])
    residual = np.concatenate([np.asarray(plasma_rows), coil_rows])
    jacobian = np.vstack([top, coil_jac])
    monitor.cache_evaluation(u, 0.5 * float(residual @ residual), None, {})
    cached.update(key=u.tobytes(), residual=residual, jacobian=jacobian)
    return residual, jacobian


print("Running single_stage_optimization_least_squares.py")
print(f"Fixed-boundary VMEX + ESSOS: {n_boundary} boundary and {x_coils0.size} coil "
      f"variables, exact Jacobian with a structurally zero plasma-coil block")
report = opt.EquilibriumReporter(
    ("QA total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"), ("min |iota|", opt.min_abs_iota, ".4f"))
seed_values = report("seed", plasma_problem.equilibrium_from_x(x_boundary0))


# Stage 0 is the scalar coil fit the other two use: no equilibrium solve.
def coil_fit_value_and_grad(u):
    rows = coil_residuals_jit(jnp.asarray(u))
    value = 0.5 * float(jnp.vdot(rows, rows))
    jac = np.asarray(coil_jacobian_jit(jnp.asarray(u)))
    return value, jac.T @ np.asarray(rows)


coil_fit = minimize(
    coil_fit_value_and_grad, np.zeros_like(x0), jac=True, method="L-BFGS-B",
    bounds=[(0.0, 0.0)] * n_boundary
           + [(-ss.PARAMETER_BOUND, ss.PARAMETER_BOUND)] * x_coils0.size,
    options={"maxiter": COIL_FIT_MAXITER, "maxcor": 20, "ftol": 1e-15, "gtol": 1e-10})
surface_seed, coils_fit = objects_from_x(jnp.asarray(x0 + scales * coil_fit.x))
print(f"[coil fit] {coil_fit.nit} L-BFGS-B iterations, no equilibrium solves: B.n/B RMS "
      f"= {100 * float(ss.normal_field_rms(coils_fit, surface_seed)):.3f}% on the seed")

u = coil_fit.x
initial_residual, _ = residual_and_jacobian(u)
initial_value = 0.5 * float(initial_residual @ initial_residual)
# TRF because it takes the same bounds L-BFGS-B does, and they are active here.
result = least_squares(
    lambda v: residual_and_jacobian(v)[0], u,
    jac=lambda v: residual_and_jacobian(v)[1], method="trf",
    bounds=(-ss.PARAMETER_BOUND * np.ones_like(x0), ss.PARAMETER_BOUND * np.ones_like(x0)),
    max_nfev=MAX_TRIALS, xtol=1e-12, ftol=1e-12, gtol=1e-8,
    verbose=0 if ci_smoke else 2)
u = result.x
final_value = 0.5 * float(result.fun @ result.fun)
print(f"[solve] {result.nfev} residual evaluations, {counts['trials']} trials, "
      f"status {result.status}: {result.message}", flush=True)

ss.finish(
    ss.Case("single_stage_optimization_least_squares", "single_stage_least_squares",
            inp, plasma_problem, objects_from_x, x0, scales, n_boundary,
            coils0, rbc0, zbs0, limits, ci_smoke),
    u=u, monitor=monitor, report=report, seed_values=seed_values,
    headline=f"Objective: {initial_value:.6e} -> {final_value:.6e} after "
             f"{result.nfev} residual evaluations and {counts['trials']} trials",
    summary_extra={"optimization_seconds": round(time.perf_counter() - started, 1),
                   "coil_fit_iterations": int(coil_fit.nit),
                   "trials": counts["trials"], "residual_evaluations": int(result.nfev)})
