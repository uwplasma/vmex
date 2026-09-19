#!/usr/bin/env python
"""Single-stage fixed-boundary plasma and coil optimization.

The simplest of the three, and the one to copy for a new problem: the boundary
Fourier coefficients and the coil Fourier coefficients are one vector, every
constraint is a one-sided quadratic penalty, and SciPy's L-BFGS-B takes it from
there.  VMEX supplies the exact equilibrium derivative and JAX differentiates
the coil objective; the two add.

``_auglag.py`` and ``_least_squares.py`` solve the same problem with the
constraints in an augmented Lagrangian and as least-squares residuals.  All
three reach every target; ``examples/README.md`` has the measured comparison.

A quadratic penalty settles just INSIDE whatever threshold it is handed, since
its pull vanishes with the violation, so the optimizer is given tightened
values (aspect 3.98 against a 4.0 limit, coil separation 0.19 against 0.17) and
the check at the end uses the real limits on an independent, finer solve.  A
run that misses one says which and exits 1; ``VMEX_EXAMPLES_CI=1`` caps the
budget, reports, and exits 0.
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

import vmex as vj
from vmex import optimize as opt

from essos.objective_functions import loss_coil_separation, loss_coil_surface_distance
from essos.surfaces import surfacerzfourier_from_boundary

import _single_stage as ss

started = time.perf_counter()
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
limits = ss.Limits()

LENGTH_TARGET, LENGTH_WEIGHT = 5.3, 1.0   # the coils land near 5.3 m regardless
CURVATURE_WEIGHT = 10.0
COIL_DISTANCE_CONSTRAINT, COIL_SURFACE_DISTANCE_CONSTRAINT = 0.19, 0.21
COIL_DISTANCE_WEIGHT = COIL_SURFACE_DISTANCE_WEIGHT = 1.0e4
CONSTRAINT_WEIGHT = 1.0e3
MAXITER, MAX_TRIALS = 200, 300
# Circular coils leave B.n/B near 11% RMS.  Fitting them to the frozen seed
# boundary first costs no equilibrium solve and keeps the normal-field term
# from dominating the joint solve.
COIL_FIT_MAXITER = 200
coil_order, n_segments = ss.COIL_ORDER, ss.N_SEGMENTS
if ci_smoke:
    MAXITER, MAX_TRIALS, COIL_FIT_MAXITER = 2, 4, 2
    coil_order, n_segments = 2, 24

inp = ss.seed_input(ci_smoke)
curves0, coils0 = ss.seed_coils(n_segments, coil_order)
qs = opt.QuasisymmetryRatioResidual(ss.SURFACES, helicity_m=1, helicity_n=0)
plasma_problem = opt.VmecProblem.from_tuples(
    inp, [(qs.residuals_state, 0.0, 1.0)], max_mode=ss.MAX_MODE,
    use_ess=True, progress=not ci_smoke)

x_boundary0 = plasma_problem.x0
rbc0, zbs0 = plasma_problem.boundary_from_x(x_boundary0)
x_coils0 = np.asarray(curves0.dofs).ravel()
x0 = np.concatenate([x_boundary0, x_coils0])
# SciPy works in dimensionless increments u, with x = x0 + scales*u.
scales = np.concatenate([0.02 * plasma_problem.scales, np.full(x_coils0.size, 0.05)])


def objects_from_x(x):
    rbc, zbs = plasma_problem.boundary_from_x(x[:x_boundary0.size])
    surface = surfacerzfourier_from_boundary(
        rbc, zbs, inp.nfp, nphi=ss.NPHI, ntheta=ss.NTHETA)
    coils = coils0.with_dofs(jnp.concatenate((x[x_boundary0.size:], coils0.dofs_currents)))
    return surface, coils


def hinge(constraints):
    """``w/2 * sum(max(-c, 0)**2)`` for constraints written as ``c >= 0``."""
    return 0.5 * CONSTRAINT_WEIGHT * jnp.sum(jnp.maximum(-constraints, 0.0)**2)


def plasma_objective(u):
    """Quasisymmetry plus the transform floor and aspect limit: one solve, one adjoint.

    Floor the profile minimum, not its average: a mean target is satisfiable
    while an interior surface sits near zero transform.
    """
    x = jnp.asarray(x0) + jnp.asarray(scales) * u
    return plasma_problem.jax_objective_from_state(
        x[:x_boundary0.size],
        lambda state, ctx: hinge(jnp.stack([
            opt.min_abs_iota(state, ctx) / limits.iota_constraint - 1.0,
            1.0 - opt.aspect_ratio(state, ctx) / limits.aspect_constraint])),
        n_extra_terms=1)


def coil_objective(u):
    """Coil regularization plus the normal-field limit."""
    surface, coils = objects_from_x(jnp.asarray(x0) + jnp.asarray(scales) * u)
    length = jnp.sqrt(LENGTH_WEIGHT) * (coils.length[:ss.N_COILS] - LENGTH_TARGET)
    curvature = jnp.sqrt(CURVATURE_WEIGHT) * jnp.maximum(
        coils.curvature[:ss.N_COILS] - limits.curvature_objective, 0.0)
    costs = jnp.asarray([
        0.5 * jnp.vdot(length, length),
        0.5 * jnp.vdot(curvature, curvature),
        0.5 * COIL_DISTANCE_WEIGHT * loss_coil_separation(
            coils, COIL_DISTANCE_CONSTRAINT, block_size=32),
        0.5 * COIL_SURFACE_DISTANCE_WEIGHT * loss_coil_surface_distance(
            coils, surface, COIL_SURFACE_DISTANCE_CONSTRAINT, block_size=32),
    ])
    normal_field = 1.0 - ss.normal_field_rms(coils, surface) / limits.normal_field_constraint
    return jnp.sum(costs) + hinge(normal_field[None]), costs


plasma_value_and_grad = jax.jit(jax.value_and_grad(plasma_objective, has_aux=True))
coil_value_and_grad = jax.jit(jax.value_and_grad(coil_objective, has_aux=True))
monitor = opt.OptimizationMonitor(plasma_problem)
counts, cached = {"trials": 0}, {}
TERM_NAMES = ("coil length", "coil curvature", "coil separation", "coil-surface separation")


def value_and_grad(u):
    u = np.asarray(u, dtype=float)
    if cached.get("key") == u.tobytes():
        return cached["value"], cached["gradient"].copy()
    counts["trials"] += 1
    (plasma_value, (qs_rows, plasma_penalty)), plasma_gradient = \
        plasma_value_and_grad(jnp.asarray(u))
    (coil_value, coil_costs), coil_gradient = coil_value_and_grad(jnp.asarray(u))
    qs_rows = np.asarray(qs_rows)
    terms = {"quasisymmetry": 0.5 * float(qs_rows @ qs_rows),
             "iota and aspect penalty": float(np.asarray(plasma_penalty)[0])}
    terms.update(zip(TERM_NAMES, map(float, np.asarray(coil_costs))))
    value, gradient = monitor.cache_evaluation(
        u, plasma_value + coil_value, plasma_gradient + coil_gradient, terms)
    cached.update(key=u.tobytes(), value=value, gradient=gradient)
    return value, gradient.copy()


print("Running single_stage_optimization.py")
print(f"Fixed-boundary VMEX + ESSOS: {x_boundary0.size} boundary and "
      f"{x_coils0.size} coil variables, exact reverse-mode derivatives")
report = opt.EquilibriumReporter(
    ("QA total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"), ("min |iota|", opt.min_abs_iota, ".4f"))
seed_values = report("seed", plasma_problem.equilibrium_from_x(x_boundary0))

# Stage 0: coils only, boundary pinned by equal bounds, no equilibrium solve.
def coil_fit_value_and_grad(u):
    (value, _), gradient = coil_value_and_grad(jnp.asarray(u))
    return float(value), np.asarray(gradient)


coil_fit = minimize(
    coil_fit_value_and_grad, np.zeros_like(x0), jac=True, method="L-BFGS-B",
    bounds=[(0.0, 0.0)] * x_boundary0.size
           + [(-ss.PARAMETER_BOUND, ss.PARAMETER_BOUND)] * x_coils0.size,
    options={"maxiter": COIL_FIT_MAXITER, "maxcor": 20, "ftol": 1e-15, "gtol": 1e-10})
surface_seed, coils_fit = objects_from_x(jnp.asarray(x0 + scales * coil_fit.x))
print(f"[coil fit] {coil_fit.nit} L-BFGS-B iterations, no equilibrium solves: B.n/B RMS "
      f"= {100 * float(ss.normal_field_rms(coils_fit, surface_seed)):.3f}% on the seed")

u = coil_fit.x
vj.FunctionProblem.from_functions(u, value_and_grad=value_and_grad).compile_value_and_gradient(
    report_interval=10.0)
initial_value = monitor.records[0].cost
# L-BFGS-B rather than BFGS: the bounds are part of the problem, keeping a trial
# boundary inside the range the seed solve converges on, and at the solution 3 of
# the 99 coil dofs sit exactly on them.  BFGS cannot represent a bound.
result = minimize(value_and_grad, u, jac=True, method="L-BFGS-B", callback=monitor,
                  bounds=[(-ss.PARAMETER_BOUND, ss.PARAMETER_BOUND)] * x0.size,
                  options={"maxiter": MAXITER, "maxfun": MAX_TRIALS, "maxcor": 20,
                           "maxls": 20, "ftol": 1e-12, "gtol": 1e-8})
u = result.x
print(f"[solve] {result.nit} L-BFGS-B iterations, {counts['trials']} trials, "
      f"status {result.status}: {result.message}", flush=True)

ss.finish(
    ss.Case("single_stage_optimization", "single_stage", inp, plasma_problem, objects_from_x, x0, scales,
            x_boundary0.size, coils0, rbc0, zbs0, limits, ci_smoke),
    u=u, monitor=monitor, report=report, seed_values=seed_values,
    headline=f"Objective: {initial_value:.6e} -> {float(result.fun):.6e} after "
             f"{result.nit} L-BFGS-B iterations and {counts['trials']} trials",
    summary_extra={"optimization_seconds": round(time.perf_counter() - started, 1),
                   "coil_fit_iterations": int(coil_fit.nit),
                   "trials": counts["trials"], "lbfgsb_iterations": int(result.nit)})
