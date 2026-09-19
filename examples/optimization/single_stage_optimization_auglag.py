#!/usr/bin/env python
"""Single-stage fixed-boundary optimization, targets as real constraints.

``single_stage_optimization.py`` with the transform floor, the aspect limit and
the coil normal-field limit in a Powell-Hestenes-Rockafellar augmented
Lagrangian instead of quadratic penalties.  Each stage is one bounded L-BFGS-B
solve at fixed multipliers; the multipliers then move, and the penalty grows
only when the violation did not fall by at least a factor of four.

What that buys, measured at 301 trials: the limits can be stated as they are
rather than tightened until the design lands outside them, and the objective
comes out 1.4x lower (4.49e-02 against 6.48e-02).  The penalized file needed
three tunings to get there; see ``examples/README.md``.
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
COIL_DISTANCE_WEIGHT = COIL_SURFACE_DISTANCE_WEIGHT = 1.0e3
# Multipliers move between stages; the penalty grows only on poor progress.
PENALTY_START, PENALTY_GROWTH, PENALTY_MAX = 10.0, 10.0, 1.0e6
CONSTRAINT_TOLERANCE, MAX_STAGES, STAGE_MAXITER, MAX_TRIALS = 1.0e-3, 8, 25, 300
# Circular coils leave B.n/B near 11% RMS.  Fitting them to the frozen seed
# boundary first costs no equilibrium solve and keeps the normal-field term
# from dominating the joint solve.
COIL_FIT_MAXITER = 200
coil_order, n_segments = ss.COIL_ORDER, ss.N_SEGMENTS
if ci_smoke:
    MAX_STAGES, STAGE_MAXITER, MAX_TRIALS, COIL_FIT_MAXITER = 1, 1, 4, 2
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


def augmented(constraints, multipliers, penalty):
    """Powell-Hestenes-Rockafellar term for constraints ``c >= 0``.

    Equal to ``-m*c + p*c**2/2`` while a constraint is violated or active, and
    constant once it holds with slack, so a satisfied target stops pulling.  Its
    derivative in the multipliers is ``(max(m - p*c, 0) - m)/p``, so the
    gradient SciPy already needs also yields the next multipliers.
    """
    shifted = jnp.maximum(multipliers - penalty * constraints, 0.0)
    return jnp.sum(shifted**2 - multipliers**2) / (2.0 * penalty)


def plasma_objective(u, multipliers, penalty):
    """Quasisymmetry plus the transform floor and aspect limit: one solve, one adjoint.

    Floor the profile minimum, not its average: a mean target is satisfiable
    while an interior surface sits near zero transform.
    """
    x = jnp.asarray(x0) + jnp.asarray(scales) * u
    return plasma_problem.jax_objective_from_state(
        x[:x_boundary0.size],
        lambda state, ctx: augmented(jnp.stack([
            opt.min_abs_iota(state, ctx) / limits.iota_constraint - 1.0,
            1.0 - opt.aspect_ratio(state, ctx) / limits.aspect_constraint]),
            multipliers[:2], penalty),
        n_extra_terms=1)


def coil_objective(u, multipliers, penalty):
    """Coil regularization plus the normal-field limit."""
    surface, coils = objects_from_x(jnp.asarray(x0) + jnp.asarray(scales) * u)
    length = jnp.sqrt(LENGTH_WEIGHT) * (coils.length[:ss.N_COILS] - LENGTH_TARGET)
    curvature = jnp.sqrt(CURVATURE_WEIGHT) * jnp.maximum(
        coils.curvature[:ss.N_COILS] - limits.curvature_objective, 0.0)
    costs = jnp.asarray([
        0.5 * jnp.vdot(length, length),
        0.5 * jnp.vdot(curvature, curvature),
        0.5 * COIL_DISTANCE_WEIGHT * loss_coil_separation(
            coils, limits.coil_coil, block_size=32),
        0.5 * COIL_SURFACE_DISTANCE_WEIGHT * loss_coil_surface_distance(
            coils, surface, limits.coil_surface, block_size=32),
    ])
    normal_field = 1.0 - ss.normal_field_rms(coils, surface) / limits.normal_field_constraint
    return (jnp.sum(costs) + augmented(normal_field[None], multipliers[2:], penalty),
            costs)


# Multipliers and penalty are arguments, not captured constants, so moving them
# between stages does not recompile either graph.
plasma_value_and_grad = jax.jit(jax.value_and_grad(
    plasma_objective, argnums=(0, 1), has_aux=True))
coil_value_and_grad = jax.jit(jax.value_and_grad(
    coil_objective, argnums=(0, 1), has_aux=True))
multipliers, penalty = np.zeros(3), PENALTY_START  # iota, aspect, normal field
next_multipliers = {}
monitor = opt.OptimizationMonitor(plasma_problem)
counts, cached = {"trials": 0}, {}
TERM_NAMES = ("coil length", "coil curvature", "coil separation", "coil-surface separation")


def value_and_grad(u):
    u = np.asarray(u, dtype=float)
    key = (u.tobytes(), multipliers.tobytes(), float(penalty))
    if cached.get("key") == key:
        return cached["value"], cached["gradient"].copy()
    counts["trials"] += 1
    args = (jnp.asarray(u), jnp.asarray(multipliers), jnp.asarray(penalty))
    (plasma_value, (qs_rows, plasma_penalty)), (plasma_gradient, d_plasma) = \
        plasma_value_and_grad(*args)
    (coil_value, coil_costs), (coil_gradient, d_coil) = coil_value_and_grad(*args)
    # max(m - p*c, 0), read straight off the multiplier gradient.
    next_multipliers[u.tobytes()] = np.maximum(
        multipliers + penalty * (np.asarray(d_plasma) + np.asarray(d_coil)), 0.0)
    qs_rows = np.asarray(qs_rows)
    terms = {"quasisymmetry": 0.5 * float(qs_rows @ qs_rows),
             "iota and aspect constraints": float(np.asarray(plasma_penalty)[0])}
    terms.update(zip(TERM_NAMES, map(float, np.asarray(coil_costs))))
    value, gradient = monitor.cache_evaluation(
        u, plasma_value + coil_value, plasma_gradient + coil_gradient, terms)
    cached.update(key=key, value=value, gradient=gradient)
    return value, gradient.copy()


print("Running single_stage_optimization_auglag.py")
print(f"Fixed-boundary VMEX + ESSOS: {x_boundary0.size} boundary and "
      f"{x_coils0.size} coil variables, exact reverse-mode derivatives")
report = opt.EquilibriumReporter(
    ("QA total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"), ("min |iota|", opt.min_abs_iota, ".4f"))
seed_values = report("seed", plasma_problem.equilibrium_from_x(x_boundary0))

# Stage 0: coils only, boundary pinned by equal bounds, no equilibrium solve.
# With the normal-field multiplier equal to the penalty this is a plain
# least-squares fit with no slack at the limit.
fit_multipliers = jnp.asarray([0.0, 0.0, PENALTY_START])


def coil_fit_value_and_grad(u):
    (value, _), (gradient, _) = coil_value_and_grad(
        jnp.asarray(u), fit_multipliers, jnp.asarray(PENALTY_START))
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
bounds = [(-ss.PARAMETER_BOUND, ss.PARAMETER_BOUND)] * x0.size
previous_violation, stages, iterations, final_value = np.inf, 0, 0, initial_value
for stage in range(MAX_STAGES):
    if counts["trials"] >= MAX_TRIALS:
        break
    result = minimize(value_and_grad, u, jac=True, method="L-BFGS-B", bounds=bounds,
                      callback=monitor,
                      options={"maxiter": STAGE_MAXITER,
                               "maxfun": MAX_TRIALS - counts["trials"], "maxcor": 20,
                               "maxls": 20, "ftol": 1e-12, "gtol": 1e-8})
    u, final_value = result.x, float(result.fun)
    stages, iterations = stages + 1, iterations + int(result.nit)
    if u.tobytes() not in next_multipliers:
        value_and_grad(u)
    updated = next_multipliers[u.tobytes()]
    # Zero exactly when every constraint holds and each multiplier is zero
    # wherever its constraint has slack.
    violation = float(np.max(np.abs(updated - multipliers))) / penalty
    print(f"[stage {stage + 1}] {result.nit} L-BFGS-B iterations, "
          f"{counts['trials']} trials, violation = {violation:.3e}, multipliers = "
          f"{np.array2string(updated, precision=4)}, penalty = {penalty:.1e}", flush=True)
    multipliers = updated
    if violation <= CONSTRAINT_TOLERANCE and result.status != 1:
        break
    if violation > 0.25 * previous_violation:
        penalty = min(PENALTY_GROWTH * penalty, PENALTY_MAX)
    previous_violation = violation

ss.finish(
    ss.Case("single_stage_optimization_auglag", "single_stage_auglag", inp,
            plasma_problem, objects_from_x, x0, scales, x_boundary0.size,
            coils0, rbc0, zbs0, limits, ci_smoke),
    u=u, monitor=monitor, report=report, seed_values=seed_values,
    headline=f"Objective: {initial_value:.6e} -> {final_value:.6e} after {stages} "
             f"augmented-Lagrangian stages, {iterations} L-BFGS-B iterations and "
             f"{counts['trials']} trials",
    summary_extra={"optimization_seconds": round(time.perf_counter() - started, 1),
                   "coil_fit_iterations": int(coil_fit.nit),
                   "trials": counts["trials"], "stages": stages,
                   "lbfgsb_iterations": iterations})
