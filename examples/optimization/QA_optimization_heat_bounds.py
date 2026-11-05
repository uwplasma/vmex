#!/usr/bin/env python
"""Optimize a QA boundary at fixed physical heat-transport operating points.

VMEX_HEAT_BOUND_MODEL=coulomb selects full Landau collisions, parallel/mirror
orbits and incompressible ExB on a fitted Boozer surface. The default Lorentz
reference retains bounce geometry and prescribed tangential precession.
Equilibrium adjoints drive bounded optimization with geometric constraints;
independent equilibrium and geometry refinement checks the endpoints. These
are bounds for the stated kinetic models; see docs/explanation/confinement.rst.
VMEX_HEAT_BOUND_CANDIDATE checks a supplied boundary using this seed's flux
and profiles, with the same heat-gradient and endpoint checks, without optimization.
"""

import gc
import json
import os
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from vmex.core.neoclassical import _state_stencils, reference_pitch_grid_from_boozer
from vmex.core.statephysics import _iotas_half

import vmex as vj
from vmex import optimize as opt

# Model, seed, surfaces, and profiles (T[eV], n[m^-3], lnLambda, |Er|[V/m]):
MODEL = os.environ.get("VMEX_HEAT_BOUND_MODEL", "lorentz")
COULOMB = MODEL == "coulomb"
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / (
    "input.LandremanPaul2021_QA_reactorScale_lowres" if COULOMB else "input.LandremanPaul2021_QA_lowres")
OUTPUT = Path(os.environ.get("VMEX_HEAT_BOUNDS_OUTPUT", "."))
VALIDATE_BOUNDARY = os.environ.get("VMEX_HEAT_BOUND_CANDIDATE")  # Optional input boundary; this seed supplies fixed physics.
SURFACES, CHECK_SURFACES = (.5,), ((.25, .5, .75) if COULOMB else (.5,))
PROFILES = ((100., 1.5e20, 15., 100.), (1000., 1.e20, 17., 1000.), (10000., 1.e20, 19., 10000.))
if os.environ.get("VMEX_HEAT_BOUND_INCLUDE_HOT", "0") == "0":
    PROFILES = PROFILES[:2]
MASS_KG, ELEMENTARY_CHARGE, EPSILON0 = 3.3435837724e-27, 1.602176634e-19, 8.8541878128e-12
MACRO_ORDER = os.environ.get("VMEX_HEAT_BOUND_MACRO_ORDER")
COLLISION_FREQUENCIES, LORENTZ_FIELDS = (100., 1000.), (-250., 0., 250.)

# Fix flux and RBC(0,0); preserve aspect, volume, and iota at s=.25,.5,.75:
MAX_MODE = int(os.environ.get("VMEX_HEAT_BOUND_MAX_MODE", "3" if COULOMB else "1"))
MAXITER = int(os.environ.get("VMEX_HEAT_BOUND_MAXITER", "80" if COULOMB else "8"))
PARAMETER_STEP, MAX_PARAMETER_CHANGE = (.05 if COULOMB else .002), 5.
CONSTRAINT_TOLERANCE, SMOOTHING = .001, .02
MAX_REJECTED_TRIALS = 10
NS, MPOL, MBOZ = (26, 8, 12) if COULOMB else (14, 6, 6)
FORWARD_FTOL, ADJOINT_TOL, REFINE_TOL = (1.e-16, 1.e-9, 1.e-12) if COULOMB else (1.e-11, 1.e-6, 1.e-10)
GEOMETRY_ORDER, VERIFY_GEOMETRY_ORDER = (8, 10), (16, 20)
FINAL_NS, FINAL_FTOL = 50, 1.e-13
FINAL_MPOL, FINAL_MBOZ = (10, 16) if COULOMB else (8, 10)
PITCH_CELLS, ENERGY_POINTS, VERIFY_PITCH_CELLS, VERIFY_ENERGY_POINTS = 32, 6, 96, 12
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    if COULOMB:
        PARAMETER_STEP, MAX_PARAMETER_CHANGE = .002, .05
    MAX_MODE, MAXITER, NS, MPOL, MBOZ, FINAL_NS = 1, 2, 14, 6, 6, 26
    FINAL_MPOL, FINAL_MBOZ = MPOL, MBOZ
    PROFILES = PROFILES[:1]
    GEOMETRY_ORDER, VERIFY_GEOMETRY_ORDER = (3, 1), (6, 4)
    PITCH_CELLS, ENERGY_POINTS, VERIFY_PITCH_CELLS, VERIFY_ENERGY_POINTS = 16, 4, 32, 6

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium and fixed operating points #########################
if MODEL not in ("lorentz", "coulomb"):
    raise ValueError("VMEX_HEAT_BOUND_MODEL must be lorentz or coulomb")
OUTPUT.mkdir(parents=True, exist_ok=True)
inp = vj.VmecInput.from_file(INPUT_FILE).change_resolution(
    mpol=MPOL, ntor=MPOL, ntheta=4 * MPOL, nzeta=4 * MPOL)
if not COULOMB:  # Perturbed small-size seed for the Lorentz reference.
    rbc = inp.rbc.copy()
    rbc[inp.ntor + 1, 0] += .005
    inp = replace(inp, rbc=rbc)
inp = replace(inp, delt=.7, ns_array=np.array([NS]),
              ftol_array=np.array([FORWARD_FTOL]), niter_array=np.array([9000]))
seed = opt.solve_equilibrium(inp, raise_on_max_iterations=True)
if not COULOMB:
    booz, _ = _state_stencils(seed.state, seed.runtime, SURFACES, 6, 6, 2)
    grid = reference_pitch_grid_from_boozer(**booz, ncells=PITCH_CELLS, helicity_m=1, helicity_n=0)


def terms(surfaces, verify=False):
    """Physical rates stay fixed; each equilibrium determines its normalized rates."""
    mboz = FINAL_MBOZ if verify else MBOZ
    if COULOMB:
        result = []
        for temperature, density, logarithm, field in PROFILES:
            speed = np.sqrt(2 * ELEMENTARY_CHARGE * temperature / MASS_KG)
            rate = ELEMENTARY_CHARGE**4 * logarithm * density / (8 * np.pi * EPSILON0**2 * MASS_KG**2 * speed**3)
            result.append(vj.ConservingHeatBound(
                surfaces, temperature_ev=temperature, mass_kg=MASS_KG, collision_strengths=[rate],
                radial_electric_fields=[-field, 0., field], mboz=mboz, nboz=mboz,
                geometry_order=VERIFY_GEOMETRY_ORDER if verify else GEOMETRY_ORDER,
                fit_grid=256 if verify else 128, integration_order=40 if verify else 32, rho_bound=.4,
                heat_inverse_bound=1.25 * np.nextafter(1.497838441524309, np.inf),
                stress_inverse_bound=np.nextafter(1.957693999985583, np.inf),
                macro_order=int(MACRO_ORDER) if MACRO_ORDER else None))
        return result
    return [vj.NeoclassicalHeatBound(
        surfaces, temperature_ev=1000., collision_frequencies=COLLISION_FREQUENCIES,
        radial_electric_fields=LORENTZ_FIELDS, helicity_m=1, helicity_n=0, well_center=grid["well_center"],
        pitch_fractions=np.linspace(0., 1., (VERIFY_PITCH_CELLS if verify else PITCH_CELLS) + 1),
        points_per_period=1024 if verify else 128, quadrature_order=256 if verify else 32,
        nenergy=VERIFY_ENERGY_POINTS if verify else ENERGY_POINTS, mboz=mboz, nboz=mboz,
        nalpha=21 if verify and not ci_smoke else 13)]


def constraints(state, runtime):
    """Relative constraints are formed against the unmodified seed values."""
    transform = jnp.interp(jnp.array([.25, .5, .75]), runtime.setup.s_half[1:], _iotas_half(state, runtime)[1:])
    return jnp.concatenate((jnp.array([opt.aspect_ratio(state, runtime), opt.volume(state, runtime)]), transform))


def common_gauge(problem):
    """Use one smooth m=1 prescription in values, adjoints, and exports."""
    from vmex.core import implicit as imp
    from vmex.core.solver import evaluate_forces

    cfg = problem.metadata["config"]
    params0 = imp.params_from_input(problem.metadata["input"], device=cfg.device)
    def params_of(x):
        return replace(params0, **dict(zip(("rbc", "zbs", "rbs", "zbc"), problem.boundary_from_x(x))))

    reference_params = params_of(jnp.asarray(problem.x0))
    frozen, mask = imp.solve_implicit_with_aux(reference_params, cfg)
    project, edge = imp._dof_projector(cfg, mask), imp._edge_mask(cfg)
    reference_guess = imp._initial_state(imp.runtime_from_params(reference_params, cfg).setup)

    def prescribed(runtime):
        return jax.tree.map(lambda a, b, c: a + (b - c), frozen, imp._initial_state(runtime.setup), reference_guess)

    specification = (jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), frozen), jax.ShapeDtypeStruct((), jnp.int32))
    memo, diagnostics = {}, []

    def anchor(guess, params):
        runtime = imp.runtime_from_params(params, cfg)
        initial = imp._assemble(project(guess), runtime, prescribed(runtime), project, edge)
        base = imp._anchor_residual(cfg, params, initial, mask)
        state = imp._refined_state(cfg, params, initial, mask, krylov=False)
        _, _, geometry = evaluate_forces(state, runtime)
        invalid = bool(geometry.jacobian_sign_changed)
        error = np.inf if invalid else imp._anchor_residual(cfg, params, state, mask)
        return state, error, base, invalid

    certified_seed, seed_residual, _, _ = anchor(frozen, reference_params)
    if not np.isfinite(seed_residual) or seed_residual > 1.e-10:
        raise RuntimeError("the continuation seed did not pass the equilibrium residual/Jacobian gate")

    def host(params):
        key = imp._params_key(params)
        if memo.get("key") == key:
            return memo["result"]
        try:
            with imp._device_context(cfg):
                solved = imp._host_solve(cfg, params)
        except imp.VmecError as error:
            imp._LAST_STATUS_ERROR[cfg] = error
            state, status, solved = frozen, 1, None
        else:
            imp._LAST_STATUS_ERROR.pop(cfg, None)
            state, status = solved.state, 0 if bool(solved.converged) or imp._fsq(solved) <= cfg.max_fsq_ratio * cfg.ftol else 2
        if status == 0:
            params = jax.tree.map(jnp.asarray, params)
            state, error, base, invalid = anchor(state, params)
            start, previous = memo.get("valid_params", reference_params), memo.get("valid_state", certified_seed)
            row = {"host_initial_residual": base, "host_final_residual": error, "host_invalid_jacobian": invalid, "initializer": "host",
                   "maximum_boundary_step_m": max(float(np.max(abs(getattr(params, name) - getattr(start, name))))
                                                  for name in ("rbc", "zbs", "rbs", "zbc"))}
            if not np.isfinite(error) or error > 1.e-10:
                retry, retry_error, _, retry_invalid = anchor(previous, params)
                row.update(previous_root_residual=retry_error, previous_root_invalid_jacobian=retry_invalid,
                           continuation_origin="previous certified root" if "valid_state" in memo else "seed",
                           seed_residual=seed_residual)
                if np.isfinite(retry_error) and retry_error <= 1.e-10:
                    state, error, row["initializer"] = retry, retry_error, row["continuation_origin"]
                else:
                    continued, fraction, increment, successes = previous, 0., min(.125, .002 / max(row["maximum_boundary_step_m"], 1.e-30)), 0
                    older, older_fraction, attempts = previous, None, []
                    print(f"Continuation from {row['continuation_origin']}, boundary step={row['maximum_boundary_step_m']:.6g} m", flush=True)
                    for _ in range(64):
                        next_fraction = min(1., fraction + increment)
                        target = params if next_fraction == 1. else jax.tree.map(lambda a, b: a + (b - a) * next_fraction, start, params)
                        guess = continued if older_fraction is None else jax.tree.map(
                            lambda a, b: a + (a - b) * (next_fraction - fraction) / (fraction - older_fraction), continued, older)
                        trial, trial_error, _, trial_invalid = anchor(guess, target)
                        valid = np.isfinite(trial_error) and trial_error <= 1.e-10
                        attempts.append({"fraction": next_fraction, "residual": float(trial_error) if np.isfinite(trial_error) else None,
                                         "invalid_jacobian": trial_invalid, "secant_predictor": older_fraction is not None, "accepted": bool(valid)})
                        print(f"Continuation {len(attempts)}/64: fraction={next_fraction:.9g}, residual={trial_error:.3g}, accepted={valid}", flush=True)
                        if valid:
                            older, older_fraction = continued, fraction
                            continued, fraction, successes = trial, next_fraction, successes + 1
                            if fraction == 1.:
                                state, error, row["initializer"] = continued, trial_error, "bounded certified continuation"
                                break
                            if successes == 2:
                                increment, successes = 2 * increment, 0
                        else:
                            increment, successes = increment / 2, 0
                            if increment < 2.**-16:
                                break
                    row.update(continuation_attempts=attempts, continuation_fraction_reached=fraction)
            row.update(final_residual=error, accepted=bool(np.isfinite(error) and error <= 1.e-10))
            diagnostics.append({key: value if not isinstance(value, float) or np.isfinite(value) else None
                                for key, value in row.items()})
            if not row["accepted"]:
                status = 3
            else:
                memo.update(valid_params=params, valid_state=state)
        result = (jax.tree.map(np.asarray, state), np.int32(status))
        memo.update(key=key, result=result, solved=solved)
        return result

    @jax.custom_vjp
    def solve(params):
        return jax.pure_callback(host, specification, params)

    def forward(params):
        state, status = solve(params)
        return (state, status), (params, state, mask, status, jnp.asarray(cfg.adjoint_tol))

    def backward(saved, cotangent):
        params, state, mask, status, tolerance = saved
        def success(_):
            evolved = project(state)
            multiplier, report = imp._adjoint_block_core(
                params, evolved, state, mask, project(cotangent[0]), tolerance, cfg)
            imp._enforce_adjoint_stats(cfg, report)
            def paths(parameters):
                runtime = imp.runtime_from_params(parameters, cfg)
                assembled = imp._assemble(evolved, runtime, prescribed(runtime), project, edge)
                return assembled, project(imp._raw_force_state(assembled, runtime))
            _, pullback = jax.vjp(paths, params)
            return pullback((cotangent[0], jax.tree.map(jnp.negative, multiplier)))[0]
        return (jax.lax.cond(status == 0, success, lambda _: jax.tree.map(jnp.zeros_like, params), None),)

    solve.defvjp(forward, backward)

    def state_runtime_status(x):
        params = params_of(x)
        state, status = solve(params)
        return state, imp.runtime_from_params(params, cfg), status

    problem.metadata["jax_state_runtime_status"] = state_runtime_status

    def equilibrium(x):
        params = params_of(jnp.asarray(x))
        state, status = host(params)
        if status:
            raise RuntimeError(f"common-gauge equilibrium rejected: status {status}")
        runtime = imp.runtime_from_params(params, cfg)
        _, force, _ = evaluate_forces(state, runtime)
        result = replace(memo["solved"], state=state, fsqr=float(force.fsqr), fsqz=float(force.fsqz), fsql=float(force.fsql))
        return opt.Equilibrium(problem.input_from_x(x), state, runtime, result)

    def clear_initializer():
        memo.clear()
        for cache in (imp._LAST_SOLVE, imp._HOT_CACHE, imp._PERTURB_SEED, imp._LAST_REFINED, imp._LAST_REFINEMENT_CORRECTION, imp._LAST_BANDS):
            cache.pop(cfg, None)

    def state_difference(reference):
        return float(np.max([np.max(abs(a - b)) / max(1., float(np.max(abs(a))), float(np.max(abs(b))))
                             for a, b in zip(jax.tree.leaves(reference), jax.tree.leaves(memo["result"][0]))]))

    equilibrium.diagnostics, equilibrium.clear_initializer = diagnostics, clear_initializer
    equilibrium.state_snapshot = lambda: memo["result"][0]
    equilibrium.state_difference = state_difference
    return equilibrium


objectives = terms(SURFACES)
problem = opt.VmecProblem.from_tuples(
    inp, [(lambda state, runtime: jnp.array(0.), 0., 1.)], max_mode=MAX_MODE,
    vary_major_radius=False, use_ess=COULOMB, ess_alpha=1.2, restart_from=seed,
    forward_ftol=FORWARD_FTOL, forward_max_iterations=9000, adjoint_tol=ADJOINT_TOL,
    refine_tol=REFINE_TOL, device="cpu", progress=True, evaluation_progress=False)
materialize = common_gauge(problem) if COULOMB else problem.equilibrium_from_x
if COULOMB:
    seed = materialize(problem.x0)
initial = [term.compute_state(seed.state, seed.runtime) for term in objectives]
if not all(np.all(item["valid"]) for item in initial):
    raise RuntimeError("invalid seed kinetic geometry")
references, targets = [item["upper"] for item in initial], constraints(seed.state, seed.runtime)
failure_value = problem.metadata["jax_failure_value"]
problem.metadata["jax_failure_value"] = lambda x: 100 * failure_value(x)
x0 = problem.x0.copy()
active = np.arange(x0.size) if COULOMB else np.array([
    problem.dof_names.index(name) for name in ("RBC(1,0)", "RBC(-1,1)", "ZBS(-1,1)")])
step = PARAMETER_STEP * problem.scales[active]


def objective(y, multipliers, penalty):
    """Combine all fixed scenarios with relative equality constraints."""
    def costs(state, runtime):
        ratios = jnp.concatenate([(term.compute_state(state, runtime)["upper"] / ref).ravel()
                                  for term, ref in zip(objectives, references)])
        heat = SMOOTHING * jax.scipy.special.logsumexp(ratios / SMOOTHING)
        delta = constraints(state, runtime) / targets - 1
        return jnp.stack((heat, jnp.vdot(multipliers, delta), .5 * penalty * jnp.vdot(delta, delta)))
    parameters = jnp.asarray(x0).at[active].add(step * y)
    return problem.jax_objective_from_state(parameters, costs, n_extra_terms=3, return_status=True)


evaluate = jax.jit(jax.value_and_grad(objective, argnums=(0, 1), has_aux=True))
history, multipliers, penalty, y = [], np.zeros(5), 0. if COULOMB else 100., np.zeros(len(active))


def value_and_gradient(y):
    """One equilibrium adjoint supplies the scaled boundary gradient."""
    (value, (_, costs, status)), (gradient, delta) = evaluate(y, multipliers, penalty)
    status = int(status) or (0 if np.isfinite(value) and np.all(np.isfinite(gradient)) and np.all(np.isfinite(delta)) else 4)
    heat = float(costs[0]) if status == 0 else None
    if COULOMB and status:
        value, gradient = 1.e6 + np.dot(y, y), 2 * np.asarray(y)
    history.append({"value": float(value), "heat": heat, "status": status, "y": np.asarray(y).tolist(),
                    "constraints": np.asarray(delta).tolist() if np.all(np.isfinite(delta)) else None})
    print(f"heat={heat}, constraint={np.max(np.abs(delta)):.3g}, status={status}", flush=True)
    return float(value), np.asarray(gradient)


### Check the derivative and optimize #########################################
seed_value, seed_gradient = value_and_gradient(y)
if history[-1]["status"] != 0 or not np.all(np.isfinite(seed_gradient)) or not np.isfinite(seed_value):
    raise RuntimeError("invalid seed equilibrium value or gradient")
direction = np.sin(np.arange(len(y)) + 1.)
direction /= np.linalg.norm(direction)

def derivative_check(point, gradient):
    """Reuse the executable; endpoint gradients are not finite-difference data."""
    analytic, differences = float(gradient @ direction), []
    for h in ((.0004, .00008, .000016) if COULOMB else (.002,)):
        (plus, ap), (_, dp) = evaluate(point + h * direction, multipliers, penalty)
        (minus, am), (_, dm) = evaluate(point - h * direction, multipliers, penalty)
        if int(ap[-1]) != 0 or int(am[-1]) != 0 or not np.all(np.isfinite([plus, minus, analytic])):
            raise RuntimeError("invalid finite-difference equilibrium or value")
        fd = float((plus - minus) / (2 * h))
        differences.append({"step": h, "finite_difference": fd,
            "relative_error": abs(analytic - fd) / max(abs(analytic), abs(fd), 1.e-14),
            "constraint_finite_difference": np.asarray((dp - dm) / (2 * h)).tolist()})
    if not np.isfinite(differences[-1]["relative_error"]) or differences[-1]["relative_error"] > (.002 if COULOMB else .03):
        raise RuntimeError(f"equilibrium derivative check failed: {differences}")
    return analytic, differences


ad, differences = derivative_check(y, seed_gradient)
accepted_iterates, constraint_gradient_check, candidate_gradient_check = [], {}, None
rejections, incumbent_retained = {"total": 0, "consecutive": 0}, False
if VALIDATE_BOUNDARY:
    supplied = vj.VmecInput.from_file(VALIDATE_BOUNDARY)
    resolution = {"mpol": max(inp.mpol, supplied.mpol), "ntor": max(inp.ntor, supplied.ntor)}
    supplied = supplied.change_resolution(**resolution)
    y = (problem.x_from_input(supplied)[active] - x0[active]) / step
    reconstructed = problem.input_from_x(np.asarray(jnp.asarray(x0).at[active].add(step * y))).change_resolution(
        **resolution)
    if supplied.nfp != inp.nfp or any(not np.allclose(getattr(supplied, name), getattr(reconstructed, name), rtol=0., atol=1.e-12)
                                     for name in ("rbc", "zbs", "rbs", "zbc")):
        raise ValueError("supplied boundary changes fixed modes or field periods")
    result = OptimizeResult(x=y, nit=0, nfev=0, message="supplied boundary verification", success=True)
    iterations = 0
    result.fun, _ = value_and_gradient(y)
elif COULOMB:
    cache = {}

    class RejectedEquilibria(RuntimeError):
        """Stop a search whose recent equilibrium trials all failed."""

    def sample(point):
        """Share one certified value and gradient between SLSQP callbacks."""
        key = np.asarray(point).tobytes()
        if cache.get("key") != key:
            value, gradient = value_and_gradient(point)
            cache.clear()
            cache.update(key=key, value=value, gradient=gradient, record=history[-1])
            rejections["total"] += bool(history[-1]["status"])
            rejections["consecutive"] = rejections["consecutive"] + 1 if history[-1]["status"] else 0
            if rejections["consecutive"] >= MAX_REJECTED_TRIALS:
                raise RejectedEquilibria(f"stopped after {MAX_REJECTED_TRIALS} consecutive rejected equilibria")
        return cache

    def bounded_value(point):
        """Return the heat objective or a finite wall for an invalid trial."""
        data = sample(point)
        return data["value"], data["gradient"]

    def inequalities(point):
        """Reserve half the final tolerance for equilibrium refinement."""
        record = sample(point)["record"]
        if record["status"]:
            return np.full(10, -1000.)
        delta = np.asarray(record["constraints"])
        return np.concatenate((CONSTRAINT_TOLERANCE / 2 - delta, CONSTRAINT_TOLERANCE / 2 + delta)) / CONSTRAINT_TOLERANCE

    def constraint_jacobian(point):
        """Recover five constraint rows from the certified equilibrium adjoint."""
        data = sample(point)
        if data["record"]["status"]:
            return np.zeros((10, len(point)))
        if "jacobian" not in data:
            matrix = np.asarray([np.asarray(evaluate(point, row, penalty)[1][0]) - data["gradient"] for row in np.eye(5)])
            if not np.all(np.isfinite(matrix)):
                raise RuntimeError("constraint adjoint did not produce a finite Jacobian")
            if not accepted_iterates:
                analytic, finite = matrix @ direction, np.asarray(differences[-1]["constraint_finite_difference"])
                error = float(np.linalg.norm(analytic - finite) / max(np.linalg.norm(analytic), np.linalg.norm(finite), 1.e-14))
                constraint_gradient_check.update(relative_error=error)
                if not np.isfinite(error) or error > .002:
                    raise RuntimeError("constraint derivative check failed")
            data["jacobian"] = np.concatenate((-matrix, matrix)) / CONSTRAINT_TOLERANCE
            if not accepted_iterates or not np.array_equal(point, accepted_iterates[-1]["y"]):
                accepted_iterates.append(dict(data["record"], iteration=len(accepted_iterates), phase="gradient requested"))
        return data["jacobian"]

    try:
        result = minimize(bounded_value, y, jac=True, method="SLSQP",
            bounds=[(-MAX_PARAMETER_CHANGE, MAX_PARAMETER_CHANGE)] * len(y),
            constraints={"type": "ineq", "fun": inequalities, "jac": constraint_jacobian},
            options={"maxiter": MAXITER, "ftol": 1.e-10, "disp": True})
    except RejectedEquilibria as error:
        rejections["consecutive_at_stop"] = rejections["consecutive"]
        result = OptimizeResult(x=None, fun=np.inf, nit=max(0, len(accepted_iterates) - 1),
            message=str(error), success=False)
    feasible = [row for row in accepted_iterates if not row["status"]
                and np.max(np.abs(row["constraints"])) <= CONSTRAINT_TOLERANCE]
    best = min(feasible, key=lambda row: row["heat"]) if feasible else None
    returned = next((row for row in reversed(history) if np.array_equal(row["y"], result.x)), None)
    if best is not None and (returned is None or returned["status"]
            or np.max(np.abs(returned["constraints"])) > CONSTRAINT_TOLERANCE or best["heat"] < result.fun):
        result.x, result.fun, result.success = np.asarray(best["y"]), best["heat"], False
        result.message = str(result.message) + "; retained best feasible accepted boundary"
        incumbent_retained = True
        materialize.clear_initializer()
        cache.clear()
    if result.x is None:
        raise RuntimeError("search stopped without a feasible accepted boundary")
    y, iterations = result.x, int(result.nit)
    sample(y)
else:
    iterations = 0
    for _ in range((MAXITER + 19) // 20):
        result = minimize(value_and_gradient, y, jac=True, method="L-BFGS-B",
            bounds=[(-MAX_PARAMETER_CHANGE, MAX_PARAMETER_CHANGE)] * len(y),
            options={"maxiter": min(20, MAXITER - iterations), "maxls": 20, "maxcor": 20, "gtol": 1.e-7, "ftol": 1.e-12})
        y, iterations = result.x, iterations + result.nit
        value_and_gradient(y)
        delta = np.asarray(history[-1]["constraints"])
        multipliers += penalty * delta
        if np.max(abs(delta)) > CONSTRAINT_TOLERANCE / 2:
            penalty *= 5
        if iterations >= MAXITER or (result.success and np.max(abs(delta)) < CONSTRAINT_TOLERANCE / 2):
            break
if history[-1]["status"] != 0 or not np.isfinite(result.fun):
    raise RuntimeError("final optimizer equilibrium was not valid")
if COULOMB:
    (warm, (_, _, warm_status)), (warm_gradient, warm_delta) = evaluate(y, multipliers, penalty)
    jax.block_until_ready((warm, warm_gradient, warm_delta, warm_status))
    warm_state = materialize.state_snapshot()
    materialize.clear_initializer()
    (cold, (_, _, cold_status)), (cold_gradient, cold_delta) = evaluate(y, multipliers, penalty)
    jax.block_until_ready((cold, cold_gradient, cold_delta, cold_status))
    cold_state, state_error = materialize.state_snapshot(), materialize.state_difference(warm_state)
    repeat_error = float(abs(warm - cold) / max(abs(warm), abs(cold), 1.))
    constraint_error = float(np.max(abs(warm_delta - cold_delta)))
    if (int(warm_status) != 0 or int(cold_status) != 0 or not np.all(np.isfinite(warm_gradient))
            or not np.isfinite(repeat_error) or not np.isfinite(constraint_error)
            or not np.isfinite(state_error) or state_error >= 1.e-9 or repeat_error >= 1.e-9 or constraint_error >= 1.e-9):
        raise RuntimeError("candidate differs between native-cold and warm initialization")
    final_ad, final_differences = derivative_check(y, cold_gradient)
    repeated = evaluate(y, multipliers, penalty)
    jax.block_until_ready(repeated)
    (repeat_value, (_, _, repeat_status)), (_, repeat_delta) = repeated
    repeat_errors = (float(abs(repeat_value - cold) / max(abs(repeat_value), abs(cold), 1.)),
                     float(np.max(abs(repeat_delta - cold_delta))), materialize.state_difference(cold_state))
    if int(repeat_status) != 0 or not all(np.isfinite(error) and error < 1.e-9 for error in repeat_errors):
        raise RuntimeError("candidate differs between seed-origin and nearby certified initialization")
    candidate_gradient_check = {"ad": final_ad, "differences": final_differences, "accepted": True,
        "native_initialization_caches_cleared": True, "cold_warm_relative_error": repeat_error,
        "cold_warm_constraint_difference": constraint_error, "cold_warm_state_relative_difference": state_error,
        "post_fd_warm": dict(zip(("value_difference", "constraint_difference", "state_difference"), repeat_errors), status=int(repeat_status), accepted=True)}
    if not accepted_iterates or not np.array_equal(y, accepted_iterates[-1]["y"]):
        accepted_iterates.append(dict(history[-1], iteration=len(accepted_iterates),
                                     phase="provided boundary" if VALIDATE_BOUNDARY else "final result"))
candidate = materialize(np.asarray(jnp.asarray(x0).at[active].add(step * y)))
anchor_checks = materialize.diagnostics if COULOMB else None
# Release coarse solve buffers and executables before increasing the resolution.
if COULOMB:
    materialize.clear_initializer()
evaluate = problem = materialize = objectives = None
jax.clear_caches()
gc.collect()

### Refine matched endpoints, including surfaces outside the objective #########
verification, records = terms(CHECK_SURFACES, verify=True), {}
refined_materialize = None
for label, equilibrium in (("seed", seed), ("candidate", candidate)):
    resolved = equilibrium.inp.change_resolution(mpol=FINAL_MPOL, ntor=FINAL_MPOL,
                                                ntheta=4 * FINAL_MPOL, nzeta=4 * FINAL_MPOL)
    refined_input = replace(resolved, ns_array=np.array([FINAL_NS]),
                            ftol_array=np.array([FINAL_FTOL]), niter_array=np.array([9000]))
    refined = opt.solve_equilibrium(refined_input, restart_from=equilibrium.wout, raise_on_max_iterations=True)
    if COULOMB:
        if refined_materialize is None:
            refined_problem = opt.VmecProblem.from_tuples(
                refined_input, [(lambda state, runtime: jnp.array(0.), 0., 1.)], max_mode=MAX_MODE,
                vary_major_radius=False, restart_from=refined, forward_ftol=FINAL_FTOL,
                forward_max_iterations=9000, adjoint_tol=ADJOINT_TOL, refine_tol=REFINE_TOL,
                device="cpu", progress=False, evaluation_progress=False)
            refined_materialize = common_gauge(refined_problem)
        refined = refined_materialize(refined_problem.x_from_input(refined_input))
    responses = [term.compute_state(refined.state, refined.runtime) for term in verification]
    if not all(np.all(response["valid"]) for response in responses):
        raise RuntimeError(f"{label}: invalid refined kinetic geometry")
    records[label] = {"constraints": np.asarray(constraints(refined.state, refined.runtime)).tolist(),
        "profiles": [{key: np.asarray(response[key]).tolist()
                   for key in ("lower", "upper", "relative_gap", "scalar_upper", "geometry_fit_error_sampled") if key in response}
                  for response in responses]}
    vj.write_wout(OUTPUT / f"wout_optimization_{label}.nc", refined.wout)
    if label == "candidate":
        refined.inp.to_indata(OUTPUT / "input.QA_heat_bound_optimized")
delta = np.asarray(records["candidate"]["constraints"]) / records["seed"]["constraints"] - 1
comparisons = [{"upper_ratio": np.asarray(c["upper"]) / s["upper"],
                   "separated_decrease": np.maximum(0., 1 - np.asarray(c["upper"]) / np.maximum(s["lower"], 1.e-300))}
               for s, c in zip(records["seed"]["profiles"], records["candidate"]["profiles"])]
optimized = np.array([CHECK_SURFACES.index(surface) for surface in SURFACES])
worst_ratio = max(float(np.max(row["upper_ratio"][optimized])) for row in comparisons)
constraints_valid = bool(np.all(np.isfinite(delta)) and np.max(abs(delta)) <= CONSTRAINT_TOLERANCE)
records.update(model=MODEL, smoke=ci_smoke, physical_profiles=PROFILES if COULOMB else None,
    equilibrium_gauge="fixed smooth boundary-following m=1 prescription with its full implicit derivative" if COULOMB else "host-solve path",
    equilibrium_anchor_checks=anchor_checks, refinement_anchor_checks=refined_materialize.diagnostics if COULOMB else None,
    source="thermal entropy response at independent pressure force A1=0", units="m^2/s", surfaces=CHECK_SURFACES,
    macro_order=MACRO_ORDER, gradient_check={"ad": ad, "differences": differences}, constraints_relative=delta,
    optimizer={"method": "supplied boundary" if VALIDATE_BOUNDARY else "SLSQP" if COULOMB else "augmented Lagrangian / L-BFGS-B",
        "success": None if VALIDATE_BOUNDARY else bool(result.success),
        "objective": "smooth maximum of dimensionless upper-bound ratios", "reference_upper_m2_s": references,
        "iterations": int(iterations), "message": str(result.message), "initial_loss": seed_value, "history": history,
        "accepted_iterates": accepted_iterates, "candidate_gradient_check": candidate_gradient_check,
        "rejected_trials": rejections if COULOMB else None, "incumbent_retained": incumbent_retained,
        "iteration_scope": "accepted gradient points" if rejections.get("consecutive_at_stop") else "optimizer result",
        "constraint_gradient_check": constraint_gradient_check},
    settings={"input": str(INPUT_FILE), "forward_ftol": FORWARD_FTOL, "adjoint_tol": ADJOINT_TOL,
              "refine_tol": REFINE_TOL, "geometry_order": GEOMETRY_ORDER, "verify_geometry_order": VERIFY_GEOMETRY_ORDER,
              "ns": NS, "mpol": MPOL, "mboz": MBOZ, "final_ns": FINAL_NS, "final_ftol": FINAL_FTOL,
              "final_mpol": FINAL_MPOL, "final_mboz": FINAL_MBOZ, "max_mode": MAX_MODE,
              "parameter_step_m": PARAMETER_STEP, "max_parameter_change_steps": MAX_PARAMETER_CHANGE,
              "constraint_tolerance": CONSTRAINT_TOLERANCE, "working_constraint_tolerance": CONSTRAINT_TOLERANCE / 2,
              "supplied_boundary": VALIDATE_BOUNDARY, "smoothing": SMOOTHING},
    optimized_surfaces=SURFACES, worst_objective_upper_ratio=worst_ratio, constraints_valid=constraints_valid,
    accepted=bool((not COULOMB or candidate_gradient_check and candidate_gradient_check.get("accepted") is True)
                  and constraints_valid and np.isfinite(worst_ratio) and worst_ratio < 1.),
    acceptance_scope="refined constraints and upper objectives; transport decrease requires separated bounds or independent kinetics",
    comparisons=comparisons)
(OUTPUT / "neoclassical_heat_bound_optimization.json").write_text(
    json.dumps(records, indent=2, allow_nan=False, default=lambda value: np.asarray(value).tolist()) + "\n")
print(f"Accepted: {records['accepted']}; constraints valid: {constraints_valid}; objective upper ratio: {worst_ratio:.9g}")
print(f"Refined constraint changes: {delta}; separate radial validation: {comparisons}")
if not records["accepted"] and not ci_smoke:
    raise RuntimeError("refined constraints or objective improvement failed; inspect the saved result")
