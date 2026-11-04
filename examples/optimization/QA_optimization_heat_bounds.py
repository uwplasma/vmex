#!/usr/bin/env python
"""Optimize a QA boundary at fixed physical heat-transport operating points.

VMEX_HEAT_BOUND_MODEL=coulomb selects full Landau collisions, parallel/mirror
orbits and incompressible ExB on a fitted Boozer surface. The default Lorentz
reference retains bounce geometry and prescribed tangential precession.
An equilibrium adjoint drives bounded, augmented-Lagrangian optimization;
independent equilibrium and geometry refinement checks the endpoints. These
are bounds for the stated kinetic models; see docs/explanation/confinement.rst.
"""

import gc
import json
import os
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
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
SURFACES, CHECK_SURFACES = (.5,), ((.25, .5, .75) if COULOMB else (.5,))
PROFILES = ((100., 1.5e20, 15., 100.), (1000., 1.e20, 17., 1000.), (10000., 1.e20, 19., 10000.))
if os.environ.get("VMEX_HEAT_BOUND_INCLUDE_HOT", "1") == "0":
    PROFILES = PROFILES[:2]
MASS_KG, ELEMENTARY_CHARGE, EPSILON0 = 3.3435837724e-27, 1.602176634e-19, 8.8541878128e-12
MACRO_ORDER = os.environ.get("VMEX_HEAT_BOUND_MACRO_ORDER")
COLLISION_FREQUENCIES, LORENTZ_FIELDS = (100., 1000.), (-250., 0., 250.)

# Fix flux and RBC(0,0); preserve aspect, volume, and iota at s=.25,.5,.75:
MAX_MODE = int(os.environ.get("VMEX_HEAT_BOUND_MAX_MODE", "4" if COULOMB else "1"))
MAXITER = int(os.environ.get("VMEX_HEAT_BOUND_MAXITER", "120" if COULOMB else "8"))
PARAMETER_STEP, MAX_PARAMETER_CHANGE = (.05 if COULOMB else .002), 5.
CONSTRAINT_TOLERANCE, SMOOTHING = .001, .02
NS, MPOL, MBOZ = (26, 8, 12) if COULOMB else (14, 6, 6)
FORWARD_FTOL, ADJOINT_TOL, REFINE_TOL = (1.e-16, 1.e-9, 1.e-12) if COULOMB else (1.e-11, 1.e-6, 1.e-10)
GEOMETRY_ORDER, VERIFY_GEOMETRY_ORDER = (8, 10), (16, 20)
FINAL_NS, FINAL_FTOL = 50, 1.e-13
FINAL_MPOL, FINAL_MBOZ = (10, 16) if COULOMB else (8, 10)
PITCH_CELLS, ENERGY_POINTS, VERIFY_PITCH_CELLS, VERIFY_ENERGY_POINTS = 32, 6, 96, 12
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
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
    """Hold the released m=1 family fixed in values, adjoints, and exports."""
    from vmex.core import implicit as imp
    from vmex.core.solver import evaluate_forces

    cfg = problem.metadata["config"]
    params0 = imp.params_from_input(problem.metadata["input"], device=cfg.device)
    def params_of(x):
        return replace(params0, **dict(zip(("rbc", "zbs", "rbs", "zbc"), problem.boundary_from_x(x))))

    frozen, mask = imp.solve_implicit_with_aux(params_of(jnp.asarray(problem.x0)), cfg)
    project, edge = imp._dof_projector(cfg, mask), imp._edge_mask(cfg)
    specification = (jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), frozen), jax.ShapeDtypeStruct((), jnp.int32))
    memo = {}

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
            initial = imp._assemble(project(state), imp.runtime_from_params(params, cfg), frozen, project, edge)
            state = imp._refined_state(cfg, params, initial, mask, krylov=False)
            residual = imp._anchor_residual(cfg, params, state, mask)
            if not np.isfinite(residual) or residual > 1.e-10:
                status = 3
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
        return imp._solve_implicit_status_bwd(cfg, saved, (cotangent[0], None, None, None))[:1]

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
history, multipliers, penalty, y = [], np.zeros(5), 100., np.zeros(len(active))


def value_and_gradient(y):
    """One equilibrium adjoint supplies the scaled boundary gradient."""
    (value, (_, costs, status)), (gradient, delta) = evaluate(y, multipliers, penalty)
    history.append({"value": float(value), "heat": float(costs[0]), "constraints": np.asarray(delta).tolist(), "status": int(status)})
    print(f"heat={float(costs[0]):.9g}, constraint={np.max(np.abs(delta)):.3g}, status={int(status)}", flush=True)
    return float(value), np.asarray(gradient)


### Check the derivative and optimize #########################################
seed_value, seed_gradient = value_and_gradient(y)
if history[-1]["status"] != 0 or not np.all(np.isfinite(seed_gradient)) or not np.isfinite(seed_value):
    raise RuntimeError("invalid seed equilibrium value or gradient")
direction = np.sin(np.arange(len(y)) + 1.)
direction /= np.linalg.norm(direction)
ad, differences = float(seed_gradient @ direction), []
# Reuse the executable; endpoint gradients are discarded, not used as FD data.
for h in ((.0004, .00008, .000016) if COULOMB else (.002,)):
    plus, ap = evaluate(y + h * direction, multipliers, penalty)[0]
    minus, am = evaluate(y - h * direction, multipliers, penalty)[0]
    if int(ap[-1]) != 0 or int(am[-1]) != 0 or not np.all(np.isfinite([plus, minus, ad])):
        raise RuntimeError("invalid finite-difference equilibrium or value")
    fd = float((plus - minus) / (2 * h))
    differences.append({"step": h, "finite_difference": fd, "relative_error": abs(ad - fd) / max(abs(ad), abs(fd), 1.e-14)})
if not np.isfinite(differences[-1]["relative_error"]) or differences[-1]["relative_error"] > (.002 if COULOMB else .03):
    raise RuntimeError(f"equilibrium derivative check failed: {differences}")
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
candidate = materialize(np.asarray(jnp.asarray(x0).at[active].add(step * y)))
# Release the optimizer executables before increasing the equilibrium resolution.
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
    equilibrium_gauge="fixed constrained family at each resolution" if COULOMB else "host-solve path",
    source="thermal entropy response at independent pressure force A1=0", units="m^2/s", surfaces=CHECK_SURFACES,
    macro_order=MACRO_ORDER, gradient_check={"ad": ad, "differences": differences}, constraints_relative=delta,
    optimizer={"iterations": int(iterations), "message": str(result.message), "initial_loss": seed_value, "history": history},
    settings={"input": str(INPUT_FILE), "forward_ftol": FORWARD_FTOL, "adjoint_tol": ADJOINT_TOL,
              "refine_tol": REFINE_TOL, "geometry_order": GEOMETRY_ORDER, "verify_geometry_order": VERIFY_GEOMETRY_ORDER,
              "ns": NS, "mpol": MPOL, "mboz": MBOZ, "final_ns": FINAL_NS, "final_ftol": FINAL_FTOL,
              "final_mpol": FINAL_MPOL, "final_mboz": FINAL_MBOZ, "max_mode": MAX_MODE,
              "parameter_step_m": PARAMETER_STEP, "max_parameter_change_steps": MAX_PARAMETER_CHANGE,
              "constraint_tolerance": CONSTRAINT_TOLERANCE, "smoothing": SMOOTHING},
    optimized_surfaces=SURFACES, worst_objective_upper_ratio=worst_ratio, constraints_valid=constraints_valid,
    accepted=bool(constraints_valid and np.isfinite(worst_ratio) and worst_ratio < 1.),
    acceptance_scope="refined constraints and upper objectives; transport decrease requires separated bounds or independent kinetics",
    comparisons=comparisons)
(OUTPUT / "neoclassical_heat_bound_optimization.json").write_text(
    json.dumps(records, indent=2, allow_nan=False, default=lambda value: np.asarray(value).tolist()) + "\n")
print(f"Accepted: {records['accepted']}; constraints valid: {constraints_valid}; objective upper ratio: {worst_ratio:.9g}")
print(f"Refined constraint changes: {delta}; separate radial validation: {comparisons}")
if not records["accepted"] and not ci_smoke:
    raise RuntimeError("refined constraints or objective improvement failed; inspect the saved result")
