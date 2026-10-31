#!/usr/bin/env python
"""Joint mapped-field and coil optimization at beta=0.5%.

Run pwO_optimization.py, then pwO_optimization_finite_beta.py first. Coils and
boundary vary together; VMEC total flux and physical field targets stay fixed.
The finite-beta prefit matches the frozen total normal-field target and fits
common current to B_VMEC minus B_plasma, without equating coil flux to PHIEDGE.
The short default run tests the workflow, not a qualified coil design: its
transport and held-boundary checks must be read before using the result.
"""

import gc
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
from vmex.core.single_stage import (
    CoilPenalty, SingleStageProblem, coil_field, common_current_scale, toroidal_flux,
    verify_design, save_outputs, design_target_report,
)
from vmex.core import virtual_casing as vc
from vmex.core.restart import restart_state
from vmex.core.pwo import MappedPwOResidual
from vmex.core.neoclassical import screen_transport, transport_improves
from vmex._compat import require_optional
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.surfaces import surfacerzfourier_from_boundary

# Input parameters.
NFP = 2
INPUT_FILE = Path("low_transport_finite_beta/input.trial")
MAX_MODE = 2
SURFACES = jnp.linspace(.25, .75, 3)
MIRROR_LIMIT = .2
TARGET_BETA = 0.005
PRESSURE_PROFILE = [1.0, -1.0]
BETA_TOLERANCE = 0.1

# Final limits; optimization constraints are tighter to allow for finer checks.
IOTA_FLOOR, ASPECT_LIMIT, NORMAL_FIELD_LIMIT = 0.41, 6.1, 0.01
COIL_SURFACE_DISTANCE_A, COIL_DISTANCE_A, CURVATURE_A = .5, .5, .7
DELBSQ_LIMIT, FREE_BOUNDARY_DISTANCE_LIMIT = 0.02, 0.02
IOTA_CONSTRAINT, ASPECT_CONSTRAINT, NORMAL_FIELD_CONSTRAINT = 0.43, 6.0, 0.008
CURVATURE_OBJECTIVE_A = .69
COIL_DISTANCE_OBJECTIVE_A, COIL_SURFACE_DISTANCE_OBJECTIVE_A = .7, .6

N_COILS, COIL_ORDER, N_SEGMENTS, STELLSYM = 3, 4, 64, True
COIL_MAJOR_RADIUS, COIL_CURRENT = 1.0, 2.7e5
COIL_RADIUS_A, LENGTH_OVER_2PI_A = 2.7, 7.0
LENGTH_WEIGHT, CURVATURE_WEIGHT = 10., 100.
COIL_DISTANCE_WEIGHT, COIL_SURFACE_DISTANCE_WEIGHT = 100., 100.
COIL_FIT_PARAMETER_BOUND = 4.0
CONSTRAINT_WEIGHT, REJECTED_TRIAL_COST, PARAMETER_BOUND = 1.0e3, 10.0, 3.0
MAXITER, MAX_TRIALS, COIL_FIT_MAXITER = 3, 12, 250
NPHI, NTHETA, VC_DIGITS = 17, 24, 4  # Odd toroidal count avoids aliasing the two-coil field.
MAKE_MOVIE, MOVIE_SURFACE_COLOR = False, "absB"

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAXITER, MAX_TRIALS, COIL_FIT_MAXITER = 2, 4, 2
    N_SEGMENTS, COIL_ORDER, NPHI, NTHETA = 24, 2, 8, 8

# Seed and coil pre-fit.
vj.configure()
require_optional("neo-jax", "low-transport NEO screen")
started = time.perf_counter()
print("Running single_stage_optimization_pwO.py")
inp = vj.VmecInput.from_file(INPUT_FILE)
if (inp.nfp != NFP or inp.lasym or inp.lfreeb or inp.phiedge <= 0
        or inp.rbc[inp.ntor, 0] != 1 or inp.curtor != 0 or inp.ncurr != 1):
    raise ValueError("Use the finite-beta candidate at its original flux and length scale")
mpol = max(MAX_MODE + 2, 5)
am = np.zeros(21)
am[:len(PRESSURE_PROFILE)] = PRESSURE_PROFILE
inp = replace(inp, delt=0.5, ns_array=np.array([35]),
              ftol_array=np.array([1e-12]), niter_array=np.array([2000]), pmass_type="power_series", am=am,
              pres_scale=1.0, ncurr=1, curtor=0.0, ac=np.zeros(21)).change_resolution(
    mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
# Keep the input total flux and physical pressure calibration fixed before
# constructing the finite-beta target. No equilibrium solve occurs in a coil trial.
previous = vj.read_wout(INPUT_FILE.with_name("wout_trial.nc"))
calibration = opt.solve_equilibrium(inp, initial_state=restart_state(previous, inp),
                                    raise_on_max_iterations=True)
beta_at_unit_pressure = float(calibration.wout.betatotal)
if not np.isfinite(beta_at_unit_pressure) or beta_at_unit_pressure <= 0:
    raise ValueError("The unit-pressure seed must have finite positive beta")
inp = replace(inp, pres_scale=TARGET_BETA / beta_at_unit_pressure)


def relative_beta(state, runtime):
    return opt.volume_average_beta(state, runtime) / TARGET_BETA


seed = opt.solve_equilibrium(inp, initial_state=calibration.state, raise_on_max_iterations=True)
mapped = MappedPwOResidual(SURFACES, mboz=8, nboz=8, oversample=2)
initial_scale = float(mapped.total_state(seed.state, seed.runtime)) or 1.
actual_surfaces = np.asarray(mapped.compute_state(seed.state, seed.runtime)["s_b"])
plasma_problem = opt.VmecProblem.from_tuples(
    inp, [(mapped.residuals_state, 0.0, 2 / initial_scale), (relative_beta, 1.0, 1.0)], max_mode=MAX_MODE,
    use_ess=True, restart_from=seed, progress=not ci_smoke, forward_ftol=1e-12,
    forward_max_iterations=2000, refine_tol=1e-12, adjoint_tol=1e-8, max_fsq_ratio=1.)
seed_equilibrium = plasma_problem.equilibrium_from_x(plasma_problem.x0)
seed_data = vc.surface_field_data_from_state(
    inp, seed_equilibrium.solution, runtime=seed_equilibrium.solver_context,
    nphi=NPHI, ntheta=NTHETA)
seed_interface = vc.PlasmaVacuumInterface.from_surface_data(
    seed_data, digits=VC_DIGITS, precision=vc.plan_vc_precision(seed_data, digits=VC_DIGITS))
external_target_B = seed_data.B_total - seed_interface.B_plasma
seed_Bmag = jnp.linalg.norm(seed_data.B_total, axis=0)

# Stage-2 choices are physical ratios, not inherited absolute A10 dimensions.
a = float(seed_equilibrium.wout.Aminor_p)
COIL_MINOR_RADIUS, LENGTH_TARGET = COIL_RADIUS_A * a, LENGTH_OVER_2PI_A * 2 * np.pi * a
COIL_SURFACE_DISTANCE_LIMIT, COIL_DISTANCE_LIMIT = COIL_SURFACE_DISTANCE_A * a, COIL_DISTANCE_A * a
CURVATURE_LIMIT, CURVATURE_OBJECTIVE_LIMIT = CURVATURE_A / a, CURVATURE_OBJECTIVE_A / a
COIL_DISTANCE_CONSTRAINT = COIL_DISTANCE_OBJECTIVE_A * a
COIL_SURFACE_DISTANCE_CONSTRAINT = COIL_SURFACE_DISTANCE_OBJECTIVE_A * a

surface_seed = surfacerzfourier_from_boundary(
    jnp.asarray(inp.rbc), jnp.asarray(inp.zbs), NFP, nphi=NPHI, ntheta=NTHETA)
curves = CreateEquallySpacedCurves(
    N_COILS, COIL_ORDER, COIL_MAJOR_RADIUS, COIL_MINOR_RADIUS,
    n_segments=N_SEGMENTS, nfp=NFP, stellsym=STELLSYM)
coils_circular = Coils(curves, np.full(N_COILS, COIL_CURRENT))
x_coils_circular = np.asarray(curves.dofs).ravel()
n_coil_shape = x_coils_circular.size
coil_scales = np.full(n_coil_shape, .5 * a)
coil_penalty = CoilPenalty(
    LENGTH_TARGET, CURVATURE_OBJECTIVE_LIMIT, COIL_DISTANCE_CONSTRAINT,
    COIL_SURFACE_DISTANCE_CONSTRAINT, LENGTH_WEIGHT, CURVATURE_WEIGHT,
    COIL_DISTANCE_WEIGHT, COIL_SURFACE_DISTANCE_WEIGHT,
    normalized=True, geometry_segments=3*N_SEGMENTS,
    surface_grid=(NFP * NPHI, NTHETA if ci_smoke else 32))


def coils_from_u(u):
    shape = jnp.asarray(x_coils_circular) + coil_scales * u[:n_coil_shape]
    relative_current = u[n_coil_shape:] - jnp.mean(u[n_coil_shape:])
    currents = coils_circular.dofs_currents * jnp.exp(.25 * relative_current)
    return coils_circular.with_dofs(jnp.concatenate((shape, currents)))


def coil_fit_field(coils):
    # One Biot-Savart evaluation; linearity supplies the rescaled field.
    field = seed_interface.external_B(coil_field(coils))
    factor = common_current_scale(field, external_target_B, seed_interface.weights)
    normal = (factor * jnp.sum(field * seed_interface.normal, axis=0)
              + seed_interface.Bn_plasma) / seed_Bmag
    pressure = seed_interface.pressure_balance_residual(
        lambda points: jnp.moveaxis(factor*field, 0, -1)) / seed_Bmag**2
    return (factor, jnp.sqrt(jnp.sum(seed_interface.weights * normal**2)),
            jnp.sqrt(jnp.sum(seed_interface.weights * pressure**2)))


def coil_fit_objective(u):
    coils = coils_from_u(u)
    _, rms, pressure = coil_fit_field(coils)
    return jnp.sum(coil_penalty(coils, surface_seed)) + .5 * (
        (rms/NORMAL_FIELD_CONSTRAINT)**2 + (pressure/(.9*DELBSQ_LIMIT))**2)


coil_fit_vg = jax.jit(jax.value_and_grad(coil_fit_objective))
coil_fit = minimize(
    lambda u: tuple(map(np.asarray, coil_fit_vg(jnp.asarray(u)))),
    np.zeros(n_coil_shape + N_COILS), jac=True, method="L-BFGS-B",
    bounds=[(-COIL_FIT_PARAMETER_BOUND, COIL_FIT_PARAMETER_BOUND)] * (n_coil_shape + N_COILS),
    options={"maxiter": COIL_FIT_MAXITER, "maxcor": 20, "ftol": 1e-15, "gtol": 1e-10})
coils0 = coils_from_u(jnp.asarray(coil_fit.x))
current_factor, fitted_rms, fitted_pressure = coil_fit_field(coils0)
current_factor = float(current_factor)
if not np.isfinite(current_factor) or current_factor == 0:
    raise ValueError("The external boundary field must admit a finite nonzero current fit")
coils0 = coils0.with_dofs(jnp.concatenate((coils0.dofs_curves.ravel(),
                                         coils0.dofs_currents * current_factor)))
print(f"[coil fit] {coil_fit.nit} iterations, frozen finite-beta target: total B.n/B RMS "
      f"= {100 * float(fitted_rms):.3f}%; common-current boundary-vector fit {current_factor:.6g}")
print(f"[field calibration] total PHIEDGE = {inp.phiedge:.6f} Wb; "
      f"PRES_SCALE = {inp.pres_scale:.1f} Pa for beta {100 * TARGET_BETA:.2f}%. "
      "Boundary-vector current calibration is not an exact external-flux constraint.")

seed_coil_flux = toroidal_flux(inp, coils0)
seed_pressure_jump = seed_interface.pressure_balance_residual(coil_field(coils0)) / seed_Bmag**2
seed_pressure_jump_rms = float(jnp.sqrt(jnp.sum(seed_interface.weights * seed_pressure_jump**2)))
print(f"[coil fit] normalized pressure-jump RMS {seed_pressure_jump_rms:.3%}")
coils0.to_json("single_stage_pwO_prefit_coils.json")
if not ci_smoke and (fitted_rms > NORMAL_FIELD_LIMIT or seed_pressure_jump_rms > DELBSQ_LIMIT):
    raise SystemExit("Stage-2 field matching failed; improve the coil fit before joint optimization")

# Release the one-time prefit executables before the joint adjoint compilation.
del coil_fit_vg
jax.clear_caches()
gc.collect()
joint = SingleStageProblem(
    plasma_problem, coils0, coil_penalty, nphi=NPHI, ntheta=NTHETA,
    iota_floor=IOTA_CONSTRAINT, aspect_limit=ASPECT_CONSTRAINT,
    normal_field_limit=NORMAL_FIELD_CONSTRAINT, constraint_weight=CONSTRAINT_WEIGHT,
    vc_digits=VC_DIGITS, rejected_trial_cost=REJECTED_TRIAL_COST)
report = opt.EquilibriumReporter(
    ("mapped variance", mapped.total, ".6e"), ("beta", opt.volume_average_beta, ".4%"),
    ("aspect", opt.aspect_ratio, ".4f"), ("mean iota", opt.mean_iota, ".4f"),
    ("min |iota|", opt.min_abs_iota, ".4f"))
seed_values = report("seed", joint.seed_equilibrium)

# Optimization: dimensionless increments, one equilibrium/VC/adjoint per trial.
u = np.zeros_like(joint.x0)
vj.FunctionProblem.from_functions(u, value_and_grad=joint.value_and_grad).compile_value_and_gradient(
    report_interval=10.0)
initial_value = joint.monitor.records[0].cost
result = minimize(
    joint.value_and_grad, u, jac=True, method="L-BFGS-B", callback=joint.monitor,
    bounds=[(-PARAMETER_BOUND, PARAMETER_BOUND)] * u.size,
    options={"maxiter": MAXITER, "maxfun": MAX_TRIALS, "maxcor": 20,
             "maxls": 20, "ftol": 1e-3, "gtol": 1e-8})
u = result.x
optimization_seconds = time.perf_counter() - started
print(f"[solve] {result.nit} L-BFGS-B iterations, {joint.counts['trials']} trials "
      f"({joint.counts['rejected']} rejected), status {result.status}: {result.message}", flush=True)

# Keep the coarse equilibrium for a matched NEO comparison after fine-grid checks.
x_final = joint.x0 + joint.scales * u
coarse = plasma_problem.equilibrium_from_x(x_final[:joint.n_boundary])

# Independent verification and final reporting.
prefix = "single_stage_pwO"
equilibrium, surface, metrics, free = verify_design(
    joint, u, prefix=prefix, ns=31 if ci_smoke else 101,
    ftol=1e-10 if ci_smoke else 1e-12, free_ns=[16] if ci_smoke else [16, 51],
    free_ftol=1e-10 if ci_smoke else 1e-12, verbose=not ci_smoke)
final_values = report("final", equilibrium)
print(f"Objective: {initial_value:.6e} -> {result.fun:.6e}")
print(f"Coil lengths = {np.asarray(joint.objects_from_x(joint.x0 + joint.scales * u)[1].length[:N_COILS])}")
for name, value in metrics.items():
    print(f"{name} = {value:.6g}")

beta_bounds = TARGET_BETA * (1 - BETA_TOLERANCE), TARGET_BETA * (1 + BETA_TOLERANCE)
verdict = design_target_report(
    metrics, free, iota_floor=IOTA_FLOOR, aspect_limit=ASPECT_LIMIT,
    normal_field_limit=NORMAL_FIELD_LIMIT, clearance=COIL_SURFACE_DISTANCE_LIMIT,
    coil_distance=COIL_DISTANCE_LIMIT, curvature=CURVATURE_LIMIT, beta_bounds=beta_bounds,
    delbsq_limit=DELBSQ_LIMIT, boundary_distance_limit=FREE_BOUNDARY_DISTANCE_LIMIT)
print("[verify] matched NEO comparison", flush=True)
before = screen_transport(joint.seed_equilibrium.wout, actual_surfaces)
after = screen_transport(coarse.wout, actual_surfaces)
physics = mapped.physical_metrics(coarse, mirror_ceiling=MIRROR_LIMIT,
                                  aspect_bounds=(5.9, 6.1), iota_floor=IOTA_FLOOR)
print(f"NEO mean epsilon_eff^(3/2): {before.mean():.6g} -> {after.mean():.6g}")
summary = {
    "example": Path(__file__).name, "smoke": ci_smoke,
    "optimization_seconds": round(optimization_seconds, 1),
    "coil_fit_iterations": int(coil_fit.nit), **joint.counts,
    "lbfgsb_iterations": int(result.nit),
    "equilibrium_solves": joint.monitor.records[-1].equilibrium_solves,
    "phiedge": float(inp.phiedge), "pres_scale": float(inp.pres_scale),
    "coil_current_calibration": "external_boundary_vector_least_squares",
    "coil_current_factor": current_factor,
    "external_flux_constrained": False,
    "seed_coil_toroidal_flux": seed_coil_flux,
    "seed_normalized_pressure_jump_rms": seed_pressure_jump_rms,
    "free_boundary_verification": free, "seed": seed_values,
    "final": {**final_values, **metrics},
    **verdict, "particle_qualified": False, "coarse_physics": physics,
    "baseline_Q": before.tolist(), "trial_Q": after.tolist(),
    "transport_screen_passed": transport_improves(before, after),
    "design_qualified": verdict["met"] and physics["feasible"] and transport_improves(before, after),
}
Path("single_stage_optimization_pwO_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
save_outputs(joint, u, equilibrium, surface, prefix=prefix,
             make_movie=MAKE_MOVIE, surface_color=MOVIE_SURFACE_COLOR)
if not summary["design_qualified"] and not ci_smoke:
    raise SystemExit(1)
