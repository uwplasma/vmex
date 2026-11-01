#!/usr/bin/env python
"""Optimize a finite-beta pwO plasma and ESSOS coils together.

Run ``pwO_optimization.py`` and ``pwO_optimization_finite_beta.py`` first.
COILS_INPUT accepts an ESSOS stage-2 fit; COIL_FIT_MAXITER=0 keeps its shapes.
A cold free-boundary solve and NEO check the design. Requires vmex[neoclassical].
"""

import gc
import os
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize, OptimizeResult

import vmex as vj
from vmex import optimize as opt
from vmex.core.single_stage import (
    CoilPenalty, SingleStageProblem, coil_field, common_current_scale, toroidal_flux,
    verify_design, save_outputs, design_target_report,
)
from vmex.core import virtual_casing as vc
from vmex.core.restart import restart_state
from vmex.core.pwo import MappedPwOResidual
from vmex.core.neoclassical import screen_transport
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.surfaces import surfacerzfourier_from_boundary

# Input parameters.
NFP = 2
INPUT_FILE = Path(f"input.pwO_finite_beta_nfp{NFP}")
WOUT_FILE = Path(f"wout_pwO_finite_beta_nfp{NFP}.nc")
COILS_INPUT = None  # Optional Path to an ESSOS stage-2 coil JSON for this input.
MAX_MODE = 2
SURFACES = np.linspace(.25, .75, 3)
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
OPTIMIZER_FTOL = 1e-8
NPHI, NTHETA, VC_DIGITS = 17, 24, 4  # Odd toroidal count avoids aliasing the two-coil field.
MAKE_MOVIE, MOVIE_SURFACE_COLOR = False, "absB"

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAXITER, MAX_TRIALS, COIL_FIT_MAXITER = 2, 4, 2
    N_SEGMENTS, COIL_ORDER, NPHI, NTHETA = 24, 2, 8, 8

if COIL_FIT_MAXITER == 0 and COILS_INPUT is None:
    raise ValueError("Skipping the prefit requires COILS_INPUT")

# Seed and coil pre-fit.
inp = vj.VmecInput.from_file(INPUT_FILE)
mpol = max(MAX_MODE + 2, 5)
am = np.zeros(21)
am[:len(PRESSURE_PROFILE)] = PRESSURE_PROFILE
inp = replace(inp, delt=0.5, ns_array=np.array([35]),
              ftol_array=np.array([1e-12]), niter_array=np.array([4000]), pmass_type="power_series", am=am,
              pres_scale=1.0, ncurr=1, curtor=0.0, ac=np.zeros(21)).change_resolution(
    mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
# Keep the input total flux and physical pressure calibration fixed before
# constructing the finite-beta target. No equilibrium solve occurs in a coil trial.
previous = vj.read_wout(WOUT_FILE)
if TARGET_BETA:
    calibration = opt.solve_equilibrium(inp, initial_state=restart_state(previous, inp),
                                        raise_on_max_iterations=True)
    beta_at_unit_pressure = float(calibration.wout.betatotal)
    if not np.isfinite(beta_at_unit_pressure) or beta_at_unit_pressure <= 0:
        raise ValueError("The unit-pressure seed must have finite positive beta")
    inp = replace(inp, pres_scale=TARGET_BETA / beta_at_unit_pressure)
    initial_state = calibration.state
else:
    inp = replace(inp, pres_scale=0.)
    initial_state = restart_state(previous, inp)


def relative_beta(state, runtime):
    return opt.volume_average_beta(state, runtime) / TARGET_BETA


seed = opt.solve_equilibrium(inp, initial_state=initial_state, raise_on_max_iterations=True)
mapped = MappedPwOResidual(SURFACES, mboz=8, nboz=8, oversample=2)
initial_scale = float(mapped.total_state(seed.state, seed.runtime)) or 1.
plasma_terms = [(mapped.residuals_state, 0.0, 2 / initial_scale)]
if TARGET_BETA:
    plasma_terms.append((relative_beta, 1.0, 1.0))
plasma_problem = opt.VmecProblem.from_tuples(
    inp, plasma_terms, max_mode=MAX_MODE,
    use_ess=True, restart_from=seed, progress=not ci_smoke, forward_ftol=1e-12,
    forward_max_iterations=4000, refine_tol=1e-12, adjoint_tol=1e-8, max_fsq_ratio=1.)
seed_equilibrium = plasma_problem.equilibrium_from_x(plasma_problem.x0)
seed_data = vc.surface_field_data_from_state(
    inp, seed_equilibrium.solution, runtime=seed_equilibrium.solver_context,
    nphi=NPHI, ntheta=NTHETA)
if TARGET_BETA:
    seed_interface = vc.PlasmaVacuumInterface.from_surface_data(
        seed_data, digits=VC_DIGITS, precision=vc.plan_vc_precision(seed_data, digits=VC_DIGITS))
else:
    # Zero pressure and prescribed current mean no plasma field or VC solve.
    area = jnp.linalg.norm(seed_data.area_vector, axis=0)
    seed_interface = vc.PlasmaVacuumInterface(
        gamma=seed_data.gamma, normal=seed_data.normal, phi_grid=seed_data.phi,
        weights=area / jnp.sum(area), B_plasma=jnp.zeros_like(seed_data.B_total),
        Bn_plasma=jnp.zeros_like(area), Bin_mag2=jnp.sum(seed_data.B_total**2, axis=0),
        p_edge=0., nfp=NFP)
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
coils_initial = Coils(curves, np.full(N_COILS, COIL_CURRENT))
if COILS_INPUT is not None:
    coils_initial = Coils.from_json(str(COILS_INPUT))
    if coils_initial.nfp != NFP or coils_initial.stellsym != STELLSYM:
        raise ValueError("Seed coils must match the equilibrium field periods and stellarator symmetry")
    if not np.all(np.isfinite(np.asarray(coils_initial.dofs))):
        raise ValueError("Seed coil DOFs must be finite")
    N_COILS = coils_initial.dofs_currents.size
    N_SEGMENTS = coils_initial.n_segments
x_coils_initial = np.asarray(coils_initial.dofs_curves).ravel()
n_coil_shape = x_coils_initial.size
coil_scales = np.full(n_coil_shape, .5 * a)
coil_penalty = CoilPenalty(
    LENGTH_TARGET, CURVATURE_OBJECTIVE_LIMIT, COIL_DISTANCE_CONSTRAINT,
    COIL_SURFACE_DISTANCE_CONSTRAINT, LENGTH_WEIGHT, CURVATURE_WEIGHT,
    COIL_DISTANCE_WEIGHT, COIL_SURFACE_DISTANCE_WEIGHT,
    normalized=True, geometry_segments=3*N_SEGMENTS,
    surface_grid=(NFP * NPHI, NTHETA if ci_smoke else 32))


def coils_from_u(u):
    shape = jnp.asarray(x_coils_initial) + coil_scales * u[:n_coil_shape]
    relative_current = u[n_coil_shape:] - jnp.mean(u[n_coil_shape:])
    currents = coils_initial.dofs_currents * jnp.exp(.25 * relative_current)
    return coils_initial.with_dofs(jnp.concatenate((shape, currents)))


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
print(f"[coil fit] {N_COILS} unique coils, at most {COIL_FIT_MAXITER} iterations", flush=True)
coil_fit = (OptimizeResult(x=np.zeros(n_coil_shape + N_COILS), nit=0) if COIL_FIT_MAXITER == 0 else minimize(
    lambda u, value_and_grad=coil_fit_vg: tuple(map(np.asarray, value_and_grad(jnp.asarray(u)))),
    np.zeros(n_coil_shape + N_COILS), jac=True, method="L-BFGS-B",
    callback=opt.OptimizationMonitor(print_every=25),
    bounds=[(-COIL_FIT_PARAMETER_BOUND, COIL_FIT_PARAMETER_BOUND)] * (n_coil_shape + N_COILS),
    options={"maxiter": COIL_FIT_MAXITER, "maxcor": 20, "ftol": 1e-15, "gtol": 1e-10}))
coils0 = coils_from_u(jnp.asarray(coil_fit.x))
current_factor, fitted_rms, fitted_pressure = coil_fit_field(coils0)
current_factor = float(current_factor)
if not np.isfinite(current_factor) or current_factor == 0:
    raise ValueError("The external boundary field must admit a finite nonzero current fit")
coils0 = coils0.with_dofs(jnp.concatenate((coils0.dofs_curves.ravel(),
                                         coils0.dofs_currents * current_factor)))
print(f"[coil fit] {coil_fit.nit} iterations, frozen target: total B.n/B RMS "
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
    ("pwO variance", mapped.total, ".6e"), ("beta", opt.volume_average_beta, ".4%"),
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
             "maxls": 20, "ftol": OPTIMIZER_FTOL, "gtol": 1e-8})
u = result.x
print(f"[solve] {result.nit} L-BFGS-B iterations, {joint.counts['trials']} trials "
      f"({joint.counts['rejected']} rejected), status {result.status}: {result.message}", flush=True)

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

# Matched NEO: the seed on the verification grid, the same half-mesh surfaces.
half_mesh = (np.arange(100) + 0.5) / 100
neo_surfaces = half_mesh[np.abs(half_mesh[:, None] - SURFACES).argmin(axis=0)]
seed_fine = opt.solve_equilibrium(
    replace(inp, ns_array=np.array([101]), ftol_array=np.array([1e-12]),
            niter_array=np.array([20000])),
    initial_state=seed.solution, verbose=False, raise_on_max_iterations=True)
if not ci_smoke:
    print(f"epsilon_eff^(3/2) at s = {neo_surfaces}: "
          f"seed {screen_transport(seed_fine.wout, neo_surfaces)}, "
          f"final {screen_transport(equilibrium.wout, neo_surfaces)}")
save_outputs(joint, u, equilibrium, surface, prefix=prefix,
             make_movie=MAKE_MOVIE, surface_color=MOVIE_SURFACE_COLOR)
if verdict["unmet"] and not ci_smoke:
    raise SystemExit(1)
