#!/usr/bin/env python
"""Finite-beta QA boundary/coil optimization with VMEX, virtual casing, and ESSOS.

The circular vacuum seed chooses the positive total flux before pressure is
calibrated for 0.5% beta. The finite-beta coil prefit then matches the frozen
external boundary field, B_VMEC minus B_plasma, without identifying external
coil flux with total PHIEDGE. The pressure is proportional to 1-s and net
toroidal current is zero.
Each trial uses a fixed-boundary equilibrium and the total-field normal
residual; a finer fixed-boundary solve and a cold NESTOR solve independently
check the optimized design. VMEX_EXAMPLES_CI=1 runs a bounded smoke pass.
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
from vmex.core import virtual_casing as vc
from vmex.core.single_stage import (
    CoilPenalty, SingleStageProblem, coil_field, common_current_scale, hinge, toroidal_flux,
    verify_design, save_outputs, design_target_report,
)
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.surfaces import surfacerzfourier_from_boundary

# Input parameters.
NFP = 2
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"
MAX_MODE = 2
SURFACES = np.linspace(0.05, 1.0, 6)
SEED_MINOR_RADIUS, SEED_ELLIPSE = 0.195, 0.10
TARGET_BETA = 0.005
PRESSURE_PROFILE = [1.0, -1.0]
BETA_TOLERANCE = 0.1

# Final limits; optimization constraints are tighter to allow for finer checks.
IOTA_FLOOR, ASPECT_LIMIT, NORMAL_FIELD_LIMIT = 0.42, 6.0, 0.01
COIL_SURFACE_DISTANCE_LIMIT, COIL_DISTANCE_LIMIT, CURVATURE_LIMIT = 0.15, 0.17, 7.0
DELBSQ_LIMIT, FREE_BOUNDARY_DISTANCE_LIMIT = 0.02, 0.02
IOTA_CONSTRAINT, ASPECT_CONSTRAINT, NORMAL_FIELD_CONSTRAINT = 0.43, 5.97, 0.008
CURVATURE_OBJECTIVE_LIMIT = 6.9
COIL_DISTANCE_CONSTRAINT, COIL_SURFACE_DISTANCE_CONSTRAINT = 0.19, 0.16

N_COILS, COIL_ORDER, N_SEGMENTS, STELLSYM = 3, 4, 64, True
COIL_MAJOR_RADIUS, COIL_MINOR_RADIUS, COIL_CURRENT = 1.0, 0.5, 2.7e5
LENGTH_TARGET, LENGTH_WEIGHT, CURVATURE_WEIGHT = 5.5, 0.5, 10.0
COIL_DISTANCE_WEIGHT, COIL_SURFACE_DISTANCE_WEIGHT = 1.0e4, 1.0e4
CONSTRAINT_WEIGHT, REJECTED_TRIAL_COST, PARAMETER_BOUND = 1.0e3, 10.0, 3.0
MAXITER, MAX_TRIALS, COIL_FIT_MAXITER = 12, 35, 200
NPHI, NTHETA, VC_DIGITS = 37, 32, 4  # 37 avoids aliasing the three-coil field.
MAKE_MOVIE, MOVIE_SURFACE_COLOR = False, "absB"

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAXITER, MAX_TRIALS, COIL_FIT_MAXITER = 2, 4, 2
    N_SEGMENTS, COIL_ORDER, NPHI, NTHETA = 24, 2, 8, 8

# Seed and coil pre-fit.
started = time.perf_counter()
print("Running single_stage_optimization_finite_beta.py")
inp = vj.VmecInput.from_file(INPUT_FILE)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor, 1] = zbs[inp.ntor, 1] = SEED_MINOR_RADIUS
rbc[inp.ntor + 1, 1], zbs[inp.ntor + 1, 1] = SEED_ELLIPSE, -SEED_ELLIPSE
mpol = max(MAX_MODE + 2, 5)
am = np.zeros(21)
am[:len(PRESSURE_PROFILE)] = PRESSURE_PROFILE
inp = replace(inp, rbc=rbc, zbs=zbs, delt=0.5, pmass_type="power_series", am=am,
              pres_scale=1.0, ncurr=1, curtor=0.0, ac=np.zeros(21)).change_resolution(
    mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
surface_seed = surfacerzfourier_from_boundary(
    jnp.asarray(inp.rbc), jnp.asarray(inp.zbs), NFP, nphi=NPHI, ntheta=NTHETA)
curves = CreateEquallySpacedCurves(
    N_COILS, COIL_ORDER, COIL_MAJOR_RADIUS, COIL_MINOR_RADIUS,
    n_segments=N_SEGMENTS, nfp=NFP, stellsym=STELLSYM)
coils_circular = Coils(curves, np.full(N_COILS, COIL_CURRENT))
# Choose the total flux/physical scale from a vacuum circular seed BEFORE
# pressure is introduced. It is not the final finite-beta external coil flux.
seed_total_flux = toroidal_flux(inp, coils_circular)
if not np.isfinite(seed_total_flux) or seed_total_flux <= 0:
    raise ValueError("The vacuum seed must have finite positive physical flux")
inp = replace(inp, phiedge=seed_total_flux)
beta_at_unit_pressure = float(opt.solve_equilibrium(inp).wout.betatotal)
if not np.isfinite(beta_at_unit_pressure) or beta_at_unit_pressure <= 0:
    raise ValueError("The unit-pressure seed must have finite positive beta")
inp = replace(inp, pres_scale=TARGET_BETA / beta_at_unit_pressure)


def relative_beta(state, runtime):
    return opt.volume_average_beta(state, runtime) / TARGET_BETA


qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
plasma_problem = opt.VmecProblem.from_tuples(
    inp, [(qs.residuals_state, 0.0, 1.0), (relative_beta, 1.0, 1.0)], max_mode=MAX_MODE,
    use_ess=True, progress=not ci_smoke)
seed_equilibrium = plasma_problem.equilibrium_from_x(plasma_problem.x0)
seed_data = vc.surface_field_data_from_state(
    inp, seed_equilibrium.solution, runtime=seed_equilibrium.solver_context,
    nphi=NPHI, ntheta=NTHETA)
seed_interface = vc.PlasmaVacuumInterface.from_surface_data(
    seed_data, digits=VC_DIGITS, precision=vc.plan_vc_precision(seed_data, digits=VC_DIGITS))
external_target_B = seed_data.B_total - seed_interface.B_plasma
seed_Bmag = jnp.linalg.norm(seed_data.B_total, axis=0)

x_coils_circular = np.asarray(curves.dofs).ravel()
coil_scales = np.full(x_coils_circular.size, 0.05)
coil_penalty = CoilPenalty(
    LENGTH_TARGET, CURVATURE_OBJECTIVE_LIMIT, COIL_DISTANCE_CONSTRAINT,
    COIL_SURFACE_DISTANCE_CONSTRAINT, LENGTH_WEIGHT, CURVATURE_WEIGHT,
    COIL_DISTANCE_WEIGHT, COIL_SURFACE_DISTANCE_WEIGHT)


def coils_from_dofs(x):
    return coils_circular.with_dofs(jnp.concatenate((x, coils_circular.dofs_currents)))


def coil_fit_field(coils):
    field = seed_interface.external_B(coil_field(coils))
    factor = common_current_scale(field, external_target_B, seed_interface.weights)
    normal = (factor * jnp.sum(field * seed_interface.normal, axis=0)
              + seed_interface.Bn_plasma) / seed_Bmag
    return factor, jnp.sqrt(jnp.sum(seed_interface.weights * normal**2))


def coil_fit_objective(u):
    coils = coils_from_dofs(jnp.asarray(x_coils_circular) + coil_scales * u)
    _, rms = coil_fit_field(coils)
    constraint = 1 - rms / NORMAL_FIELD_CONSTRAINT
    return jnp.sum(coil_penalty(coils, surface_seed)) + hinge(constraint[None], CONSTRAINT_WEIGHT)


coil_fit_vg = jax.jit(jax.value_and_grad(coil_fit_objective))
coil_fit = minimize(
    lambda u: tuple(map(np.asarray, coil_fit_vg(jnp.asarray(u)))),
    np.zeros_like(x_coils_circular), jac=True, method="L-BFGS-B",
    bounds=[(-PARAMETER_BOUND, PARAMETER_BOUND)] * x_coils_circular.size,
    options={"maxiter": COIL_FIT_MAXITER, "maxcor": 20, "ftol": 1e-15, "gtol": 1e-10})
coils0 = coils_from_dofs(jnp.asarray(x_coils_circular + coil_scales * coil_fit.x))
current_factor, fitted_rms = coil_fit_field(coils0)
current_factor = float(current_factor)
if not np.isfinite(current_factor) or current_factor == 0:
    raise ValueError("The external boundary field must admit a finite nonzero current fit")
coils0 = coils0.with_dofs(jnp.concatenate((coils0.dofs_curves.ravel(),
                                         coils0.dofs_currents * current_factor)))
print(f"[coil fit] {coil_fit.nit} iterations on a frozen finite-beta target: total B.n/B RMS "
      f"= {100 * float(fitted_rms):.3f}%; current boundary-vector fit {current_factor:.6g}")
print(f"[field calibration] total PHIEDGE = {inp.phiedge:.6f} Wb; "
      f"PRES_SCALE = {inp.pres_scale:.1f} Pa for beta {100 * TARGET_BETA:.2f}%. "
      "Boundary-vector current calibration is not an exact external-flux constraint.")
seed_coil_flux = toroidal_flux(inp, coils0)
seed_pressure_jump = seed_interface.pressure_balance_residual(coil_field(coils0)) / seed_Bmag**2
seed_pressure_jump_rms = float(jnp.sqrt(jnp.sum(seed_interface.weights * seed_pressure_jump**2)))

joint = SingleStageProblem(
    plasma_problem, coils0, coil_penalty, nphi=NPHI, ntheta=NTHETA,
    iota_floor=IOTA_CONSTRAINT, aspect_limit=ASPECT_CONSTRAINT,
    normal_field_limit=NORMAL_FIELD_CONSTRAINT, constraint_weight=CONSTRAINT_WEIGHT,
    vc_digits=VC_DIGITS, rejected_trial_cost=REJECTED_TRIAL_COST)
report = opt.EquilibriumReporter(
    ("QA total", qs.total, ".6e"), ("beta", opt.volume_average_beta, ".4%"),
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
             "maxls": 20, "ftol": 1e-12, "gtol": 1e-8})
u = result.x
optimization_seconds = time.perf_counter() - started
print(f"[solve] {result.nit} L-BFGS-B iterations, {joint.counts['trials']} trials "
      f"({joint.counts['rejected']} rejected), status {result.status}: {result.message}", flush=True)

# Independent verification and final reporting.
prefix = "single_stage_finite_beta"
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
summary = {
    "example": Path(__file__).name, "smoke": ci_smoke,
    "optimization_seconds": round(optimization_seconds, 1),
    "coil_fit_iterations": int(coil_fit.nit), **joint.counts,
    "lbfgsb_iterations": int(result.nit),
    "equilibrium_solves": joint.monitor.records[-1].equilibrium_solves,
    "phiedge": float(inp.phiedge), "pres_scale": float(inp.pres_scale),
    "coil_current_calibration": "external_boundary_vector_least_squares",
    "coil_current_factor": current_factor, "external_flux_constrained": False,
    "seed_coil_toroidal_flux": seed_coil_flux,
    "seed_normalized_pressure_jump_rms": seed_pressure_jump_rms,
    "free_boundary_verification": free, "seed": seed_values,
    "final": {**final_values, **metrics},
    **verdict,
}
Path("single_stage_optimization_finite_beta_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
save_outputs(joint, u, equilibrium, surface, prefix=prefix,
             make_movie=MAKE_MOVIE, surface_color=MOVIE_SURFACE_COLOR)
if verdict["unmet"] and not ci_smoke:
    raise SystemExit(1)
