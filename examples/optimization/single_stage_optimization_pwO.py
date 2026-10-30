#!/usr/bin/env python
"""Joint mapped-field and coil optimization at beta=0.5%.

Run pwO_optimization.py, then pwO_optimization_finite_beta.py first. Coils and
boundary vary together; toroidal flux and field normalization stay fixed.
The short default run tests the workflow, not a qualified coil design: its
transport and held-boundary checks must be read before using the result.
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
from vmex.core.single_stage import (
    CoilPenalty, SingleStageProblem, hinge, normal_field_rms, toroidal_flux,
    verify_design, save_outputs, design_target_report,
)
from vmex.core.pwo import MappedPwOResidual
from vmex.core.neoclassical import screen_transport, transport_improves
from vmex._compat import require_optional
from essos.coils import Coils, CreateEquallySpacedCurves
from essos.surfaces import surfacerzfourier_from_boundary

# Input parameters.
NFP = 4
INPUT_FILE = Path("low_transport_finite_beta/input.trial")
MAX_MODE = 2
SURFACES = (.25, 16.5 / 34, .75)
# Same fixed calibration as the vacuum and pressure continuations.
B_REF = np.array([2.7123510440502594, 2.699696279828112, 2.6832357706580665])
B_MAX_REF = np.array([2.9284215849767965, 2.979692353278536, 3.008858221344288])
RAW_SCALE = .004813522516264787
TARGET_BETA = 0.005
PRESSURE_PROFILE = [1.0, -1.0]
BETA_TOLERANCE = 0.1

# Final limits; optimization constraints are tighter to allow for finer checks.
IOTA_FLOOR, ASPECT_LIMIT, NORMAL_FIELD_LIMIT = 0.41, 10.1, 0.01
COIL_SURFACE_DISTANCE_LIMIT, COIL_DISTANCE_LIMIT, CURVATURE_LIMIT = 0.15, 0.17, 7.0
DELBSQ_LIMIT, FREE_BOUNDARY_DISTANCE_LIMIT = 0.02, 0.02
IOTA_CONSTRAINT, ASPECT_CONSTRAINT, NORMAL_FIELD_CONSTRAINT = 0.43, 10.0, 0.008
CURVATURE_OBJECTIVE_LIMIT = 6.9
COIL_DISTANCE_CONSTRAINT, COIL_SURFACE_DISTANCE_CONSTRAINT = 0.19, 0.16

N_COILS, COIL_ORDER, N_SEGMENTS, STELLSYM = 2, 3, 48, True
COIL_MAJOR_RADIUS, COIL_MINOR_RADIUS, COIL_CURRENT = 1.0, 0.6, 2.7e5
LENGTH_TARGET, LENGTH_WEIGHT, CURVATURE_WEIGHT = 5.5, 0.5, 10.0
COIL_DISTANCE_WEIGHT, COIL_SURFACE_DISTANCE_WEIGHT = 1.0e4, 1.0e4
CONSTRAINT_WEIGHT, REJECTED_TRIAL_COST, PARAMETER_BOUND = 1.0e3, 10.0, 3.0
MAXITER, MAX_TRIALS, COIL_FIT_MAXITER = 3, 12, 30
NPHI, NTHETA, VC_DIGITS = 17, 16, 4  # Odd toroidal count avoids aliasing the two-coil field.
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
if (inp.nfp != NFP or inp.lasym or inp.lfreeb or inp.phiedge != .083
        or inp.rbc[inp.ntor, 0] != 1 or inp.curtor != 0 or inp.ncurr != 1):
    raise ValueError("Use the finite-beta candidate at its original flux and length scale")
mpol = max(MAX_MODE + 2, 5)
am = np.zeros(21)
am[:len(PRESSURE_PROFILE)] = PRESSURE_PROFILE
inp = replace(inp, delt=0.5, ns_array=np.array([35]),
              ftol_array=np.array([1e-12]), niter_array=np.array([2000]), pmass_type="power_series", am=am,
              pres_scale=1.0, ncurr=1, curtor=0.0, ac=np.zeros(21)).change_resolution(
    mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
surface_seed = surfacerzfourier_from_boundary(
    jnp.asarray(inp.rbc), jnp.asarray(inp.zbs), NFP, nphi=NPHI, ntheta=NTHETA)
curves = CreateEquallySpacedCurves(
    N_COILS, COIL_ORDER, COIL_MAJOR_RADIUS, COIL_MINOR_RADIUS,
    n_segments=N_SEGMENTS, nfp=NFP, stellsym=STELLSYM)
coils_circular = Coils(curves, np.full(N_COILS, COIL_CURRENT))
x_coils_circular = np.asarray(curves.dofs).ravel()
coil_scales = np.full(x_coils_circular.size, 0.05)
coil_penalty = CoilPenalty(
    LENGTH_TARGET, CURVATURE_OBJECTIVE_LIMIT, COIL_DISTANCE_CONSTRAINT,
    COIL_SURFACE_DISTANCE_CONSTRAINT, LENGTH_WEIGHT, CURVATURE_WEIGHT,
    COIL_DISTANCE_WEIGHT, COIL_SURFACE_DISTANCE_WEIGHT)


def coils_from_dofs(x):
    return coils_circular.with_dofs(jnp.concatenate((x, coils_circular.dofs_currents)))


def coil_fit_objective(u):
    coils = coils_from_dofs(jnp.asarray(x_coils_circular) + coil_scales * u)
    constraint = 1 - normal_field_rms(coils, surface_seed) / NORMAL_FIELD_CONSTRAINT
    return jnp.sum(coil_penalty(coils, surface_seed)) + hinge(constraint[None], CONSTRAINT_WEIGHT)


coil_fit_vg = jax.jit(jax.value_and_grad(coil_fit_objective))
coil_fit = minimize(
    lambda u: tuple(map(np.asarray, coil_fit_vg(jnp.asarray(u)))),
    np.zeros_like(x_coils_circular), jac=True, method="L-BFGS-B",
    bounds=[(-PARAMETER_BOUND, PARAMETER_BOUND)] * x_coils_circular.size,
    options={"maxiter": COIL_FIT_MAXITER, "maxcor": 20, "ftol": 1e-15, "gtol": 1e-10})
coils0 = coils_from_dofs(jnp.asarray(x_coils_circular + coil_scales * coil_fit.x))
print(f"[coil fit] {coil_fit.nit} L-BFGS-B iterations, no equilibrium solves: B.n/B RMS "
      f"= {100 * float(normal_field_rms(coils0, surface_seed)):.3f}% on the seed")

# Rescale currents to preserve the candidate's flux and B_REF, then calibrate pressure.
fitted_flux = toroidal_flux(inp, coils0)
if not np.isfinite(fitted_flux) or fitted_flux * inp.phiedge <= 0:
    raise ValueError("Coil flux must be finite and have the same sign as PHIEDGE")
coils_circular = Coils(curves, np.full(N_COILS, COIL_CURRENT * inp.phiedge / fitted_flux))
coils0 = coils_from_dofs(jnp.asarray(x_coils_circular + coil_scales * coil_fit.x))
beta_at_unit_pressure = float(opt.solve_equilibrium(inp).wout.betatotal)
inp = replace(inp, pres_scale=TARGET_BETA / beta_at_unit_pressure)
print(f"[flux] PHIEDGE = {inp.phiedge:.6f} Wb from the fitted coils; "
      f"PRES_SCALE = {inp.pres_scale:.1f} Pa for beta {100 * TARGET_BETA:.2f}%")


def relative_beta(state, runtime):
    return opt.volume_average_beta(state, runtime) / TARGET_BETA


mapped = MappedPwOResidual(SURFACES, b_ref=B_REF, mboz=8, nboz=8, oversample=2)
plasma_problem = opt.VmecProblem.from_tuples(
    inp, [(mapped.residuals_state, 0.0, 2 / RAW_SCALE), (relative_beta, 1.0, 1.0)], max_mode=MAX_MODE,
    use_ess=True, progress=not ci_smoke, forward_ftol=1e-12,
    forward_max_iterations=2000, refine_tol=1e-12, adjoint_tol=1e-8, max_fsq_ratio=1.)
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
before = screen_transport(joint.seed_equilibrium.wout, SURFACES)
after = screen_transport(coarse.wout, SURFACES)
physics = mapped.physical_metrics(coarse, bmax_ref=1.1*B_MAX_REF,
                                  aspect_bounds=(9.9, 10.1), iota_floor=IOTA_FLOOR)
print(f"NEO mean epsilon_eff^(3/2): {before.mean():.6g} -> {after.mean():.6g}")
summary = {
    "example": Path(__file__).name, "smoke": ci_smoke,
    "optimization_seconds": round(optimization_seconds, 1),
    "coil_fit_iterations": int(coil_fit.nit), **joint.counts,
    "lbfgsb_iterations": int(result.nit),
    "equilibrium_solves": joint.monitor.records[-1].equilibrium_solves,
    "phiedge": float(inp.phiedge), "pres_scale": float(inp.pres_scale),
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
