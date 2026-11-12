#!/usr/bin/env python
"""Optimize a quasi-axisymmetric boundary and its coils together (VMEX + ESSOS).

Stage one finds a vacuum quasi-axisymmetric boundary with a mode ladder, as in
``QA_optimization.py``; set ``STAGE_ONE_INPUT`` to start from that script's
``input.QA_optimized`` instead. The coils are then fitted to the frozen boundary
without an equilibrium solve, and the boundary and coils are optimized as one
vector. VMEX returns the exact derivative of the converged equilibrium; ESSOS and
JAX differentiate the coils. The quasisymmetry residual is the objective. The
transform floor, the aspect limit, the normal field B.n/|B| and the field-strength
match |B_coil|^2/|B|^2 = 1 are inequality constraints in an augmented Lagrangian,
so no weight needs tuning per seed; the coil length, curvature and distance
limits are one-sided penalties. The design is then re-solved on a finer grid
and cold-started as a free-boundary equilibrium in the coil field.
"""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize

import vmex as vj
from vmex import optimize as opt
from vmex.core.single_stage import (
    CoilPenalty, SingleStageProblem, coils_around_boundary, design_target_report,
    fit_vacuum_coils, save_outputs, verify_design)

# Seed deck and field periods; STAGE_ONE_INPUT, if set, replaces stage one.
NFP = 2
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"
STAGE_ONE_INPUT = None
STAGE_NS, STAGE_FTOL, STAGE_NITER = 31, 1.0e-12, 5500  # radial grid of every trial
SEED_PERTURBATION = 0.05  # rotating-ellipse amplitude: a circular torus has no first-order iota
SURFACES = np.linspace(0.1, 1.0, 10)
STAGE_ONE_MAX_MODES, STAGE_ONE_MAX_NFEV = [1, 2], [10, 15]
MAX_MODE = 2

# Final limits. The optimizer is handed slightly tighter values.
ASPECT_LIMIT, IOTA_FLOOR, NORMAL_FIELD_LIMIT = 6.0, 0.42, 0.01
ASPECT_CONSTRAINT, IOTA_CONSTRAINT, NORMAL_FIELD_CONSTRAINT = 5.9, 0.43, 0.008
JUMP_CONSTRAINT = 0.04  # RMS of |B_coil|^2/|B|^2 - 1 on the boundary

# Coils, in units of the plasma minor radius a. Lengths are in units of 2 pi a.
N_COILS, COIL_ORDER, N_SEGMENTS, STELLSYM = 3, 4, 64, True
COIL_OFFSET_A, COIL_SCALE_A = 2.0, 0.3  # starting distance from the boundary; step per scaled variable
LENGTH_OVER_2PI_A = 5.0
COIL_SURFACE_DISTANCE_A, COIL_DISTANCE_A, CURVATURE_A = 0.9, 1.0, 1.15
CURVATURE_WEIGHT = 100.0
CONSTRAINT_MARGIN = 1.15  # the optimizer enforces coil limits 15 % tighter than the check
COIL_FIT_MAXITER = 400

# Augmented Lagrangian: outer multiplier updates, each an L-BFGS-B solve.
OUTER_ITERATIONS, MAXITER, MAX_TRIALS = 5, 10, 25
PENALTY, PARAMETER_BOUND = 100.0, 3.0
NPHI, NTHETA = 37, 32  # odd toroidal count avoids aliasing the coil field
FREE_MPOL = 8  # mode numbers of the free-boundary check; the coil field excites higher harmonics

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    STAGE_ONE_MAX_MODES, STAGE_ONE_MAX_NFEV = [1], [3]
    OUTER_ITERATIONS, MAXITER, MAX_TRIALS, COIL_FIT_MAXITER = 1, 2, 4, 2
    N_SEGMENTS, COIL_ORDER, NPHI, NTHETA = 24, 2, 8, 8

###############################################################################
# End of input parameters.
###############################################################################

# Stage one: a mode ladder on the quasi-axisymmetric residual, the aspect ratio and the transform floor.
mpol = max(MAX_MODE + 2, 5)
if STAGE_ONE_INPUT is None:
    inp = vj.VmecInput.from_file(INPUT_FILE)
    rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
    rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
    inp = replace(inp, rbc=rbc, zbs=zbs)
else:
    inp = vj.VmecInput.from_file(STAGE_ONE_INPUT)
inp = replace(inp, delt=0.5, ns_array=np.array([STAGE_NS]), ftol_array=np.array([STAGE_FTOL]),
              niter_array=np.array([STAGE_NITER])).change_resolution(
    mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
if STAGE_ONE_INPUT is None:
    stage_one = opt.VmecProblem.from_tuples(
        inp, [(qs, 0.0, 1.0), (opt.aspect_ratio, ASPECT_CONSTRAINT, 1.0),
                    (opt.Hinge(opt.min_abs_iota, floor=IOTA_CONSTRAINT), 0.0, 10.0)],
        max_mode=MAX_MODE, use_ess=True, progress=not ci_smoke)
    x = stage_one.x0
    for max_mode, max_nfev in zip(STAGE_ONE_MAX_MODES, STAGE_ONE_MAX_NFEV):
        sub = stage_one.subproblem(max_mode=max_mode, x=x)
        step = 0.02 * sub.scales
        fit = least_squares(sub.residual, sub.x0, jac=sub.residual_jac, x_scale=step,
                            bounds=(sub.x0 - 5 * step, sub.x0 + 5 * step),
                            max_nfev=max_nfev, ftol=1e-6, xtol=1e-10, verbose=2)
        x = sub.embed(fit.x)
    inp = stage_one.input_from_x(x)

# The single-stage problem: the residual is scaled to cost 1 on the stage-one boundary.
qs_cost = 0.5 * float(qs.total(opt.solve_equilibrium(inp)))
plasma = opt.VmecProblem.from_tuples(
    inp, [(qs.residuals_state, 0.0, 1 / qs_cost)], max_mode=MAX_MODE, use_ess=True,
    progress=not ci_smoke)
seed_equilibrium = plasma.equilibrium_from_x(plasma.x0)
a = float(seed_equilibrium.wout.Aminor_p)

# Coils start as offset cross-sections of the boundary and are fitted to it.
penalty = CoilPenalty(
    LENGTH_OVER_2PI_A * 2 * np.pi * a / CONSTRAINT_MARGIN, CURVATURE_A / a / CONSTRAINT_MARGIN,
    COIL_DISTANCE_A * a * CONSTRAINT_MARGIN, COIL_SURFACE_DISTANCE_A * a * CONSTRAINT_MARGIN,
    curvature_weight=CURVATURE_WEIGHT, normalized=True, geometry_segments=3 * N_SEGMENTS)
coils = fit_vacuum_coils(
    inp, seed_equilibrium,
    coils_around_boundary(inp, N_COILS, COIL_ORDER, COIL_OFFSET_A * a,
                          n_segments=N_SEGMENTS, stellsym=STELLSYM),
    penalty, nphi=NPHI, ntheta=NTHETA, normal_field_limit=NORMAL_FIELD_CONSTRAINT,
    jump_limit=JUMP_CONSTRAINT, maxiter=COIL_FIT_MAXITER, scale=COIL_SCALE_A * a)
joint = SingleStageProblem(
    plasma, coils, penalty, nphi=NPHI, ntheta=NTHETA, iota_floor=IOTA_CONSTRAINT,
    aspect_limit=ASPECT_CONSTRAINT, normal_field_limit=NORMAL_FIELD_CONSTRAINT,
    jump_limit=JUMP_CONSTRAINT, constraint_weight=PENALTY, coil_scale=COIL_SCALE_A * a)
report = opt.EquilibriumReporter(
    ("QA total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"), ("min |iota|", opt.min_abs_iota, ".4f"))
report("stage one", joint.seed_equilibrium)

# Joint optimization: L-BFGS-B on the boundary and coil variables, then a multiplier step.
u = np.zeros_like(joint.x0)
for outer in range(OUTER_ITERATIONS):
    result = minimize(
        joint.value_and_grad, u, jac=True, method="L-BFGS-B", callback=joint.monitor,
        bounds=[(-PARAMETER_BOUND, PARAMETER_BOUND)] * u.size,
        options={"maxiter": MAXITER, "maxfun": MAX_TRIALS, "maxcor": 20, "ftol": 1e-10, "gtol": 1e-8})
    u = result.x
    constraints = joint.update_multipliers(u)
    print(f"[augmented Lagrangian {outer + 1}] {result.nit} iterations; constraints c >= 0: "
          + ", ".join(f"{name} {'inactive' if c is None else f'{c:.3g}'}" for name, c in constraints.items())
          + f"; multipliers {np.round(joint.multipliers, 3)}", flush=True)
print(f"[solve] {joint.counts['trials']} trials ({joint.counts['rejected']} rejected)", flush=True)

prefix = "single_stage_QA"
equilibrium, surface, metrics, free = verify_design(
    joint, u, prefix=prefix, ns=31 if ci_smoke else 101, ftol=1e-10 if ci_smoke else 1e-12,
    free_ns=[16] if ci_smoke else [16, 51], free_ftol=1e-10 if ci_smoke else 1e-12,
    free_mpol=None if ci_smoke else FREE_MPOL, verbose=not ci_smoke)
report("final", equilibrium)
final_coils = joint.objects_from_x(joint.x0 + joint.scales * u)[1]
print(f"Coil lengths = {np.asarray(final_coils.length[:N_COILS])} m, minor radius a = {a:.4f} m")
for name, value in metrics.items():
    print(f"{name} = {value:.6g}")
verdict = design_target_report(
    metrics, free, iota_floor=IOTA_FLOOR, aspect_limit=ASPECT_LIMIT,
    normal_field_limit=NORMAL_FIELD_LIMIT, clearance=COIL_SURFACE_DISTANCE_A * a,
    coil_distance=COIL_DISTANCE_A * a, curvature=CURVATURE_A / a,
    delbsq_limit=0.02, boundary_distance_limit=0.02)
save_outputs(joint, u, equilibrium, surface, prefix=prefix)
if verdict["unmet"] and not ci_smoke:
    raise SystemExit(1)
