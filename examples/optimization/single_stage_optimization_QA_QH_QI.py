#!/usr/bin/env python
"""Optimize a quasisymmetric or quasi-isodynamic boundary and its coils together (VMEX + ESSOS).

Set ``FAMILY`` to ``"QA"``, ``"QH"`` or ``"QI"``. Stage one finds a vacuum boundary
of that family with a mode ladder, as in ``QA_optimization.py``; set
``STAGE_ONE_INPUT`` to start from that script's ``input.*_optimized`` instead. The
coils are then fitted to the frozen boundary without an equilibrium solve, and the
boundary and coils are optimized as one vector. VMEX returns the exact derivative
of the converged equilibrium; ESSOS and JAX differentiate the coils. The
quasisymmetry residual (the constructed QI residual, with mirror-ratio and
elongation hinges, for QI) is the objective. The transform floor, the aspect limit,
the normal field B.n/|B| and the field-strength match |B_coil|^2/|B|^2 = 1 are
inequality constraints in an augmented Lagrangian, so no weight needs tuning per
seed; the coil length, curvature and distance limits are one-sided penalties. The
design is then re-solved on a finer grid and cold-started as a free-boundary
equilibrium in the coil field.
"""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize

import vmex as vj
from vmex import optimize as opt
from vmex.core.qi import ConstructedQIResidual
from vmex.core.single_stage import (
    CoilPenalty, SingleStageProblem, coils_around_boundary, design_target_report,
    fit_vacuum_coils, save_outputs, verify_design)

FAMILY = "QA"  # "QA", "QH" or "QI"
OUTPUT_NAME = f"single_stage_{FAMILY}"
STAGE_ONE_INPUT = None  # replaces stage one with an existing input.*_optimized file
DATA = Path(__file__).resolve().parents[1] / "data"
STAGE_NS, STAGE_FTOL, STAGE_NITER = 31, 1.0e-12, 5500  # radial grid of every trial
STAGE_ONE_MAX_NFEV = [10, 15]
STAGE_STEP, STAGE_TRUST = 0.02, 5.0  # scaled step of stage one; trust bound in steps
CURVATURE_WEIGHT = 100.0
CONSTRAINT_MARGIN = 1.15  # the optimizer enforces coil limits 15 % tighter than the check
COIL_FIT_MAXITER = 400
JUMP_CONSTRAINT = 0.04  # RMS of |B_coil|^2/|B|^2 - 1 on the boundary
N_SEGMENTS, STELLSYM = 64, True

# Augmented Lagrangian: outer multiplier updates, each an L-BFGS-B solve.
OUTER_ITERATIONS, MAXITER, MAX_TRIALS = 5, 10, 25
PENALTY, PARAMETER_BOUND = 100.0, 3.0
NPHI, NTHETA = 37, 32  # odd toroidal count avoids aliasing the coil field
FREE_MPOL = 8  # mode numbers of the free-boundary check; the coil field excites higher harmonics

# Per family: seed deck, stage-one ladder, final limits (the optimizer is handed slightly tighter
# values) and coils, in units of the plasma minor radius a; lengths in units of 2 pi a.
SURFACES = np.linspace(0.1, 1.0, 10)
STAGE_ONE_MAX_MODES, MAX_MODE = [1, 2], 2
STAGE_ONE_IOTA_HINGE = True
if FAMILY == "QA":
    INPUT_FILE = DATA / "input.minimal_seed_nfp2"
    SEED_PERTURBATION = 0.05  # rotating-ellipse amplitude: a circular torus has no first-order iota
    HELICITY_N = 0
    # B.n/|B| at 0.2 % (limit 0.002, constraint 0.0016, JUMP_CONSTRAINT 0.01, N_COILS 4,
    # COIL_ORDER 6) cut the free-boundary alpha loss from 11 % to 4 % and its eps_eff^(3/2)
    # twentyfold, at a 30 % higher maximum coil curvature.
    ASPECT_LIMIT, IOTA_FLOOR, NORMAL_FIELD_LIMIT = 6.0, 0.42, 0.01
    ASPECT_CONSTRAINT, IOTA_CONSTRAINT, NORMAL_FIELD_CONSTRAINT = 5.9, 0.43, 0.008
    N_COILS, COIL_ORDER = 3, 4
    COIL_OFFSET_A, COIL_SCALE_A = 2.0, 0.3  # starting distance from the boundary; step per scaled variable
    LENGTH_OVER_2PI_A = 5.0
    COIL_SURFACE_DISTANCE_A, COIL_DISTANCE_A, CURVATURE_A = 0.9, 1.0, 1.15
elif FAMILY == "QH":
    INPUT_FILE = DATA / "input.minimal_seed_nfp4"
    SEED_PERTURBATION = 0.12  # rotating-ellipse amplitude that starts stage one in the QH basin
    HELICITY_N = -1
    STAGE_NS, STAGE_FTOL, STAGE_NITER = 35, 1.0e-13, 4000
    STAGE_ONE_IOTA_HINGE = False
    ASPECT_LIMIT, IOTA_FLOOR, NORMAL_FIELD_LIMIT = 6.0, 1.0, 0.01
    ASPECT_CONSTRAINT, IOTA_CONSTRAINT, NORMAL_FIELD_CONSTRAINT = 5.9, 1.05, 0.008
    N_COILS, COIL_ORDER = 3, 4
    COIL_OFFSET_A, COIL_SCALE_A = 1.5, 0.3
    LENGTH_OVER_2PI_A = 6.0
    COIL_SURFACE_DISTANCE_A, COIL_DISTANCE_A, CURVATURE_A = 0.75, 0.5, 1.5
elif FAMILY == "QI":
    INPUT_FILE = DATA / "input.QI_nfp2_initial"
    SEED_PERTURBATION = None
    SURFACES = np.linspace(0.1, 1.0, 6)
    QI_OPTIONS = dict(mboz=12, nboz=12, nphi=61, nalpha=18, n_bounce=21)
    STAGE_ONE_MAX_MODES, MAX_MODE = [1, 3], 3
    STAGE_STEP, STAGE_TRUST = 1.0, np.inf
    MIRROR_LIMIT, ELONGATION_LIMIT, ASPECT_TARGET = 0.21, 8.0, 8.0  # stage-one aspect target
    ASPECT_LIMIT, IOTA_FLOOR, NORMAL_FIELD_LIMIT = 8.5, 0.5, 0.012
    ASPECT_CONSTRAINT, IOTA_CONSTRAINT, NORMAL_FIELD_CONSTRAINT = 8.3, 0.51, 0.0096
    N_COILS, COIL_ORDER, N_SEGMENTS = 5, 8, 96
    COIL_OFFSET_A, COIL_SCALE_A = 1.5, 0.3
    LENGTH_OVER_2PI_A = 9.0
    COIL_SURFACE_DISTANCE_A, COIL_DISTANCE_A, CURVATURE_A = 0.9, 0.6, 1.6
else:
    raise ValueError(f"FAMILY must be QA, QH or QI, not {FAMILY!r}")

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    STAGE_ONE_MAX_MODES, STAGE_ONE_MAX_NFEV = [1], [3]
    OUTER_ITERATIONS, MAXITER, MAX_TRIALS, COIL_FIT_MAXITER = 1, 2, 4, 2
    N_SEGMENTS, COIL_ORDER, NPHI, NTHETA = 24, 2, 8, 8

###############################################################################
# End of input parameters.
###############################################################################

# Stage one: a mode ladder on the family's residual, the aspect ratio and the shape limits.
mpol = max(MAX_MODE + 2, 5)
if STAGE_ONE_INPUT is None:
    inp = vj.VmecInput.from_file(INPUT_FILE)
    if SEED_PERTURBATION:
        rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
        rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
        inp = replace(inp, rbc=rbc, zbs=zbs)
else:
    inp = vj.VmecInput.from_file(STAGE_ONE_INPUT)
inp = replace(inp, delt=0.5, ns_array=np.array([STAGE_NS]), ftol_array=np.array([STAGE_FTOL]),
              niter_array=np.array([STAGE_NITER])).change_resolution(
    mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
iota_hinge = [(opt.Hinge(opt.min_abs_iota, floor=IOTA_CONSTRAINT), 0.0, 10.0)]
if FAMILY == "QI":
    qs = ConstructedQIResidual(SURFACES, **QI_OPTIONS)
    shape_limits = [(opt.Hinge(opt.mirror_ratio, ceiling=0.99 * MIRROR_LIMIT), 0.0, 1000.0),
                    (opt.Hinge(opt.max_elongation, ceiling=ELONGATION_LIMIT), 0.0, 10.0)]
    stage_terms = [(qs, 0.0, 10.0), (opt.aspect_ratio, ASPECT_TARGET, 0.01), *iota_hinge, *shape_limits]
else:
    qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=HELICITY_N)
    shape_limits = []
    stage_terms = [(qs, 0.0, 1.0), (opt.aspect_ratio, ASPECT_CONSTRAINT, 1.0),
                   *(iota_hinge if STAGE_ONE_IOTA_HINGE else [])]
if STAGE_ONE_INPUT is None:
    stage_one = opt.VmecProblem.from_tuples(
        inp, stage_terms, max_mode=MAX_MODE, use_ess=True, progress=not ci_smoke)
    x = stage_one.x0
    for max_mode, max_nfev in zip(STAGE_ONE_MAX_MODES, STAGE_ONE_MAX_NFEV):
        sub = stage_one.subproblem(max_mode=max_mode, x=x)
        step = STAGE_STEP * sub.scales
        fit = least_squares(sub.residual, sub.x0, jac=sub.residual_jac, x_scale=step,
                            bounds=(sub.x0 - STAGE_TRUST * step, sub.x0 + STAGE_TRUST * step),
                            max_nfev=max_nfev, ftol=1e-6, xtol=1e-10, verbose=2)
        x = sub.embed(fit.x)
    inp = stage_one.input_from_x(x)

# The single-stage problem: the residual is scaled to cost 1 on the stage-one boundary.
qs_cost = 0.5 * float(qs.total(opt.solve_equilibrium(inp)))
plasma = opt.VmecProblem.from_tuples(
    inp, [(qs.residuals_state, 0.0, 1 / qs_cost), *shape_limits], max_mode=MAX_MODE, use_ess=True,
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
    (f"{FAMILY} total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
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

equilibrium, surface, metrics, free = verify_design(
    joint, u, prefix=OUTPUT_NAME, ns=31 if ci_smoke else 101, ftol=1e-10 if ci_smoke else 1e-13,
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
save_outputs(joint, u, equilibrium, surface, prefix=OUTPUT_NAME)
if verdict["unmet"] and not ci_smoke:
    raise SystemExit(1)
