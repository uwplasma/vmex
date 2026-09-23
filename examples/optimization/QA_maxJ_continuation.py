#!/usr/bin/env python
"""Walk a QA boundary into a favorable-J outer volume at finite beta.

Exact maximum-J is incompatible with quasisymmetry near the magnetic axis, so
this does not chase it there. A vacuum QA ladder shapes the boundary, the
Landreman-Buller-Drevlak profiles and a Picard loop make the current
self-consistent, and a second ladder then adds the maximum-J residual on the
outer surfaces only, where pressure can reverse the trapped-particle
precession.

One physical lambda must describe the same trapped particles on every radius
and field-line label, so the pitch grid is selected once and held fixed across
the maximum-J stages. The script reports the retained QA, the bootstrap
mismatch and the maximum-J fraction of the outer volume.
"""

import os
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.bootstrap import (ELEMENTARY_CHARGE, KineticProfiles,
                                 RedlBootstrapMismatch, self_consistent_bootstrap)
from vmex.core.maxj import MaximumJResidual, common_trapped_pitches_state

# Number of field periods, and the seed deck the boundary is shaped from:
NFP = 2
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"

# Rotating-ellipse amplitude added to the circular seed:
SEED_PERTURBATION = 0.05

# Flux surfaces the quasisymmetry and bootstrap residuals use, and the outer
# ones the maximum-J residual is evaluated on:
QA_SURFACES = np.linspace(0.1, 0.9, 8)
SURFACES = np.array([0.6, 0.7, 0.8, 0.9])

# Vacuum QA ladder, then the maximum-J ladder: highest boundary mode number
# varied in each stage, and the residual evaluations each stage may spend:
QA_MAX_MODES = [1, 2]
QA_MAX_NFEV = [10, 15]
MAXJ_MAX_MODES = [2]
MAXJ_MAX_NFEV = [10]

# Targets:
ASPECT_TARGET = 6.0
IOTA_FLOOR = 0.42                 # minimum |iota| over the profile
MAGNETIC_WELL_TARGET = 0.01
TARGET_BETA = 0.025
MAXJ_TARGET = -0.01

# Term weights. The beta residual is relative because its target is small:
BETA_WEIGHT = 1.0 / TARGET_BETA**2
MAXJ_WEIGHT = 100.0
QA_WEIGHT = 10.0
BOOTSTRAP_WEIGHT = 1.0
WELL_WEIGHT = 10.0

# Trapped-particle sampling: the well depths the common pitches are drawn
# from, and the bounce-action quadrature the J integral uses:
TRAPPING_DEPTHS = (0.4, 0.8)
ACTION_NALPHA = 7
ACTION_POINTS = 32
ACTION_PERIODS = 8
ACTION_MAX_WELLS = 20
ACTION_QUADRATURE = 24
ACTION_MBOZ = 10

# Picard loop that makes the seed current self-consistent:
PICARD_ITERATIONS = 8
PICARD_TOLERANCE = 1e-3
N_CURRENT_SPLINE = 8
REDL_N_LAMBDA = 32

# Step control. Boundary coefficients move in metres; the current dofs are
# dimensionless, so they carry their own optimizer scale:
PARAMETER_STEP = 0.02
CURRENT_PARAMETER_STEP = 0.05
MAX_PARAMETER_CHANGE = 5.0
STAGE_MAX_ITERATIONS = 3000       # forward solve cap inside an optimizer trial

# Equilibrium resolution: poloidal and toroidal mode numbers grow with the
# stage but never fall below MINIMUM_MPOL:
MINIMUM_MPOL = 5

# Verification solve of the optimized boundary:
FINAL_NS = 71
FINAL_FTOL = 1e-14
FINAL_NITER = 20000

# Every output file name contains this:
OUTPUT_NAME = "QA_maxJ_optimized"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs. It keeps the
# minimal-seed QA wiring above but swaps in a bundled self-consistent
# finite-beta state before the maximum-J stages, rather than spending minutes
# forming matched wells to test one AD step.
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
SMOKE_INPUT_FILE = (Path(__file__).resolve().parents[1] / "data"
                    / "input.LandremanPaul2021_QA_beta2p5_bootstrap")
if ci_smoke:
    QA_SURFACES, MINIMUM_MPOL = np.linspace(0.2, 0.8, 4), 3
    QA_MAX_MODES, QA_MAX_NFEV = [1], [2]
    MAXJ_MAX_MODES, MAXJ_MAX_NFEV = [1], [2]
    ACTION_QUADRATURE, REDL_N_LAMBDA = 16, 12
    STAGE_MAX_ITERATIONS = 100
    PARAMETER_STEP = 0.001
    FINAL_NS, FINAL_FTOL = 31, 1e-10

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

# VmecInput is frozen, so copy its arrays before shaping the seed boundary.
inp = vj.VmecInput.from_file(INPUT_FILE)
if ci_smoke:
    inp = replace(inp, ns_array=np.array([11]), ftol_array=np.array([1e-8]),
                  niter_array=np.array([1500]))
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
inp = replace(inp, rbc=rbc, zbs=zbs)
equilibrium = opt.solve_equilibrium(inp)

### Set up the objective ######################################################

def iota_floor(equilibrium_state, solver_context):
    """Hinge on the profile minimum of |iota|: a mean target can hide a near-zero surface.

    opt.mean_iota targets the average instead; opt.soft_min_abs_iota is the smooth minimum.
    """
    return jnp.maximum(
        IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)


# Each term is (function, target, weight).
qs = opt.QuasisymmetryRatioResidual(QA_SURFACES, helicity_m=1, helicity_n=0)
shape_terms = [(qs, 0.0, QA_WEIGHT), (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (iota_floor, 0.0, 10.0), (opt.magnetic_well, MAGNETIC_WELL_TARGET, WELL_WEIGHT),
]
report = opt.EquilibriumReporter(
    ("QS", qs.total, ".4e"), ("aspect", opt.aspect_ratio, ".3f"),
    ("min |iota|", opt.min_abs_iota, ".3f"),
    ("magnetic well", opt.magnetic_well, ".3f"))
monitor = opt.OptimizationMonitor()

### Run the optimization ######################################################

# A RuntimeWarning about uncertified Jacobian columns is expected once the
# optimizer leaves the seed and needs no action; see examples/README.md.
for max_mode, max_nfev in zip(QA_MAX_MODES, QA_MAX_NFEV):
    print(f"\n===== vacuum QA seed stage, max_mode = {max_mode} =====")
    mpol, ntor = max(inp.mpol, max_mode + 2, MINIMUM_MPOL), max(inp.ntor, max_mode + 2)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=ntor, ntheta=2 * mpol + 6, nzeta=2 * ntor + 4)
    problem = opt.VmecProblem.from_tuples(inp, shape_terms, max_mode=max_mode,
        use_ess=True, restart_from=equilibrium,
        forward_max_iterations=STAGE_MAX_ITERATIONS,
        progress=True, evaluation_progress=True)
    problem.compile_residual_and_jacobian()
    monitor.problem = problem
    step = PARAMETER_STEP * problem.scales
    result = least_squares(problem.residual, problem.x0, jac=problem.residual_jac,
        x_scale=step, bounds=(problem.x0 - MAX_PARAMETER_CHANGE * step,
                             problem.x0 + MAX_PARAMETER_CHANGE * step),
        max_nfev=max_nfev, ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor)
    inp, equilibrium = problem.input_from_x(result.x), problem.equilibrium_from_x(result.x)
    report(f"QA mode {max_mode}", equilibrium)

if ci_smoke:
    inp = vj.VmecInput.from_file(SMOKE_INPUT_FILE)
    equilibrium = opt.solve_equilibrium(inp)

# Add the Landreman--Buller--Drevlak pressure profiles to the optimized QA
# boundary, then alternate hot-started VMEX solves and Redl current updates.
n0 = 3.0e20 * (TARGET_BETA / 0.05) ** (1 / 3)
T0 = 15.0e3 * (TARGET_BETA / 0.05) ** (2 / 3)
profiles = KineticProfiles(n0 * np.array([1, 0, 0, 0, 0, -1]),
                           T0 * np.array([1, -1]), T0 * np.array([1, -1]))
if not ci_smoke:
    am = np.zeros(21); am[[0, 1, 5, 6]] = [1.0, -1.0, -1.0, 1.0]
    ac = np.zeros(21); ac[0] = 1.0
    inp = replace(inp, pmass_type="power_series", am=am,
        pres_scale=2 * ELEMENTARY_CHARGE * n0 * T0, ncurr=1,
        pcurr_type="power_series", ac=ac, curtor=0.0)
    calibration = opt.solve_equilibrium(inp, initial_state=equilibrium.solution)
    profile_scale = TARGET_BETA / float(calibration.wout.betatotal)
    n0 *= profile_scale ** (1 / 3); T0 *= profile_scale ** (2 / 3)
    profiles = KineticProfiles(n0 * np.array([1, 0, 0, 0, 0, -1]),
                               T0 * np.array([1, -1]), T0 * np.array([1, -1]))
    inp = replace(inp, pres_scale=inp.pres_scale * profile_scale)
    picard = self_consistent_bootstrap(inp, profiles, 0,
        n_iter=PICARD_ITERATIONS, tol=PICARD_TOLERANCE,
        degree=N_CURRENT_SPLINE - 1, s_eval=QA_SURFACES, verbose=True)
    inp, equilibrium = picard.input, picard.equilibrium

bootstrap = RedlBootstrapMismatch(
    profiles, helicity_n=0, surfaces=QA_SURFACES, n_lambda=REDL_N_LAMBDA)
finite_beta_terms = [(qs, 0.0, QA_WEIGHT), (bootstrap, 0.0, BOOTSTRAP_WEIGHT),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0), (iota_floor, 0.0, 10.0),
    (opt.volume_average_beta, TARGET_BETA, BETA_WEIGHT),
    (opt.magnetic_well, MAGNETIC_WELL_TARGET, WELL_WEIGHT)]

# One physical lambda must represent the same particles on every radius and
# field-line label. Keep these pitches fixed throughout the maximum-J stages.
pitch = np.asarray(common_trapped_pitches_state(
    equilibrium.solution, equilibrium.solver_context, SURFACES, TRAPPING_DEPTHS,
    nalpha=ACTION_NALPHA, points_per_period=ACTION_POINTS,
    num_periods=ACTION_PERIODS))
maximum_j = MaximumJResidual(SURFACES, pitch, mboz=ACTION_MBOZ, nboz=ACTION_MBOZ,
    nalpha=ACTION_NALPHA, points_per_period=ACTION_POINTS,
    num_periods=ACTION_PERIODS, max_wells=ACTION_MAX_WELLS,
    quadrature_order=ACTION_QUADRATURE, target=MAXJ_TARGET)
for max_mode, max_nfev in zip(MAXJ_MAX_MODES, MAXJ_MAX_NFEV):
    print(f"\n===== QA + maximum-J stage, max_mode = {max_mode} =====")
    mpol, ntor = max(inp.mpol, max_mode + 2, MINIMUM_MPOL), max(inp.ntor, max_mode + 2)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=ntor, ntheta=2 * mpol + 6, nzeta=2 * ntor + 4)
    inp = opt.resample_current_profile(inp, N_CURRENT_SPLINE)
    terms = [(maximum_j, 0.0, MAXJ_WEIGHT), *finite_beta_terms]
    problem = opt.VmecProblem.from_tuples(inp, terms, max_mode=max_mode,
        current_dofs=N_CURRENT_SPLINE - 1, use_ess=True, restart_from=equilibrium,
        forward_max_iterations=STAGE_MAX_ITERATIONS,
        progress=True)
    print(f"dof_names = {problem.dof_names}")
    problem.compile_residual_and_jacobian()
    monitor.problem = problem
    step = PARAMETER_STEP * problem.scales
    step[-N_CURRENT_SPLINE:] = CURRENT_PARAMETER_STEP
    result = least_squares(problem.residual, problem.x0, jac=problem.residual_jac,
        x_scale=step, bounds=(problem.x0 - MAX_PARAMETER_CHANGE * step,
                             problem.x0 + MAX_PARAMETER_CHANGE * step),
        max_nfev=max_nfev, ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor)
    inp, equilibrium = problem.input_from_x(result.x), problem.equilibrium_from_x(result.x)
    report(f"maximum-J mode {max_mode}", equilibrium)

### Check the result ##########################################################

# The optimizer's grid is not the certificate: re-solve on a finer radial grid.
final_input = replace(inp, ns_array=np.array([FINAL_NS]),
    ftol_array=np.array([FINAL_FTOL]), niter_array=np.array([FINAL_NITER]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution, verbose=not ci_smoke,
    raise_on_max_iterations=True)
diagnostics = maximum_j.compute_state(final_equilibrium.solution,
                                      final_equilibrium.solver_context)

### Print, plot and save ######################################################

report("final", final_equilibrium)
print(f"maximum-J residual = {float(diagnostics['total']):.4e}, "
      f"outer-radius maximum-J fraction = {float(diagnostics['maximum_j_fraction']):.1%}")
print(f"beta = {float(final_equilibrium.wout.betatotal):.3%}; "
      "the reported fraction excludes the near-axis region where QA and maximum-J are incompatible.")

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}\nWrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
for path in vj.plot_wout(wout_path, ".", j_pitch=float(pitch[0])).values():
    print(f"Wrote {path}")
