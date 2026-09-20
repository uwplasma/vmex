#!/usr/bin/env python
"""LASYM finite-beta constructed-QI boundary optimization."""

import os
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.qi import ConstructedQIResidual

# Number of field periods, and the seed deck the boundary is shaped from:
NFP = 2
INPUT_FILE = Path(__file__).resolve().parents[2] / "data" / f"input.minimal_seed_nfp{NFP}"

SEED_PERTURBATION = 0.05
ASYMMETRY_PERTURBATION = 0.01

# Flux surfaces the QI residual is evaluated on:
SURFACES = np.linspace(0.1, 1.0, 6)

# Mode ladder: highest boundary mode number varied in each stage, and the
# residual evaluations each stage may spend:
MAX_MODES = [3, 5]
MAX_NFEV = [80, 100]

# Boozer resolution the constructed-QI residual is evaluated on:
QI_OPTIONS = dict(mboz=12, nboz=12, nphi=61, nalpha=18, n_bounce=21)

# Targets and limits.  This lane carries the beta residual but no Mercier or
# resistive-interchange rows:
TARGET_BETA = 0.01
ASPECT_TARGET = 5.0
IOTA_FLOOR = 0.51                 # minimum |iota| over the profile
MIRROR_LIMIT = 0.21
ELONGATION_LIMIT = 8.0

# Step control. One scaled variable moves a low-order coefficient by
# PARAMETER_STEP metres, and a stage may move it MAX_PARAMETER_CHANGE steps:
PARAMETER_STEP = 0.01
MAX_PARAMETER_CHANGE = 3.0
ESS_ALPHA = 1.2                   # lower only after a low-mode QI basin has converged

# The deck's own radial ladder is used at the shipped settings:
STAGE_OVERRIDES = {}

# Forward-solve iteration cap inside an optimizer trial:
STAGE_MAX_ITERATIONS = 3000

# Equilibrium resolution: poloidal and toroidal mode numbers are max_mode + 2,
# but never below MINIMUM_MPOL:
MINIMUM_MPOL = 5

# One calibration solve rescales this to TARGET_BETA:
CALIBRATION_PRES_SCALE = 100.0

# Verification solve of the optimized boundary:
FINAL_NS = 101
FINAL_FTOL = 1e-14
FINAL_NITER = 20000

# Every output file name contains this; each stage also writes its own
# boundary as input.<STAGE_NAME>_max_mode_NNN:
OUTPUT_NAME = "QI_LASYM_finite_beta_optimized"
STAGE_NAME = "QI_finite_beta"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    SURFACES = np.array([0.25, 0.6, 0.9])
    MINIMUM_MPOL = 3
    MAX_MODES, MAX_NFEV = [1], [2]
    QI_OPTIONS = dict(mboz=8, nboz=8, nphi=31, nalpha=7, n_bounce=7)
    STAGE_OVERRIDES = dict(ns_array=np.array([11]), ftol_array=np.array([1e-8]),
                           niter_array=np.array([1500]))
    STAGE_MAX_ITERATIONS = 100
    PARAMETER_STEP = 0.001
    FINAL_NS, FINAL_FTOL = 31, 1e-10

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

# VmecInput is frozen, so copy its arrays before shaping the seed boundary.
inp = replace(vj.VmecInput.from_file(INPUT_FILE), **STAGE_OVERRIDES)
rbc, zbs, rbs, zbc = inp.rbc.copy(), inp.zbs.copy(), inp.rbs.copy(), inp.zbc.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
# Nonzero RBS(1,1)/ZBC(1,1) prevents the local solve from remaining in the
# original stellarator-symmetric subspace.
rbs[inp.ntor + 1, 1], zbc[inp.ntor + 1, 1] = ASYMMETRY_PERTURBATION, -ASYMMETRY_PERTURBATION
am = np.zeros(21)
am[:2] = [1.0, -1.0]  # p(s) = PRES_SCALE * (1 - s)
inp = replace(inp, lasym=True, rbc=rbc, zbs=zbs, rbs=rbs, zbc=zbc,
    pmass_type="power_series", am=am, pres_scale=CALIBRATION_PRES_SCALE)

# One solve calibrates the pressure amplitude to the requested beta.
calibration = opt.solve_equilibrium(inp)
inp = replace(inp, pres_scale=inp.pres_scale * TARGET_BETA / float(calibration.wout.betatotal))
equilibrium = opt.solve_equilibrium(inp, initial_state=calibration.solution)

### Set up the objective ######################################################

qi = ConstructedQIResidual(SURFACES, **QI_OPTIONS)
def iota_floor(equilibrium_state, solver_context):
    """Hinge on the profile minimum of |iota|: a mean target can hide a near-zero surface.

    opt.mean_iota targets the average instead; opt.soft_min_abs_iota is the smooth minimum.
    """
    return jnp.maximum(IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)

def mirror_excess(equilibrium_state, solver_context):
    """Hinge on the mirror ratio above its limit; zero while the limit holds."""
    return jnp.maximum(opt.mirror_ratio(equilibrium_state, solver_context) - MIRROR_LIMIT, 0.0)

def elongation_excess(equilibrium_state, solver_context):
    """Hinge on the cross-section elongation above its limit."""
    return jnp.maximum(opt.max_elongation(equilibrium_state, solver_context) - ELONGATION_LIMIT, 0.0)

# Each term is (function, target, weight).
objective_function_terms = [(qi, 0.0, 10.0), (opt.aspect_ratio, ASPECT_TARGET, 0.005),
    (iota_floor, 0.0, 10.0), (mirror_excess, 0.0, 10.0),
    (elongation_excess, 0.0, 10.0),
    (opt.volume_average_beta, TARGET_BETA, 1.0 / TARGET_BETA**2)]
report = opt.EquilibriumReporter(
    ("constructed QI", qi.total, ".4e"), ("beta", opt.volume_average_beta, ".3%"),
    ("aspect", opt.aspect_ratio, ".3f"), ("iota", opt.mean_iota, ".3f"))
monitor = opt.OptimizationMonitor()

### Run the optimization ######################################################

# A RuntimeWarning about uncertified Jacobian columns is expected once the
# optimizer leaves the seed and needs no action; see examples/README.md.
for max_mode, max_nfev in zip(MAX_MODES, MAX_NFEV):
    print(f"\n===== LASYM finite-beta QI stage, max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    problem = opt.VmecProblem.from_tuples(inp, objective_function_terms, max_mode=max_mode,
        use_ess=True, ess_alpha=ESS_ALPHA, restart_from=equilibrium, progress=True, evaluation_progress=True,
        forward_max_iterations=100 if ci_smoke else 3000)
    print(f"dof_names = {problem.dof_names}")
    problem.compile_residual_and_jacobian()
    monitor.problem = problem
    step = PARAMETER_STEP * problem.scales
    result = least_squares(problem.residual, problem.x0, jac=problem.residual_jac,
        x_scale=step, bounds=(problem.x0 - MAX_PARAMETER_CHANGE * step,
                             problem.x0 + MAX_PARAMETER_CHANGE * step),
        max_nfev=max_nfev, ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor)
    inp, equilibrium = problem.input_from_x(result.x), problem.equilibrium_from_x(result.x)
    inp.to_indata(f"input.{STAGE_NAME}_max_mode_{max_mode:03d}")
    report(f"mode {max_mode}", equilibrium)

### Check the result ##########################################################

# The optimizer's grid is not the certificate: re-solve the optimized boundary
# on a finer radial grid to a tighter tolerance and quote that.
final_input = replace(inp, ns_array=np.array([FINAL_NS]),
    ftol_array=np.array([FINAL_FTOL]), niter_array=np.array([FINAL_NITER]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution, verbose=not ci_smoke,
    raise_on_max_iterations=True)
### Print, plot and save ######################################################

print(f"asymmetric boundary norm = "
      f"{np.linalg.norm(final_input.rbs) + np.linalg.norm(final_input.zbc):.6e}")
report("final", final_equilibrium)

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}\nWrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
