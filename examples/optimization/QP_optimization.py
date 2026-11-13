#!/usr/bin/env python
"""Optimize a boundary for quasi-poloidal symmetry with shape limits.

SciPy nonlinear least squares varies the boundary Fourier coefficients of a
vacuum equilibrium in stages of increasing mode number. The residual vector
holds the quasisymmetry ratio, the aspect ratio, and one-sided limits on the
rotational transform, the mirror ratio and the elongation; VMEX supplies its
exact Jacobian.

Quasi-poloidal symmetry is the hard member of the family: the first-order
near-axis field is helical, so the residual is evaluated on s >= 0.5, and the
shape limits bind (the elongation limit sits at its limit in the result).
Loosening the limits lowers the residual at the price of an extremely shaped
boundary.
"""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt

# Number of field periods, and the seed deck the boundary is shaped from:
NFP = 2
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"

# Rotating-ellipse amplitude added to the circular seed:
SEED_PERTURBATION = 0.05

# Radial ladder every optimizer trial is solved on:
STAGE_NS_ARRAY = [25, 35]
STAGE_FTOL_ARRAY = [1.0e-11, 1e-12]
STAGE_NITER_ARRAY = [300, 8000]

# Flux surfaces the quasisymmetry residual is evaluated on:
SURFACES = np.array([0.5, 0.7, 0.9])

# Mode ladder: highest boundary mode number varied in each stage, and the
# residual evaluations each stage may spend:
MAX_MODES = [1, 2, 3]
MAX_NFEV = [10, 15, 20]

# Targets and limits:
ASPECT_TARGET = 6.0
IOTA_FLOOR = 0.51                 # minimum |iota| over the profile
MIRROR_LIMIT = 0.35
ELONGATION_LIMIT = 12.0

# Hinge weights. Heavy limits on the shape limits make the trust region crawl
# along the active constraints (a perturbed seed stalled at twice the residual):
IOTA_WEIGHT = 10.0
SHAPE_WEIGHT = 1.0

# Step control:
ESS_ALPHA = 1.2                   # smaller values let high Fourier modes move more
VARY_MAJOR_RADIUS = False         # True optimizes RBC(0,0) instead of fixing it

# Equilibrium resolution: poloidal and toroidal mode numbers are max_mode + 2,
# but never below MINIMUM_MPOL:
MINIMUM_MPOL = 5

# Verification solve of the optimized boundary:
FINAL_NS = 71
FINAL_FTOL = 1.0e-14
FINAL_NITER = 35000

# Every output file name contains this; each stage also writes its own
# boundary as input.<OUTPUT_NAME>_max_mode_NNN:
OUTPUT_NAME = "QP_optimized"
STAGE_NAME = "QP"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODES, MAX_NFEV = [1], [4]
    FINAL_NS, FINAL_FTOL = 31, 1.0e-10

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

# VmecInput is frozen, so copy its arrays before shaping the seed boundary.
inp = vj.VmecInput.from_file(INPUT_FILE)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
inp = replace(inp, rbc=rbc, zbs=zbs, delt=0.5,
              niter_array=np.array(STAGE_NITER_ARRAY),
              ftol_array=np.array(STAGE_FTOL_ARRAY),
              ns_array=np.array(STAGE_NS_ARRAY))

### Set up the objective ######################################################

# Each term is (function, target, weight).
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=0, helicity_n=1)
objective_function_terms = [
    (qs, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (opt.min_abs_iota, opt.at_least(IOTA_FLOOR), IOTA_WEIGHT),
    (opt.mirror_ratio, opt.at_most(MIRROR_LIMIT), SHAPE_WEIGHT),
    (opt.max_elongation, opt.at_most(ELONGATION_LIMIT), SHAPE_WEIGHT),
]

report = opt.EquilibriumReporter(
    ("QS total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"), ("elongation", opt.max_elongation, ".4f"),
    ("mirror", opt.mirror_ratio, ".4f"))
monitor = opt.OptimizationMonitor()

### Run the optimization ######################################################

# Every stage shares one resolution, so one problem at the largest max_mode
# serves them all: problem.subproblem() frees only the stage's harmonics and
# every jitted residual and Jacobian is compiled once.
mpol = max(max(MAX_MODES) + 2, MINIMUM_MPOL)
inp = replace(inp, delt=0.5).change_resolution(
    mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
# A RuntimeWarning about uncertified Jacobian columns is expected once the
# optimizer leaves the seed and needs no action; see examples/README.md.
problem = opt.VmecProblem.from_tuples(
    inp, objective_function_terms, max_mode=max(MAX_MODES),
    vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True, ess_alpha=ESS_ALPHA,
    progress=not ci_smoke, evaluation_progress=not ci_smoke)
monitor.problem = problem
if not ci_smoke:
    problem.compile_residual_and_jacobian()
x = problem.x0
for max_mode, max_nfev in zip(MAX_MODES, MAX_NFEV):
    print(f"\n===== QP stage, max_mode = {max_mode} =====")
    stage = problem.subproblem(max_mode=max_mode, x=x)
    print(f"dof_names = {stage.dof_names}")
    result = least_squares(
        stage.residual, stage.x0, jac=stage.residual_jac,
        x_scale=stage.scales, max_nfev=max_nfev, ftol=1e-6, xtol=1e-10,
        verbose=2, callback=monitor)
    x = stage.embed(result.x)
    inp = problem.input_from_x(x)
    equilibrium = problem.equilibrium_from_x(x)
    report(f"mode {max_mode}", equilibrium)
    inp.to_indata(f"input.{STAGE_NAME}_max_mode_{max_mode:03d}")

### Check the result ##########################################################

# The optimizer's grid is not the certificate: re-solve the optimized boundary
# on a finer radial grid to a tighter tolerance and quote that.
final_input = replace(
    inp, ns_array=np.array([FINAL_NS]), ftol_array=np.array([FINAL_FTOL]),
    niter_array=np.array([FINAL_NITER]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.state,
    verbose=not ci_smoke, raise_on_max_iterations=True)

### Print, plot and save ######################################################

final_total = report("final", final_equilibrium)["QS total"]
opt.report_targets(final_equilibrium, aspect=ASPECT_TARGET, iota_floor=IOTA_FLOOR,
                   mirror_limit=MIRROR_LIMIT, elongation_limit=ELONGATION_LIMIT)
print(f"\nQS total {final_total:.3e}")

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}")
print(f"Wrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
