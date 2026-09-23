#!/usr/bin/env python
"""Optimize a boundary for quasi-isodynamic confinement with shape limits.

SciPy nonlinear least squares varies the boundary Fourier coefficients of a
vacuum equilibrium in stages of increasing mode number. The residual vector
holds the constructed-QI residual, the aspect ratio, and one-sided hinges on
the rotational transform, the mirror ratio and the elongation; VMEX supplies
its exact Jacobian.

The QI residual is evaluated on a Boozer grid whose resolution is a knob of
its own. The run optimizes on QI_OPTIONS and re-scores the finished boundary
on the finer VALIDATION_OPTIONS, so an optimum that only exists at the
optimizer's grid resolution is visible as a gap between the two numbers.
"""

import os
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.qi import ConstructedQIResidual

# The seed deck. This nfp = 2 boundary already scores constructed QI 5e-3,
# where a perturbed circular seed scores 1.3
# (benchmarks/qi_optimization_profile_office.json):
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "input.QI_nfp2_initial"

# Radial grid every optimizer trial is solved on:
STAGE_NS = 31
STAGE_FTOL = 1.0e-12
STAGE_NITER = 5500

# Flux surfaces the QI residual is evaluated on:
SURFACES = np.linspace(0.1, 1.0, 6)

# Mode ladder: highest boundary mode number varied in each stage, and the
# residual evaluations each stage may spend. A max_mode = 3 stage lowered the
# cost by 1 % for 45 % of the run, so it is not shipped:
MAX_MODES = [1, 2]
MAX_NFEV = [10, 15]  # Boozer resolution the QI residual is optimized on, and the finer one the
# finished boundary is re-scored with:
QI_OPTIONS = dict(mboz=12, nboz=12, nphi=61, nalpha=18, n_bounce=21)
VALIDATION_OPTIONS = dict(mboz=14, nboz=14, nphi=101, nalpha=29, n_bounce=31)

# Targets and limits:
ASPECT_TARGET = 6.0
IOTA_FLOOR = 0.51                 # minimum |iota| over the profile
MIRROR_LIMIT = 0.21
ELONGATION_LIMIT = 8.0

# A finite-weight hinge settles just above its threshold, and the final
# FINAL_NS solve reads the mirror ratio about 4e-4 above the stage grid, so
# the mirror hinge starts pulling this far below the limit:
MIRROR_HINGE_FRACTION = 0.99

# Step control:
ESS_ALPHA = 1.2                   # lower only after a low-mode QI basin has converged
VARY_MAJOR_RADIUS = False         # True optimizes RBC(0,0) instead of fixing it

# Equilibrium resolution: poloidal and toroidal mode numbers are max_mode + 2,
# but never below MINIMUM_MPOL:
MINIMUM_MPOL = 5

# Verification solve of the optimized boundary:
FINAL_NS = 71
FINAL_FTOL = 1.0e-14
FINAL_NITER = 8000

# Every output file name contains this:
OUTPUT_NAME = "QI_optimized"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    QI_OPTIONS = dict(mboz=8, nboz=8, nphi=31, nalpha=7, n_bounce=7)
    VALIDATION_OPTIONS = QI_OPTIONS
    MAX_MODES, MAX_NFEV = [2], [5]
    FINAL_NS, FINAL_FTOL = 31, 1.0e-10

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

inp = replace(vj.VmecInput.from_file(INPUT_FILE),
              ns_array=np.array([STAGE_NS]), ftol_array=np.array([STAGE_FTOL]),
              niter_array=np.array([STAGE_NITER]))

### Set up the objective ######################################################

def iota_floor(equilibrium_state, solver_context):
    """Hinge on the profile minimum of |iota|: a mean target can hide a near-zero surface.

    opt.mean_iota targets the average instead; opt.soft_min_abs_iota is the smooth minimum.
    """
    return jnp.maximum(
        IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)


def mirror_excess(equilibrium_state, solver_context):
    """Hinge on the mirror ratio, pulling from MIRROR_HINGE_FRACTION of the limit."""
    return jnp.maximum(
        opt.mirror_ratio(equilibrium_state, solver_context)
        - MIRROR_HINGE_FRACTION * MIRROR_LIMIT, 0.0)


def elongation_excess(equilibrium_state, solver_context):
    """Hinge on the cross-section elongation above its limit."""
    return jnp.maximum(
        opt.max_elongation(equilibrium_state, solver_context) - ELONGATION_LIMIT, 0.0)


# Each term is (function, target, weight).
qi = ConstructedQIResidual(SURFACES, **QI_OPTIONS)
objective_function_terms = [
    (opt.aspect_ratio, ASPECT_TARGET, 0.005),
    (iota_floor, 0.0, 10.0),
    (mirror_excess, 0.0, 1000.0),
    (elongation_excess, 0.0, 10.0),
]
qi_terms = [(qi, 0.0, 10.0), *objective_function_terms]

report = opt.EquilibriumReporter(
    ("constructed QI", qi.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"), ("min |iota|", opt.min_abs_iota, ".4f"),
    ("mirror", opt.mirror_ratio, ".4f"),
    ("elongation", opt.max_elongation, ".4f"))
monitor = opt.OptimizationMonitor()

### Run the optimization ######################################################

for max_mode, max_nfev in zip(MAX_MODES, MAX_NFEV):
    print(f"\n===== QI stage, max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    # A fresh problem restarts SciPy's trust-region model; equal-shape JAX
    # executables are reused.  A RuntimeWarning about uncertified Jacobian
    # columns is expected once the optimizer leaves the seed and needs no
    # action; see examples/README.md.
    problem = opt.VmecProblem.from_tuples(
        inp, qi_terms, max_mode=max_mode, use_ess=True, progress=not ci_smoke,
        evaluation_progress=not ci_smoke,
        ess_alpha=ESS_ALPHA, vary_major_radius=VARY_MAJOR_RADIUS)
    print(f"dof_names = {problem.dof_names}")
    monitor.problem = problem
    if not ci_smoke:
        problem.compile_residual_and_jacobian()
    result = least_squares(
        problem.residual, problem.x0, jac=problem.residual_jac,
        x_scale=problem.scales, max_nfev=max_nfev,
        ftol=1.0e-6, xtol=1.0e-10, verbose=2, callback=monitor)
    inp = problem.input_from_x(result.x)
    equilibrium = problem.equilibrium_from_x(result.x)
    report(f"QI mode {max_mode}", equilibrium)

### Check the result ##########################################################

# The optimizer's grid is not the certificate: re-solve the optimized boundary
# on a finer radial grid, then re-score QI on the finer Boozer grid too.
final_input = replace(
    inp, ns_array=np.array([FINAL_NS]), ftol_array=np.array([FINAL_FTOL]),
    niter_array=np.array([FINAL_NITER]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution,
    verbose=not ci_smoke, raise_on_max_iterations=True)

### Print, plot and save ######################################################

qi_final = report("final", final_equilibrium)["constructed QI"]
qi_validation = ConstructedQIResidual(SURFACES, **VALIDATION_OPTIONS)
print(f"\nQI total {qi_final:.3e}; independent fine-grid validation "
      f"{float(qi_validation.total(final_equilibrium)):.3e}")

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}")
print(f"Wrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
