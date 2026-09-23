#!/usr/bin/env python
"""Optimize a boundary for quasi-isodynamic confinement with a SciPy scalar-gradient method.

A mode ladder of bounded L-BFGS-B solves, each driven by the aggregate
objective and its exact gradient: VMEX scalarizes the residual rows and returns
the gradient from one reverse equilibrium adjoint per evaluation.

The QI residual is evaluated on a Boozer grid whose resolution is a knob of its
own. The run optimizes on QI_OPTIONS and re-scores the finished boundary on the
finer VALIDATION_OPTIONS, so an optimum that exists only at the optimizer's
grid resolution shows up as a gap between the two numbers.
"""

import os
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

import vmex as vj
from vmex import OptimizationMonitor
from vmex import optimize as opt
from vmex.core.input import VmecInput
from vmex.core.qi import ConstructedQIResidual

# Number of field periods, and the seed deck the boundary is shaped from:
NFP = 2
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"
SURFACES = np.linspace(0.1, 1.0, 6)
MAX_MODES = [1, 2]
MAXITER = 15
METHOD = "L-BFGS-B"  # or "BFGS"
PARAMETER_BOUND = 3.0
BOUNDARY_STEP = 0.05              # metres represented by one scaled variable
ASPECT_TARGET = 6.0
IOTA_FLOOR = 0.26                 # minimum |iota| over the profile
MIRROR_LIMIT = 0.21
ELONGATION_LIMIT = 8.0
MINIMUM_MPOL = 5
VARY_MAJOR_RADIUS = False         # True optimizes RBC(0,0) instead of fixing it
SEED_PERTURBATION = 0.05

# Verification solve of the optimized boundary:
FINAL_NS = 71
FINAL_FTOL = 1.0e-14
FINAL_NITER = 8000

QI_OPTIONS = dict(mboz=12, nboz=12, nphi=61, nalpha=18, n_bounce=21)
VALIDATION_OPTIONS = dict(mboz=14, nboz=14, nphi=101, nalpha=29, n_bounce=31)

# Every output file name contains this:
OUTPUT_NAME = f"QI_scipy_{METHOD}"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    QI_OPTIONS = dict(mboz=8, nboz=8, nphi=31, nalpha=7, n_bounce=7)
    VALIDATION_OPTIONS = QI_OPTIONS
    MAX_MODES, MAXITER = [2], 1
    FINAL_NS, FINAL_FTOL = 31, 1.0e-10

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

# VmecInput is frozen, so copy its arrays before shaping the seed boundary.
inp = VmecInput.from_file(INPUT_FILE)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
inp = replace(inp, rbc=rbc, zbs=zbs)

### Set up the objective ######################################################


qi = ConstructedQIResidual(SURFACES, **QI_OPTIONS)

def iota_floor(equilibrium_state, solver_context):
    """Hinge on the profile minimum of |iota|: a mean target can hide a near-zero surface.

    opt.mean_iota targets the average instead; opt.soft_min_abs_iota is the smooth minimum.
    """
    return jnp.maximum(
        IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)

def elongation_excess(equilibrium_state, solver_context):
    return jnp.maximum(
        opt.max_elongation(equilibrium_state, solver_context) - ELONGATION_LIMIT, 0.0)

def mirror_excess(equilibrium_state, solver_context):
    return jnp.maximum(
        opt.mirror_ratio(equilibrium_state, solver_context) - MIRROR_LIMIT, 0.0)

objective_function_terms = [
    (opt.aspect_ratio, ASPECT_TARGET, 0.005),
    (iota_floor, 0.0, 10.0),
    (mirror_excess, 0.0, 10.0),
    (elongation_excess, 0.0, 10.0),
]
qi_terms = [(qi, 0.0, 10.0), *objective_function_terms]

report = opt.EquilibriumReporter(
    ("constructed QI", qi.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"), ("mirror", opt.mirror_ratio, ".4f"),
    ("elongation", opt.max_elongation, ".4f"))

def x_from_y(y):
    return x0 + scales * y

def cost(y):
    return problem.fun(x_from_y(y))

def gradient(y):
    return scales * problem.grad(x_from_y(y))

options = {"maxiter": MAXITER, "gtol": 1.0e-6}
if METHOD == "L-BFGS-B":
    options.update(maxls=20, ftol=1.0e-12, maxcor=20)

# If a RuntimeWarning reports uncertified Jacobian columns, it is expected
# once the optimizer leaves the seed and needs no action: the shipped
# jacobian_adjoint_tol=1e-4 and jacobian_adjoint_maxiter=10 are the measured
# optimum, since ten times that budget moved the Jacobian by 2e-8 and
# certified no extra column. Both are from_tuples arguments; pass
# evaluation_progress=False to drop the per-evaluation timing lines.
### Run the optimization ######################################################

for max_mode in MAX_MODES:
    print(f"\n===== QI stage, max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    problem = opt.VmecProblem.from_tuples(
        inp, qi_terms, max_mode=max_mode, use_ess=True, progress=not ci_smoke,
        evaluation_progress=not ci_smoke,
        vary_major_radius=VARY_MAJOR_RADIUS,
    )
    print(f"dof_names = {problem.dof_names}")
    if not ci_smoke:
        problem.compile_value_and_gradient()
    x0, scales = problem.x0, BOUNDARY_STEP * problem.scales
    monitor = OptimizationMonitor(problem)

    def monitor_y(intermediate_result):
        monitor({"x": x_from_y(intermediate_result.x), "fun": intermediate_result.fun,
                 "jac": gradient(intermediate_result.x)})

    result = minimize(cost, np.zeros_like(x0), jac=gradient, method=METHOD,
        bounds=[(-PARAMETER_BOUND, PARAMETER_BOUND)] * x0.size if METHOD == "L-BFGS-B" else None,
        callback=monitor_y, options=options)
    result.x = x_from_y(result.x)
    equilibrium = problem.equilibrium_from_x(result.x)
    inp = problem.input_from_x(result.x)

### Check the result ##########################################################

# The optimizer's grid is not the certificate: re-solve the optimized boundary
# on a finer radial grid to a tighter tolerance and quote that.
final_input = replace(inp,
    ns_array=np.array([FINAL_NS]),
    ftol_array=np.array([FINAL_FTOL]),
    niter_array=np.array([FINAL_NITER]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution,
    verbose=not ci_smoke, raise_on_max_iterations=True)
qi_final = report("final", final_equilibrium)["constructed QI"]
print(f"\n{METHOD}: final cost = {float(result.fun):.12e}, QI total = {qi_final:.3e}")
qi_validation = ConstructedQIResidual(SURFACES, **VALIDATION_OPTIONS)
print(f"\nQI total {qi_final:.3e}; independent fine-grid validation "
      f"{float(qi_validation.total(final_equilibrium)):.3e}")

### Print, plot and save ######################################################

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}")
print(f"Wrote {wout_path}")

print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
