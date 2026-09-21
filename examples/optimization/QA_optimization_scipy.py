#!/usr/bin/env python
"""Optimize a boundary for quasi-axisymmetry with a SciPy scalar-gradient method.

One bounded L-BFGS-B solve over a single mode number, driven by the aggregate
objective and its exact gradient: VMEX scalarizes the residual rows and returns
the gradient from one reverse equilibrium adjoint per evaluation.

Set METHOD to "BFGS" to run unbounded instead. The companion
QA_optimization.py keeps the residual vector and its full Jacobian, and
QA_optimization_scalar.py runs the same scalar lane over a mode ladder.
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


# Number of field periods, and the seed deck the boundary is shaped from:
NFP = 2
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"
SURFACES = np.linspace(0.1, 1.0, 10)
MAX_MODE = 3
MAXITER = 70
METHOD = "L-BFGS-B"         # or "BFGS"
PARAMETER_BOUND = 1.0
BOUNDARY_STEP = 0.2               # metres represented by one scaled variable
ASPECT_TARGET = 6.0
IOTA_FLOOR = 0.42                 # minimum |iota| over the profile
MINIMUM_MPOL = 5
VARY_MAJOR_RADIUS = False         # True optimizes RBC(0,0) instead of fixing it
SEED_PERTURBATION = 0.05

# Verification solve of the optimized boundary:
FINAL_NS = 71
FINAL_FTOL = 1.0e-14
FINAL_NITER = 8000

# Every output file name contains this:
OUTPUT_NAME = f"QA_scipy_{METHOD}"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODE, MAXITER = 1, 4
    FINAL_NS, FINAL_FTOL = 31, 1.0e-10

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

# VmecInput is frozen, so copy its arrays before shaping the seed boundary.
inp = vj.VmecInput.from_file(INPUT_FILE)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
inp = replace(inp, rbc=rbc, zbs=zbs)
mpol = max(MAX_MODE + 2, MINIMUM_MPOL)
inp = replace(inp, delt=0.5).change_resolution(
    mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)

def iota_floor(equilibrium_state, solver_context):
    """Hinge on the profile minimum of |iota|: a mean target can hide a near-zero surface.

    opt.mean_iota targets the average instead; opt.soft_min_abs_iota is the smooth minimum.
    """
    return jnp.maximum(
        IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)


### Set up the objective ######################################################


qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
objective_function_terms = [
    (qs, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (iota_floor, 0.0, 10.0),
]
problem = opt.VmecProblem.from_tuples(inp, objective_function_terms, max_mode=MAX_MODE,
                                      vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True, progress=True, evaluation_progress=True)
print(f"dof_names = {problem.dof_names}")
problem.compile_value_and_gradient()
x0, scales = problem.x0, BOUNDARY_STEP * problem.scales

report = opt.EquilibriumReporter(
    ("QS total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("min |iota|", opt.min_abs_iota, ".4f"), ("magnetic well", opt.magnetic_well, ".4f"))

def x_from_y(y):
    return x0 + scales * y

def cost(y):
    return problem.fun(x_from_y(y))

def gradient(y):
    return scales * problem.grad(x_from_y(y))

monitor = OptimizationMonitor(problem)
def monitor_y(intermediate_result):
    monitor({"x": x_from_y(intermediate_result.x), "fun": intermediate_result.fun,
             "jac": gradient(intermediate_result.x)})

options = {"maxiter": MAXITER, "gtol": 1.0e-6}
if METHOD == "L-BFGS-B":
    options.update(maxls=20, ftol=1.0e-12, maxcor=20)
### Run the optimization ######################################################

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
final_total = report("final", final_equilibrium)["QS total"]
print(f"\n{METHOD}: final cost = {float(result.fun):.12e}, QS total = {final_total:.3e}")

### Print, plot and save ######################################################

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}")
print(f"Wrote {wout_path}")

print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
