#!/usr/bin/env python
"""Explore quasi-axisymmetric basins with basin hopping, then polish by least squares.

A local optimizer finds the basin it starts in. SciPy basin hopping perturbs
the boundary, runs a short bounded L-BFGS-B descent from each perturbation,
and keeps the best accepted point; exact residual/Jacobian least squares then
exploits the tuple structure for an efficient local finish.

Both phases drive the same objective. The global phase only chooses where the
local one starts, so the numbers worth comparing are the basin costs printed
per hop and the polished cost at the end.
"""

import os
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import basinhopping, least_squares

import vmex as vj
from vmex import optimize as opt

# Number of field periods, and the seed deck the boundary is shaped from:
NFP = 1
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"

# Rotating-ellipse amplitude added to the circular seed:
SEED_PERTURBATION = 0.05

# Flux surfaces the quasisymmetry residual is evaluated on:
SURFACES = np.linspace(0.1, 1.0, 10)

# Highest boundary Fourier mode number that is varied:
MAX_MODE = 2  # Global phase: hops, and the L-BFGS-B iterations each hop may spend:
N_BASINS = 2
LOCAL_MAXITER = 8

# Basin-hopping acceptance temperature, perturbation size, and the random
# seed that makes the walk reproducible:
BASIN_TEMPERATURE = 0.05
BASIN_STEPSIZE = 0.25
BASIN_SEED = 7

# Local finish: residual evaluations the polishing least squares may spend:
POLISH_NFEV = 15

# Targets:
ASPECT_TARGET = 6.0
IOTA_FLOOR = 0.37                 # minimum |iota| over the profile

# Step control. One scaled variable moves a low-order coefficient by
# BOUNDARY_STEP metres, and the walk is bounded at PARAMETER_BOUND of them:
BOUNDARY_STEP = 0.1
PARAMETER_BOUND = 1.0
# Smaller alpha damps high Fourier modes less than the default 1.2, helping
# basin exploration produce shapes unlike a low-mode local optimum:
ESS_ALPHA = 0.7
VARY_MAJOR_RADIUS = False         # True optimizes RBC(0,0) instead of fixing it

# Equilibrium resolution: poloidal and toroidal mode numbers are MAX_MODE + 2,
# but never below MINIMUM_MPOL:
MINIMUM_MPOL = 5

# Verification solve of the optimized boundary:
FINAL_NS = 71
FINAL_FTOL = 1e-14
FINAL_NITER = 20000

# Every output file name contains this:
OUTPUT_NAME = "QA_global_optimized"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODE, N_BASINS, LOCAL_MAXITER, POLISH_NFEV = 1, 1, 2, 3
    FINAL_NS, FINAL_FTOL = 31, 1e-10

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

# VmecInput is frozen, so copy its arrays before shaping the seed boundary.
inp = vj.VmecInput.from_file(INPUT_FILE)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
mpol = max(MAX_MODE + 2, MINIMUM_MPOL)
inp = replace(inp, rbc=rbc, zbs=zbs, delt=0.5).change_resolution(
    mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)

### Set up the objective ######################################################

def iota_floor(equilibrium_state, solver_context):
    """Hinge on the profile minimum of |iota|: a mean target can hide a near-zero surface.

    opt.mean_iota targets the average instead; opt.soft_min_abs_iota is the smooth minimum.
    """
    return jnp.maximum(
        IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)


# Each term is (function, target, weight).
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
objective_function_terms = [
    (qs, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (iota_floor, 0.0, 100.0),
]

problem = opt.VmecProblem.from_tuples(
    inp, objective_function_terms, max_mode=MAX_MODE,
    vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True, ess_alpha=ESS_ALPHA,
    progress=True, evaluation_progress=True)
print(f"dof_names = {problem.dof_names}")
problem.compile_value_and_gradient()

# SciPy works in dimensionless increments y, with x = x0 + scales * y, so every
# coefficient moves on a similar scale.
x0, scales = problem.x0, BOUNDARY_STEP * problem.scales


def x_from_y(y):
    return x0 + scales * y


def value_and_gradient(y):
    value, gradient = problem.value_and_grad(x_from_y(y))
    return value, scales * gradient


monitor = opt.OptimizationMonitor(problem)
best = {"y": np.zeros_like(x0), "value": np.inf}


def basin_report(y, value, accepted):
    """Basin-hopping callback: record the hop and keep the best accepted point."""
    gradient = value_and_gradient(y)[1]
    monitor({"x": x_from_y(y), "fun": value, "jac": gradient})
    if accepted and value < best["value"]:
        best.update(y=np.asarray(y).copy(), value=float(value))
    print(f"basin cost = {value:.6e}, accepted = {accepted}")


### Run the optimization ######################################################

print("First print can take more than ten minutes")
bounds = [(-PARAMETER_BOUND, PARAMETER_BOUND)] * x0.size
basinhopping(value_and_gradient, np.zeros_like(x0), niter=N_BASINS,
    T=BASIN_TEMPERATURE, stepsize=BASIN_STEPSIZE,
    minimizer_kwargs={"method": "L-BFGS-B", "jac": True,
        "bounds": bounds, "options": {"maxiter": LOCAL_MAXITER, "ftol": 1e-10}},
    callback=basin_report, rng=np.random.default_rng(BASIN_SEED), disp=True)

# The global phase selected a basin; exact residual/Jacobian least squares now
# finishes inside it.
x_global = x_from_y(best["y"])
polish = least_squares(problem.residual, x_global, jac=problem.residual_jac,
    x_scale=problem.scales, bounds=(x0 - scales, x0 + scales), max_nfev=POLISH_NFEV,
    ftol=1e-7, xtol=1e-10, verbose=2, callback=monitor)
inp, equilibrium = problem.input_from_x(polish.x), problem.equilibrium_from_x(polish.x)

### Check the result ##########################################################

# The optimizer's grid is not the certificate: re-solve the optimized boundary
# on a finer radial grid to a tighter tolerance and quote that.
final_input = replace(inp, ns_array=np.array([FINAL_NS]),
    ftol_array=np.array([FINAL_FTOL]), niter_array=np.array([FINAL_NITER]))
final_equilibrium = opt.solve_equilibrium(final_input, initial_state=equilibrium.solution,
    verbose=not ci_smoke, raise_on_max_iterations=True)

### Print, plot and save ######################################################

report = opt.EquilibriumReporter(("QS", qs.total, ".4e"),
    ("aspect", opt.aspect_ratio, ".3f"), ("iota", opt.mean_iota, ".3f"))
report("final", final_equilibrium)

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}\nWrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
