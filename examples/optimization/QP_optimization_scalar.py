#!/usr/bin/env python
"""Optimize a boundary for quasi-poloidal symmetry with one adjoint per gradient.

This minimizes the same weighted sum of squared residuals as the least-squares
example of the same family. The difference is algorithmic: the residual rows
are summed into one scalar before implicit differentiation, so SciPy L-BFGS-B
receives a value and a gradient from one reverse equilibrium adjoint instead of
a residual vector and its full Jacobian.

The scalar form trades objective progress per evaluation for a cheaper cold
start and lower peak memory: at a matched evaluation budget the least-squares
driver reached roughly a 3x lower objective on the same problem, so it remains
the default for objective progress.
"""

import os
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

import vmex as vj
from vmex import optimize as opt

NFP = 2
SURFACES = np.array([0.5, 0.7, 0.9])
MAX_MODES = [1, 2]
MAXITER = [8, 10]
ASPECT_TARGET = 6.0
IOTA_FLOOR = 0.51                 # minimum |iota| over the profile
MIRROR_LIMIT = 0.35
ELONGATION_LIMIT = 12.0
PARAMETER_STEP = 0.02
MAX_PARAMETER_CHANGE = 5.0
ESS_ALPHA = 1.2                   # smaller values let high Fourier modes move more
MINIMUM_MPOL = 5
VARY_MAJOR_RADIUS = False         # True optimizes RBC(0,0) instead of fixing it
SEED_PERTURBATION = 0.05
POLISH_FORCE_BALANCE = False      # True polishes only the final saved state

# Verification solve of the optimized boundary:
FINAL_NS = 71
FINAL_FTOL = 1e-14
FINAL_NITER = 35000

# Every output file name contains this:
OUTPUT_NAME = "QP_scalar_optimized"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODES, MAXITER = [1], [4]
    FINAL_NS, FINAL_FTOL = 31, 1e-10

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"
inp = vj.VmecInput.from_file(INPUT_FILE)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
inp = replace(
    inp, rbc=rbc, zbs=zbs, delt=0.5,
    niter_array=np.array([300, 8000]),
    ftol_array=np.array([1e-11, 1e-12]), ns_array=np.array([25, 35]))


def mirror_excess(state, runtime):
    return jnp.maximum(opt.mirror_ratio(state, runtime) - MIRROR_LIMIT, 0.0)


def iota_floor(state, runtime):
    return jnp.maximum(IOTA_FLOOR - opt.min_abs_iota(state, runtime), 0.0)


def elongation_excess(state, runtime):
    return jnp.maximum(opt.max_elongation(state, runtime) - ELONGATION_LIMIT, 0.0)


### Set up the objective ######################################################


qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=0, helicity_n=1)
objective_terms = [
    (qs, 0.0, 1.0), (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (iota_floor, 0.0, 100.0), (mirror_excess, 0.0, 10.0),
    (elongation_excess, 0.0, 10.0)]


report = opt.EquilibriumReporter(
    ("QS total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"),
    ("elongation", opt.max_elongation, ".4f"),
    ("mirror", opt.mirror_ratio, ".4f"))
monitor = opt.OptimizationMonitor()
equilibrium = opt.solve_equilibrium(inp)

### Run the optimization ######################################################

for max_mode, maxiter in zip(MAX_MODES, MAXITER):
    print(f"\n===== scalar QP stage, max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)

    def loss(state, runtime, terms=objective_terms):
        """The scalar that is differentiated: 0.5 * r.T @ r over all residual rows."""
        rows = opt.residuals_from_tuples(state, runtime, terms)
        return 0.5 * jnp.vdot(rows, rows)

    problem = opt.VmecProblem.from_loss(
        inp, loss, max_mode=max_mode, vary_major_radius=VARY_MAJOR_RADIUS,
        use_ess=True, ess_alpha=ESS_ALPHA, restart_from=equilibrium,
        progress=False, evaluation_progress=False)
    print(f"dof_names = {problem.dof_names}")
    monitor.problem = problem
    if not ci_smoke:
        problem.compile_value_and_gradient()

    # SciPy works in dimensionless increments y, with x = x0 + step * y, so
    # every coefficient moves on a similar scale.
    x0 = problem.x0
    step = PARAMETER_STEP * problem.scales

    def value_and_gradient(y):
        """What SciPy calls.  The monitor caches it, so its callback never re-solves."""
        value, gradient = problem.value_and_grad(x0 + step * y)
        return monitor.cache_evaluation(x0 + step * y, value, step * gradient)

    def record(intermediate_result):
        """SciPy callback: one monitor row per accepted iterate."""
        monitor({"x": x0 + step * intermediate_result.x, "fun": intermediate_result.fun})

    initial_value = float(value_and_gradient(np.zeros_like(x0))[0])
    result = minimize(
        value_and_gradient, np.zeros_like(x0), jac=True, method="L-BFGS-B",
        bounds=[(-MAX_PARAMETER_CHANGE, MAX_PARAMETER_CHANGE)] * x0.size,
        callback=record,
        options={"maxiter": maxiter, "gtol": 1e-6, "ftol": 1e-12,
                 "maxls": 20, "maxcor": 20})
    print(f"scalar cost: {initial_value:.12e} -> {float(result.fun):.12e}")
    x = x0 + step * result.x
    inp = problem.input_from_x(x)
    equilibrium = problem.equilibrium_from_x(x)
    report(f"mode {max_mode}", equilibrium)

### Check the result ##########################################################

# The optimizer's grid is not the certificate: re-solve the optimized boundary
# on a finer radial grid to a tighter tolerance and quote that.
final_input = replace(
    inp, ns_array=np.array([FINAL_NS]), ftol_array=np.array([FINAL_FTOL]),
    niter_array=np.array([FINAL_NITER]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution,
    verbose=not ci_smoke, raise_on_max_iterations=True,
    polish_force_balance=POLISH_FORCE_BALANCE)
report("final", final_equilibrium)

### Print, plot and save ######################################################

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}\nWrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
