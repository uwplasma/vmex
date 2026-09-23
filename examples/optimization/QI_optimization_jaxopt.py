#!/usr/bin/env python
"""Optimize a constructed-QI boundary with a JAXopt solver.

VMEX exposes the problem through ``jax_value_and_grad`` and ``jax_residual``,
so an external JAX-native optimizer drives it directly. METHOD selects LBFGS,
which consumes the scalar value and gradient, or LM, a Levenberg-Marquardt
solver fed the residual vector through a custom JVP.

Neither solver is jitted: an equilibrium solve is a host callback, and only its
kernels run under JIT.
"""

import os
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np

import vmex as vj
from vmex import OptimizationMonitor, optimize as opt
from vmex.core.input import VmecInput
from vmex.core.qi import ConstructedQIResidual


# The seed deck:
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "input.QI_nfp2_initial"

# Highest boundary Fourier mode number that is varied:
MAX_MODE = 3

# JAXopt solver, and the iterations it may spend:
METHOD = "LM"                     # LBFGS's line search stalls on this problem
BUDGET = 3
LINE_SEARCH_STEPS = 10
INITIAL_STEP = 0.0                # LBFGS picks its own step when this is zero

# Flux surfaces the QI residual is evaluated on, and the Boozer resolution:
SURFACES = np.linspace(0.1, 1.0, 6)
QI_OPTIONS = dict(mboz=12, nboz=12, nphi=61, nalpha=18, n_bounce=21)

# Targets and limits:
ASPECT_TARGET = 8.0
IOTA_FLOOR = 0.3                  # floor on |mean iota|
ELONGATION_LIMIT = 8.0

# Step control:
BOUNDARY_STEP = 0.02              # metres represented by one scaled variable
VARY_MAJOR_RADIUS = False         # True optimizes RBC(0,0) instead of fixing it

# Equilibrium resolution: poloidal and toroidal mode numbers are MAX_MODE + 2,
# but never below MINIMUM_MPOL:
MINIMUM_MPOL = 5

# Verification solve of the optimized boundary:
FINAL_NS = 71
FINAL_FTOL = 1e-14
FINAL_NITER = 8000

# Every output file name contains this:
OUTPUT_NAME = f"QI_jaxopt_{METHOD}"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODE, BUDGET = 1, 1
    QI_OPTIONS = dict(mboz=8, nboz=8, nphi=31, nalpha=7, n_bounce=7)
    LINE_SEARCH_STEPS, INITIAL_STEP = 3, 1.0e-3
    FINAL_NS, FINAL_FTOL = 31, 1e-10

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

# JAX 0.9 removed this deprecated alias before JAXopt 0.8.3 stopped using it.
# Keep the compatibility local to this external-backend example.
if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree_util.tree_map

inp = VmecInput.from_file(INPUT_FILE)
mpol = max(MAX_MODE + 2, MINIMUM_MPOL)
inp = replace(inp, delt=0.5).change_resolution(
    mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)

### Set up the objective ######################################################

qi = ConstructedQIResidual(SURFACES, **QI_OPTIONS)


def iota_floor(equilibrium_state, solver_context):
    """Hinge on |mean iota| below its floor."""
    return jnp.maximum(
        IOTA_FLOOR - jnp.abs(opt.mean_iota(equilibrium_state, solver_context)), 0.0)


def elongation_excess(equilibrium_state, solver_context):
    """Hinge on the cross-section elongation above its limit."""
    return jnp.maximum(
        opt.max_elongation(equilibrium_state, solver_context) - ELONGATION_LIMIT, 0.0)


# Each term is (function, target, weight).
terms = [(qi, 0.0, 10.0), (opt.aspect_ratio, ASPECT_TARGET, 0.01),
         (iota_floor, 0.0, 10.0), (elongation_excess, 0.0, 1.0)]
problem = opt.VmecProblem.from_tuples(inp, terms, max_mode=MAX_MODE,
    vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True, progress=True, evaluation_progress=True)
print(f"dof_names = {problem.dof_names}")
x0, scales = jnp.asarray(problem.x0), BOUNDARY_STEP * jnp.asarray(problem.scales)

def x_from_y(y):
    return x0 + scales * y

def value_and_grad_y(y):
    value, gradient = problem.jax_value_and_grad(x_from_y(y))
    return value, scales * gradient

def optimality(state):
    gradient = getattr(state, "grad", getattr(state, "gradient", None))
    return float(state.error if gradient is None else jnp.linalg.norm(gradient, ord=jnp.inf))

### Run the optimization ######################################################

if METHOD == "LBFGS":
    problem.compile_value_and_gradient()
    solver = jaxopt.LBFGS(
        value_and_grad_y,
        value_and_grad=True,
        maxiter=BUDGET,
        maxls=LINE_SEARCH_STEPS,
        stepsize=INITIAL_STEP,
        jit=False,  # equilibrium is a host callback; only its kernels are jitted
    )
else:
    problem.compile_residual_and_jacobian()

    @jax.custom_jvp
    def residual(y):
        return problem.jax_residual(x_from_y(y))

    @residual.defjvp
    def residual_jvp(primals, tangents):
        y, = primals
        tangent, = tangents
        return residual(y), problem.jax_residual_jac(x_from_y(y)) @ (scales * tangent)

    solver = jaxopt.LevenbergMarquardt(
        residual,
        maxiter=BUDGET,
        materialize_jac=True,
        solver="cholesky",
        jit=False,
    )

params = jnp.zeros_like(x0)
state = solver.init_state(params)
monitor = OptimizationMonitor(problem)
monitor.record(x_from_y(params), cost=float(state.value), optimality=optimality(state), terms={})
for iteration in range(1, BUDGET + 1):
    params, state = solver.update(params, state)
    monitor.record(x_from_y(params), cost=float(state.value), optimality=optimality(state),
                   iteration=iteration, terms={})
    if float(state.error) <= solver.tol:
        break

### Check the result ##########################################################

x = x_from_y(params)
equilibrium = problem.equilibrium_from_x(x)
final_input = replace(problem.input_from_x(x), ns_array=np.array([FINAL_NS]),
                      ftol_array=np.array([FINAL_FTOL]), niter_array=np.array([FINAL_NITER]))
final_equilibrium = opt.solve_equilibrium(final_input, initial_state=equilibrium.solution,
                                          verbose=not ci_smoke, raise_on_max_iterations=True)

### Print, plot and save ######################################################

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"JAXopt {METHOD}: final cost = {float(problem.jax_fun(x)):.12e}, "
      f"QI total = {float(qi.total(final_equilibrium)):.6e}")
print(f"Wrote {input_path}\nWrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
