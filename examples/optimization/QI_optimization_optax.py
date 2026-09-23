#!/usr/bin/env python
"""Optimize a constructed-QI boundary with an Optax gradient transform.

VMEX exposes the problem through ``jax_value_and_grad``, so any Optax chain
drives it: here gradient clipping followed by Adam, stepped in a plain Python
loop rather than by a line-searching optimizer. That is the trade — a fixed
step schedule, no line search, and the monitor records every step.
"""

import os
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import optax

import vmex as vj
from vmex import OptimizationMonitor
from vmex import optimize as opt
from vmex.core.input import VmecInput
from vmex.core.qi import ConstructedQIResidual


# The seed deck:
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "input.QI_nfp2_initial"

# Highest boundary Fourier mode number that is varied, and the Adam steps:
MAX_MODE = 2
STEPS = 20

# Optax transform: gradient clipping, then Adam at this learning rate:
GRADIENT_CLIP = 1.0
LEARNING_RATE = 1.0e-2

# Flux surfaces the QI residual is evaluated on, and the Boozer resolution:
SURFACES = np.linspace(0.1, 1.0, 6)
QI_OPTIONS = dict(mboz=12, nboz=12, nphi=61, nalpha=18, n_bounce=21)

# Targets and limits:
ASPECT_TARGET = 6.0
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
OUTPUT_NAME = "QI_optax_adam"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODE, STEPS = 1, 1
    QI_OPTIONS = dict(mboz=8, nboz=8, nphi=31, nalpha=7, n_bounce=7)
    FINAL_NS, FINAL_FTOL = 31, 1e-10

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

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
terms = [(qi, 0.0, 10.0), (opt.aspect_ratio, ASPECT_TARGET, 0.005),
         (iota_floor, 0.0, 10.0), (elongation_excess, 0.0, 1.0)]
problem = opt.VmecProblem.from_tuples(inp, terms, max_mode=MAX_MODE,
    vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True, progress=True, evaluation_progress=True)
print(f"dof_names = {problem.dof_names}")
problem.compile_value_and_gradient()
transform = optax.chain(
    optax.clip_by_global_norm(GRADIENT_CLIP),
    optax.adam(LEARNING_RATE),
)
x0, scales = jnp.asarray(problem.x0), BOUNDARY_STEP * jnp.asarray(problem.scales)
y = jnp.zeros_like(x0)
state = transform.init(y)
monitor = OptimizationMonitor(problem)

### Run the optimization ######################################################

for iteration in range(STEPS):
    x = x0 + scales * y
    value, gradient_x = problem.jax_value_and_grad(x)
    gradient = scales * gradient_x
    updates, state = transform.update(gradient, state, y)
    y = optax.apply_updates(y, updates)
    monitor.record(
        x,
        cost=float(value),
        optimality=float(jnp.linalg.norm(gradient, ord=jnp.inf)),
        iteration=iteration,
    )

### Check the result ##########################################################

x = x0 + scales * y
equilibrium = problem.equilibrium_from_x(x)
final_input = replace(problem.input_from_x(x), ns_array=np.array([FINAL_NS]),
                      ftol_array=np.array([FINAL_FTOL]), niter_array=np.array([FINAL_NITER]))
final_equilibrium = opt.solve_equilibrium(final_input, initial_state=equilibrium.solution,
                                          verbose=not ci_smoke, raise_on_max_iterations=True)

### Print, plot and save ######################################################

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Optax Adam: final cost = {float(problem.jax_fun(x)):.12e}, "
      f"QI total = {float(qi.total(final_equilibrium)):.6e}")
print(f"Wrote {input_path}\nWrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
