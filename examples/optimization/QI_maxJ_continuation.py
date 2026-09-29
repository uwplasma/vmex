#!/usr/bin/env python
"""Continue a constructed-QI boundary into a maximum-J field.

The ladder starts from the bundled nfp = 3 QI equilibrium, because a common
physical pitch generally does not exist on a circular boundary: evaluating
dJ/ds before a QI basin exists would compare different trapped-particle
populations on adjacent surfaces, and forming that basin from a minimal seed
takes far more evaluations than this example budgets (a 25-evaluation QI
ladder from the rotating ellipse left min |iota| at 0.73 and lost the matched
wells). Each stage adds the J-invariance and maximum-J residuals,
strengthening their weights and tightening the maximum-J target, while the
aspect ratio is held at the seed's.

The trapped pitches are selected once after the weak first maximum-J stage and
held fixed afterwards, so every later stage differentiates the same particles.
The script refuses to continue if the seed loses usable wells.
"""

import os
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.maxj import (
    JInvariantQIAndMaximumJResidual, common_trapped_pitches_state,
)
from vmex.core.qi import ConstructedQIResidual

# The QI equilibrium the continuation starts from, and the radial grid and
# tolerance its stages solve on:
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "input.nfp3_QI_fixed_resolution_final"
STAGE_NS, STAGE_FTOL, STAGE_NITER = 31, 1e-11, 4000

# Flux surfaces every residual is evaluated on:
SURFACES = np.array([0.20, 0.35, 0.50, 0.65, 0.80, 0.90])

# Maximum-J ladder, one entry per stage:
MAX_MODES = [3, 3, 3]
MAX_NFEV = [4, 4, 8]
MAXIMUM_J_TARGETS = [0.0, -0.002, -0.005]
MAXIMUM_J_WEIGHTS = [500.0, 2.0e3, 5.0e3]
QI_INVARIANCE_WEIGHTS = [1.0e3, 5.0e3, 1.0e4]
CONSTRUCTED_QI_WEIGHTS = [1.0e4, 1.0e4, 1.0e4]
MAGNETIC_WELL_WEIGHTS = [100.0, 1.0e3, 1.0e3]
MAXJ_ESS_ALPHA = 0.7

# Targets and limits (the aspect ratio is held at the seed's):
IOTA_FLOOR = 1.03                 # minimum |iota| over the profile
MIRROR_LIMIT = 0.35
MAGNETIC_WELL_TARGET = 0.01

# Boozer resolution of the constructed-QI residual:
QI_MBOZ = 14
QI_OPTIONS = dict(nphi=61, nalpha=18, n_bounce=21)

# Bounce-action quadrature. The last stages cover a full poloidal transit and
# more alpha values, which removes an alias a short coarse trace misses:
COARSE_ACTION = dict(nalpha=5, points_per_period=24, num_periods=6,
                     max_wells=16, quadrature_order=16)
RESOLVED_ACTION = dict(nalpha=9, points_per_period=32, num_periods=10,
                       max_wells=24, quadrature_order=24)
ACTION_MBOZ = [8, 8, 10]
ACTION_OPTIONS = [COARSE_ACTION, COARSE_ACTION, RESOLVED_ACTION]

# Field strengths that trap the same particles on every sampled line:
TRAPPING_DEPTHS = (0.35, 0.55, 0.75)

# Step control. A local trust region: large enough to move, small enough to
# preserve the trapped wells:
BOUNDARY_STEP = 0.05
MAXJ_FORWARD_FTOL = 1e-7
MAXJ_FORWARD_ITERATIONS = 800
VARY_MAJOR_RADIUS = False         # True optimizes RBC(0,0) instead of fixing it

# Verification solve of the optimized boundary:
FINAL_NS = 71
FINAL_FTOL = 1e-14
FINAL_NITER = 8000

# Every output file name contains this:
OUTPUT_NAME = "QI_maxJ_optimized"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    STAGE_NS, STAGE_FTOL, STAGE_NITER = 11, 1e-8, 2000
    MAX_MODES, MAX_NFEV = [1], [2]
    MAXIMUM_J_TARGETS, MAXIMUM_J_WEIGHTS = [0.0], [500.0]
    QI_INVARIANCE_WEIGHTS = [1.0e3]
    CONSTRUCTED_QI_WEIGHTS = [1.0e4]
    MAGNETIC_WELL_WEIGHTS = [100.0]
    SURFACES, TRAPPING_DEPTHS = np.array([0.25, 0.45, 0.65, 0.85]), (0.5,)
    QI_MBOZ = 8
    QI_OPTIONS = dict(nphi=25, nalpha=5, n_bounce=5)
    ACTION_MBOZ = [8]
    ACTION_OPTIONS = [COARSE_ACTION]
    FINAL_NS, FINAL_FTOL = 31, 1e-10

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

inp = replace(vj.VmecInput.from_file(INPUT_FILE), ns_array=np.array([STAGE_NS]),
              ftol_array=np.array([STAGE_FTOL]), niter_array=np.array([STAGE_NITER]))
equilibrium = opt.solve_equilibrium(inp)
ASPECT_TARGET = float(opt.aspect_ratio(equilibrium.solution, equilibrium.solver_context))

### Set up the objective ######################################################

qi = ConstructedQIResidual(SURFACES, mboz=QI_MBOZ, nboz=QI_MBOZ, **QI_OPTIONS)


def mirror_excess(equilibrium_state, solver_context):
    """Hinge on the mirror ratio above its limit; zero while the limit holds."""
    return jnp.maximum(
        opt.mirror_ratio(equilibrium_state, solver_context) - MIRROR_LIMIT, 0.0)


def iota_floor(equilibrium_state, solver_context):
    """Hinge on the profile minimum of |iota|: a mean target can hide a near-zero surface.

    opt.mean_iota targets the average instead; opt.soft_min_abs_iota is the smooth minimum.
    """
    return jnp.maximum(
        IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)


def magnetic_well_floor(equilibrium_state, solver_context):
    """Hinge below the magnetic-well target; zero once the well is deep enough."""
    return jnp.maximum(
        MAGNETIC_WELL_TARGET - opt.magnetic_well(equilibrium_state, solver_context), 0.0)


report = opt.EquilibriumReporter(
    ("QI", qi.total, ".4e"), ("aspect", opt.aspect_ratio, ".3f"),
    ("iota", opt.mean_iota, ".3f"), ("mirror", opt.mirror_ratio, ".3f"),
    ("magnetic well", opt.magnetic_well, ".3f"))
monitor = opt.OptimizationMonitor()

### Run the optimization ######################################################

report("seed", equilibrium)

# Select field strengths that trap the same particles on every sampled line.
# After the weak first maximum-J stage, keep those pitches fixed so every
# later stage differentiates the same particle population.
pitch = np.asarray(common_trapped_pitches_state(
    equilibrium.solution, equilibrium.solver_context, SURFACES, TRAPPING_DEPTHS))
for stage, (max_mode, max_nfev, maxj_target, maxj_weight, qi_weight,
            constructed_weight, well_weight, action_mboz, action_options) in enumerate(zip(
        MAX_MODES, MAX_NFEV, MAXIMUM_J_TARGETS, MAXIMUM_J_WEIGHTS,
        QI_INVARIANCE_WEIGHTS, CONSTRUCTED_QI_WEIGHTS, MAGNETIC_WELL_WEIGHTS,
        ACTION_MBOZ, ACTION_OPTIONS)):
    print(f"\n===== QI + maximum-J stage, max_mode = {max_mode}, "
          f"target = {maxj_target:g}, weight = {maxj_weight:g} =====")
    # Re-select the common physical pitches once after the weak first stage,
    # then keep the same trapped particles in every stronger stage.
    if stage == 1:
        pitch = np.asarray(common_trapped_pitches_state(
            equilibrium.solution, equilibrium.solver_context, SURFACES, TRAPPING_DEPTHS))
    qi_maxj = JInvariantQIAndMaximumJResidual(SURFACES, pitch,
        mboz=action_mboz, nboz=action_mboz,
        qi_options=action_options, qi_weight=qi_weight,
        maxj_weight=maxj_weight,
        maxj_options={**action_options, "target": maxj_target})
    maxj_diagnostics = qi_maxj.compute_state(
        equilibrium.solution, equilibrium.solver_context)["maximum_j"]
    if not bool(jnp.all(maxj_diagnostics["valid_pitch_pair"])):
        raise RuntimeError(
            "the equilibrium no longer has usable trapped wells on every sampled "
            "surface; reduce BOUNDARY_STEP or the maximum-J weights")
    stage_shape_terms = [
        (qi, 0.0, constructed_weight),
        (opt.aspect_ratio, ASPECT_TARGET, 1.0),
        (iota_floor, 0.0, 10.0),
        (mirror_excess, 0.0, 100.0),
        (magnetic_well_floor, 0.0, well_weight),
    ]
    objective_function_terms = [(qi_maxj, 0.0, 1.0), *stage_shape_terms]
    problem = opt.VmecProblem.from_tuples(inp, objective_function_terms, max_mode=max_mode,
        vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True, ess_alpha=MAXJ_ESS_ALPHA,
        restart_from=equilibrium, forward_ftol=MAXJ_FORWARD_FTOL,
        forward_max_iterations=MAXJ_FORWARD_ITERATIONS, progress=not ci_smoke)
    print(f"dof_names = {problem.dof_names}")
    monitor.problem = problem
    step = BOUNDARY_STEP * problem.scales
    result = least_squares(problem.residual, problem.x0, jac=problem.residual_jac,
        x_scale=problem.scales, bounds=(problem.x0 - step, problem.x0 + step), max_nfev=max_nfev,
        ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor)
    print(f"normalized boundary displacement = "
          f"{np.linalg.norm((result.x - problem.x0) / problem.scales):.3e}")
    inp = problem.input_from_x(result.x)
    equilibrium = problem.equilibrium_from_x(result.x)
    report(f"mode {max_mode}", equilibrium)
    stage_maxj = qi_maxj.compute_state(
        equilibrium.solution, equilibrium.solver_context)["maximum_j"]
    print(f"actual-field maximum-J fraction = "
          f"{float(stage_maxj['maximum_j_fraction']):.1%}")

### Check the result ##########################################################

# The optimizer's grid is not the certificate: re-solve on a finer radial grid.
final_input = replace(inp, ns_array=np.array([FINAL_NS]),
    ftol_array=np.array([FINAL_FTOL]), niter_array=np.array([FINAL_NITER]))
final_equilibrium = opt.solve_equilibrium(final_input, initial_state=equilibrium.solution,
    verbose=not ci_smoke, raise_on_max_iterations=True)

### Print, plot and save ######################################################

report("final", final_equilibrium)
opt.report_targets(final_equilibrium, aspect=ASPECT_TARGET, iota_floor=IOTA_FLOOR,
                   mirror_limit=MIRROR_LIMIT, well_floor=MAGNETIC_WELL_TARGET)
# The final report repeats the actual-field J-invariance and matched-well dJ/ds
# at the most resolved action quadrature used by the continuation.
qi_maxj_certificate = JInvariantQIAndMaximumJResidual(SURFACES, pitch,
    mboz=ACTION_MBOZ[-1], nboz=ACTION_MBOZ[-1],
    qi_options=ACTION_OPTIONS[-1], qi_weight=1.0, maxj_weight=1.0,
    maxj_options={**ACTION_OPTIONS[-1], "target": MAXIMUM_J_TARGETS[-1]})
diagnostics = qi_maxj_certificate.compute_state(
    final_equilibrium.solution, final_equilibrium.solver_context)
print(f"J-invariance = {float(diagnostics['qi']['total']):.4e}, "
      f"maximum-J = {float(diagnostics['maximum_j']['total']):.4e}, "
      f"maximum-J fraction = {float(diagnostics['maximum_j']['maximum_j_fraction']):.1%}, "
      f"target-margin fraction = {float(diagnostics['maximum_j']['target_fraction']):.1%}")
input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}\nWrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
for path in vj.plot_wout(wout_path, ".", j_pitch=float(pitch[0])).values():
    print(f"Wrote {path}")
