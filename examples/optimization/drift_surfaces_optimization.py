#!/usr/bin/env python
"""Optimize a boundary so that trapped-particle drift surfaces close inside it.

SciPy nonlinear least squares varies the boundary Fourier coefficients of a
vacuum equilibrium in stages of increasing mode number, exactly as in
``QA_optimization.py``, with quasisymmetry replaced by drift confinement. For
six physical pitches across the trapped window,
:class:`~vmex.core.maxj.DriftConfinementResidual` launches bounce-centre
orbits on three surfaces and eight field-line labels, traces each drift
surface once, and penalizes how far each drift
surface reaches toward the LCFS. The weighted mean reach is reported as
``Gamma_W``. No symmetry is imposed. VMEX supplies the exact Jacobian.
"""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.maxj import DriftConfinementResidual, common_trapped_pitches_state

# Number of field periods, and the seed deck the boundary is shaped from:
NFP = 2
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"

# Rotating-ellipse amplitude added to the circular seed. The exactly circular
# torus has zero first-order iota sensitivity:
SEED_PERTURBATION = 0.05

# Boozer surfaces the action J is sampled on, launch radii of the drift
# orbits, and the trapping depths in [0, 1] (shallowest to deepest common
# well) that choose the physical pitches:
SURFACES = np.linspace(0.1, 1.0, 10)
START_RADII = (0.15, 0.25, 0.4)
TRAPPING_DEPTHS = (0.1, 0.25, 0.4, 0.55, 0.7, 0.85)

# Drift-orbit sampling: field lines, traced periods, quadrature, RK4 steps,
# and the arc length each drift surface is traced over in the polar plane:
DRIFT_OPTIONS = dict(nalpha=32, points_per_period=48, num_periods=16,
                     max_wells=40, quadrature_order=24, nsteps=512, length=8.0)
BOOZER_MODES = 10

# Mode ladder: highest boundary mode number varied in each stage, and the
# residual evaluations each stage may spend:
MAX_MODES = [1, 2]
MAX_NFEV = [10, 15]

# Targets:
ASPECT_TARGET = 6.0
MAGNETIC_WELL_TARGET = 0.01
IOTA_FLOOR = 0.42                 # minimum |iota| over the profile
MIRROR_LIMIT = 0.25               # (Bmax - Bmin)/(Bmax + Bmin) on the LCFS

# Step control. One scaled variable moves a low-order coefficient by
# PARAMETER_STEP metres, and a stage may move it MAX_PARAMETER_CHANGE steps:
PARAMETER_STEP = 0.02
MAX_PARAMETER_CHANGE = 5.0
ESS_ALPHA = 1.2                   # smaller values let high Fourier modes move more

# Equilibrium resolution: poloidal and toroidal mode numbers are max_mode + 2,
# but never below MINIMUM_MPOL:
MINIMUM_MPOL = 5

# Verification solve of the optimized boundary:
FINAL_NS = 51
FINAL_FTOL = 1.0e-14
FINAL_NITER = 8000

# Every output file name contains this:
OUTPUT_NAME = f"drift_surfaces_nfp{NFP}_optimized"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODES, MAX_NFEV = [1], [2]
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
equilibrium = opt.solve_equilibrium(inp)

### Set up the objective ######################################################

# One physical lambda must describe the same particles on every radius and
# field-line label, so the pitches are chosen once, on the seed, and held fixed.
pitch = np.asarray(common_trapped_pitches_state(
    equilibrium.solution, equilibrium.solver_context, SURFACES, TRAPPING_DEPTHS,
    nalpha=DRIFT_OPTIONS["nalpha"], points_per_period=DRIFT_OPTIONS["points_per_period"],
    num_periods=DRIFT_OPTIONS["num_periods"]))
drift = DriftConfinementResidual(SURFACES, pitch, START_RADII,
                                 mboz=BOOZER_MODES, nboz=BOOZER_MODES, **DRIFT_OPTIONS)

# Each term is (function, target, weight).
objective_function_terms = [
    (drift, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (opt.min_abs_iota, opt.at_least(IOTA_FLOOR), 10.0),
    (opt.magnetic_well, MAGNETIC_WELL_TARGET, 1.0),
    (opt.mirror_ratio, opt.at_most(MIRROR_LIMIT), 10.0),
]


def gamma_w(equilibrium):
    """Weighted mean reach of the sampled drift orbits, 0 when all stay put."""
    return float(drift.compute_state(equilibrium.state, equilibrium.runtime)["gamma_w"])


report = opt.EquilibriumReporter(
    ("drift total", drift.total, ".6e"), ("Gamma_W", gamma_w, ".4f"),
    ("aspect", opt.aspect_ratio, ".4f"), ("mean iota", opt.mean_iota, ".4f"),
    ("magnetic well", opt.magnetic_well, ".4f"), ("mirror", opt.mirror_ratio, ".4f"))
monitor = opt.OptimizationMonitor()
report("seed", equilibrium)

### Run the optimization ######################################################

# Stages that share a boundary resolution share one problem; see
# QA_optimization.py for why problem.subproblem() avoids recompilation.
resolution_of = {max_mode: max(max_mode + 2, MINIMUM_MPOL) for max_mode in MAX_MODES}
group_max_mode = {
    max_mode: max(other for other in MAX_MODES
                  if resolution_of[other] == resolution_of[max_mode])
    for max_mode in MAX_MODES}
problem, mpol = None, None
for max_mode, max_nfev in zip(MAX_MODES, MAX_NFEV):
    print(f"\n===== drift-surface stage, max_mode = {max_mode} =====")
    if resolution_of[max_mode] != mpol:
        mpol = resolution_of[max_mode]
        inp = replace(inp, delt=0.5).change_resolution(
            mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
        problem = opt.VmecProblem.from_tuples(
            inp, objective_function_terms, max_mode=group_max_mode[max_mode],
            use_ess=True, ess_alpha=ESS_ALPHA, restart_from=equilibrium,
            progress=True)
        x = problem.x0
        monitor.problem = problem
        if not ci_smoke:
            problem.compile_residual_and_jacobian()
    stage = problem.subproblem(max_mode=max_mode, x=x)
    print(f"dof_names = {stage.dof_names}")
    step = PARAMETER_STEP * stage.scales
    result = least_squares(
        stage.residual, stage.x0, jac=stage.residual_jac,
        x_scale=step, max_nfev=max_nfev,
        bounds=(stage.x0 - MAX_PARAMETER_CHANGE * step,
                stage.x0 + MAX_PARAMETER_CHANGE * step),
        ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor)
    x = stage.embed(result.x)
    inp = problem.input_from_x(x)
    equilibrium = problem.equilibrium_from_x(x)
    report(f"mode {max_mode}", equilibrium)

### Check the result ##########################################################

# Re-solve the optimized boundary on a finer radial grid to a tighter tolerance.
final_input = replace(
    inp, ns_array=np.array([FINAL_NS]), ftol_array=np.array([FINAL_FTOL]),
    niter_array=np.array([FINAL_NITER]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution,
    verbose=not ci_smoke, raise_on_max_iterations=True)

### Print, plot and save ######################################################

final_total = report("final", final_equilibrium)["drift total"]
opt.report_targets(final_equilibrium, aspect=ASPECT_TARGET, iota_floor=IOTA_FLOOR,
                   well_floor=MAGNETIC_WELL_TARGET)
print(f"\ndrift total {final_total:.3e}")

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}\nWrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
