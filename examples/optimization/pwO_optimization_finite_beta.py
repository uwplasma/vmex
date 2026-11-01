#!/usr/bin/env python
"""Continue a vacuum pwO boundary to finite beta, with p proportional to 1-s.

Run ``pwO_optimization.py`` first. Pressure is calibrated at fixed toroidal
flux, then the workflow is that of the vacuum example with beta added to the
residual vector. NEO compares the vacuum start at this pressure with the result.
Requires vmex[neoclassical].
"""

import os
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.neoclassical import screen_transport
from vmex.core.pwo import MappedPwOResidual
from vmex.core.restart import restart_state

# Number of field periods, and the vacuum result this script continues:
NFP = 2
INPUT_FILE = Path(f"input.pwO_optimized_nfp{NFP}")
WOUT_FILE = Path(f"wout_pwO_optimized_nfp{NFP}.nc")

# Volume-averaged beta the pressure is calibrated to, and the optimizer's grid:
TARGET_BETA = 0.005
NS, FTOL, NITER = 31, 1.0e-12, 5500

# Flux surfaces the pwO residual is evaluated on, and its Boozer resolution:
SURFACES = np.linspace(0.25, 0.75, 3)
MBOZ = 8

# Mode ladder: highest boundary mode number varied in each stage, and the
# residual evaluations each stage may spend:
MAX_MODES = [2]
MAX_NFEV = [25]

# Targets:
ASPECT_TARGET = 6.0
MAGNETIC_WELL_TARGET = 0.01
IOTA_FLOOR = 0.42                 # minimum |iota| over the profile
MIRROR_LIMIT = 0.15               # maximum (Bmax-Bmin)/(Bmax+Bmin) on the pwO surfaces
PWO_WEIGHT = 1.0e3                # the seed's raw pwO variance is of order 0.01

# Step control. One scaled variable moves a low-order coefficient by
# PARAMETER_STEP metres, and a stage may move it MAX_PARAMETER_CHANGE steps:
PARAMETER_STEP = 0.02
MAX_PARAMETER_CHANGE = 1.0        # a continuation stays near its vacuum start
ESS_ALPHA = 1.2                   # smaller values let high Fourier modes move more
VARY_MAJOR_RADIUS = False         # True optimizes RBC(0,0) instead of fixing it

# Equilibrium resolution: poloidal and toroidal mode numbers are max_mode + 2,
# but never below MINIMUM_MPOL:
MINIMUM_MPOL = 5

# Verification solve of the optimized boundary:
FINAL_NS = 71
FINAL_FTOL = 1.0e-12
FINAL_NITER = 20000

# Every output file name contains this:
OUTPUT_NAME = f"pwO_finite_beta_nfp{NFP}"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODES, MAX_NFEV = [1], [4]
    FINAL_NS, FINAL_FTOL = 31, 1.0e-10

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

inp = vj.VmecInput.from_file(INPUT_FILE)
inp = replace(inp, am=np.array([1.0, -1.0]), pres_scale=1.0, ns_array=np.array([NS]),
              ftol_array=np.array([FTOL]), niter_array=np.array([NITER]))
calibration = opt.solve_equilibrium(
    inp, initial_state=restart_state(vj.read_wout(WOUT_FILE), inp),
    raise_on_max_iterations=True)
inp = replace(inp, pres_scale=TARGET_BETA / float(calibration.wout.betatotal))
restart = calibration.solution

### Set up the objective ######################################################

def iota_floor(equilibrium_state, solver_context):
    """Hinge on the profile minimum of |iota|: a mean target can hide a near-zero surface."""
    return jnp.maximum(
        IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)


def mirror_excess(equilibrium_state, solver_context):
    """Hinge on the full-surface mirror ratio, which the mapped variance does not see."""
    mirror = pwo.diagnostics_state(equilibrium_state, solver_context)[3]
    return jnp.maximum(mirror - MIRROR_LIMIT, 0.0)


# Each term is (function, target, weight).
pwo = MappedPwOResidual(SURFACES, mboz=MBOZ, nboz=MBOZ, oversample=2)
objective_function_terms = [
    (pwo, 0.0, PWO_WEIGHT),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (iota_floor, 0.0, 100.0),
    (opt.magnetic_well, MAGNETIC_WELL_TARGET, 100.0),
    (mirror_excess, 0.0, 100.0),
    (opt.volume_average_beta, TARGET_BETA, 1.0 / TARGET_BETA**2),
]

report = opt.EquilibriumReporter(
    ("pwO total", pwo.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("beta", opt.volume_average_beta, ".3%"), ("mean iota", opt.mean_iota, ".4f"), ("magnetic well", opt.magnetic_well, ".4f"))
monitor = opt.OptimizationMonitor()

### Run the optimization ######################################################

seed_input = inp
equilibrium = opt.solve_equilibrium(inp, initial_state=restart)
report("seed", equilibrium)
# Stages that share a boundary resolution share ONE problem, as in
# QA_optimization.py: every jitted graph is compiled once for the whole group.
resolution_of = {max_mode: max(max_mode + 2, MINIMUM_MPOL) for max_mode in MAX_MODES}
group_max_mode = {
    max_mode: max(other for other in MAX_MODES
                  if resolution_of[other] == resolution_of[max_mode])
    for max_mode in MAX_MODES}
problem, mpol = None, None
for max_mode, max_nfev in zip(MAX_MODES, MAX_NFEV):
    print(f"\n===== finite-beta pwO stage, max_mode = {max_mode} =====")
    if resolution_of[max_mode] != mpol:
        mpol = resolution_of[max_mode]
        inp = replace(inp, delt=0.5).change_resolution(
            mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
        problem = opt.VmecProblem.from_tuples(
            inp, objective_function_terms, max_mode=group_max_mode[max_mode],
            vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True,
            ess_alpha=ESS_ALPHA, restart_from=equilibrium,
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

# The optimizer's grid is not the certificate: re-solve the optimized boundary
# on a finer radial grid to a tighter tolerance and quote that.
final_resolution = dict(ns_array=np.array([FINAL_NS]), ftol_array=np.array([FINAL_FTOL]),
                        niter_array=np.array([FINAL_NITER]))
final_input = replace(inp, **final_resolution)
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution,
    verbose=not ci_smoke, raise_on_max_iterations=True)

# Matched NEO: the seed on the same radial grid, the same half-mesh surfaces.
seed_equilibrium = opt.solve_equilibrium(
    replace(seed_input, **final_resolution), verbose=False, raise_on_max_iterations=True)
half_mesh = (np.arange(FINAL_NS - 1) + 0.5) / (FINAL_NS - 1)
neo_surfaces = half_mesh[np.abs(half_mesh[:, None] - SURFACES).argmin(axis=0)]
seed_Q = screen_transport(seed_equilibrium.wout, neo_surfaces)
final_Q = screen_transport(final_equilibrium.wout, neo_surfaces)

### Print, plot and save ######################################################

final_total = report("final", final_equilibrium)["pwO total"]
final_beta = float(opt.volume_average_beta(final_equilibrium.solution,
                                           final_equilibrium.solver_context))
opt.report_targets(final_equilibrium, aspect=ASPECT_TARGET, iota_floor=IOTA_FLOOR,
                   well_floor=MAGNETIC_WELL_TARGET,
                   extra=[("beta", final_beta, 0.9 * TARGET_BETA, "min"),
                          ("beta", final_beta, 1.1 * TARGET_BETA, "max")])
print(f"\npwO total {final_total:.3e}")
print(f"epsilon_eff^(3/2) at s = {neo_surfaces}: seed {seed_Q}, final {final_Q}")

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}\nWrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
