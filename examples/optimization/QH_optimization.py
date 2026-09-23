#!/usr/bin/env python
"""Optimize a boundary for quasi-helical symmetry, optionally with a stability proxy.

SciPy nonlinear least squares varies the boundary Fourier coefficients of a
vacuum equilibrium in stages of increasing mode number. The residual vector
holds the quasisymmetry ratio and the aspect ratio, and VMEX supplies its exact
Jacobian.

Setting ``USE_TRIAL_STABILITY`` adds trial-pressure Mercier and
resistive-interchange residuals from the second stage on. Those evaluate the
stability criteria a small pressure *would* produce on the frozen vacuum
geometry; they are a screen, not a finite-pressure certificate.
"""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt

# Number of field periods, and the seed deck the boundary is shaped from:
NFP = 4
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"

# Rotating-ellipse amplitude added to the circular seed. A circular torus
# cannot acquire iota to first order; this starts the optimizer in the QH basin:
SEED_PERTURBATION = 0.12

# Flux surfaces the quasisymmetry residual is evaluated on:
SURFACES = np.linspace(0.1, 1.0, 10)

# Mode ladder: highest boundary mode number varied in each stage, and the
# residual evaluations each stage may spend:
MAX_MODES = [1, 2]
MAX_NFEV = [10, 15]

# Targets:
ASPECT_TARGET = 6.0

# Trial-pressure stability screen, added from the second stage on. TRIAL_BETA
# is the pressure proxy: beta = 2.5 %, p(s) proportional to 1 - s:
USE_TRIAL_STABILITY = False
TRIAL_BETA = 0.025
STABILITY_COST_PER_SURFACE = 1.0e-2
EDGE_WEIGHT_FACTOR = 10.0         # how much harder the edge is weighted than s = 0.2
STABILITY_MIN_S = 0.2             # the singular core is excluded below this
STABILITY_MARGIN = 1.0e-3

# Step control. One scaled variable moves a low-order coefficient by
# PARAMETER_STEP metres, and a stage may move it MAX_PARAMETER_CHANGE steps:
PARAMETER_STEP = 0.02
MAX_PARAMETER_CHANGE = 5.0
ESS_ALPHA = 1.2                   # smaller values let high Fourier modes move more
VARY_MAJOR_RADIUS = False         # True optimizes RBC(0,0) instead of fixing it

# Equilibrium resolution: poloidal and toroidal mode numbers are max_mode + 2,
# but never below MINIMUM_MPOL:
MINIMUM_MPOL = 5

# Verification solve of the optimized boundary:
FINAL_NS = 71
FINAL_FTOL = 1.0e-14
FINAL_NITER = 8000

# Every output file name contains this:
OUTPUT_NAME = "QH_optimized"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODES, MAX_NFEV = [1], [4]
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

### Set up the objective ######################################################

def trial_dmerc(equilibrium_state, solver_context):
    """Mercier residual a TRIAL_BETA pressure would produce on this geometry."""
    return opt.trial_pressure_mercier_stability_residual(
        equilibrium_state, solver_context, beta=TRIAL_BETA, margin=STABILITY_MARGIN)


def trial_dr(equilibrium_state, solver_context):
    """Resistive-interchange counterpart of trial_dmerc."""
    return opt.trial_pressure_glasser_stability_residual(
        equilibrium_state, solver_context, beta=TRIAL_BETA, margin=STABILITY_MARGIN)


# Zero weight on the singular core, rising smoothly toward the difficult edge.
stability_s = np.linspace(0.0, 1.0, int(inp.ns_array[-1]))[2:-1]
stability_shape = np.where(stability_s >= STABILITY_MIN_S,
    1.0 + (EDGE_WEIGHT_FACTOR - 1.0) * stability_s**4, 0.0)

# Each term is (function, target, weight).
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=-1)
objective_function_terms = [
    (qs, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
]

report = opt.EquilibriumReporter(
    ("QS total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"), ("magnetic well", opt.magnetic_well, ".4f"))
monitor = opt.OptimizationMonitor()

### Run the optimization ######################################################

equilibrium = opt.solve_equilibrium(inp)
# Consecutive stages that share a boundary resolution and a term list share ONE
# problem: problem.subproblem() frees that stage's boundary harmonics and
# freezes the higher ones, so the whole group is traced and compiled once.
# Rebuilding per stage is a cache miss by construction (the decision vector
# changes length, although MINIMUM_MPOL keeps every array shape inside the
# solve identical), and that recompilation is about half of a shipped ladder
# run. A stage that raises mpol -- or, with USE_TRIAL_STABILITY, one that adds
# residual rows -- starts a new group.
resolution_of = {max_mode: max(max_mode + 2, MINIMUM_MPOL) for max_mode in MAX_MODES}
problem, mpol, built_terms = None, None, None
for stage, (max_mode, max_nfev) in enumerate(zip(MAX_MODES, MAX_NFEV)):
    print(f"\n===== QH stage, max_mode = {max_mode} =====")
    stage_terms = objective_function_terms
    if USE_TRIAL_STABILITY and stage > 0:
        # Normalize each dimensional row at the established QH seed, so the
        # Mercier rows cannot overwhelm quasisymmetry.
        dmerc0 = np.asarray(trial_dmerc(equilibrium.solution, equilibrium.solver_context))
        dr0 = np.asarray(trial_dr(equilibrium.solution, equilibrium.solver_context))
        stability_scale = np.maximum.reduce((np.abs(dmerc0), np.abs(dr0), np.ones_like(dmerc0)))
        stability_weights = STABILITY_COST_PER_SURFACE * stability_shape / stability_scale**2
        stage_terms = [*objective_function_terms,
            (trial_dmerc, 0.0, stability_weights), (trial_dr, 0.0, stability_weights)]
        print(f"Adding trial-pressure stability on s >= {STABILITY_MIN_S:.1f}; "
              "weights rise smoothly toward the edge.")
    if resolution_of[max_mode] != mpol or stage_terms is not built_terms:
        mpol, built_terms = resolution_of[max_mode], stage_terms
        group = max_mode  # this group runs while the resolution holds -- and,
        if not USE_TRIAL_STABILITY:  # with trial stability, ends here, since
            for other in MAX_MODES[stage:]:  # every later stage adds rows
                if resolution_of[other] != mpol:
                    break
                group = max(group, other)
        inp = replace(inp, delt=0.5).change_resolution(
            mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
        # A RuntimeWarning about uncertified Jacobian columns is expected once
        # the optimizer leaves the seed and needs no action; see
        # examples/README.md.
        problem = opt.VmecProblem.from_tuples(inp, stage_terms, max_mode=group,
            vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True, ess_alpha=ESS_ALPHA,
            restart_from=equilibrium)
        x = problem.x0
        monitor.problem = problem
        if not ci_smoke:
            problem.compile_residual_and_jacobian()
    stage_problem = problem.subproblem(max_mode=max_mode, x=x)
    print(f"dof_names = {stage_problem.dof_names}")
    step = PARAMETER_STEP * stage_problem.scales
    result = least_squares(
        stage_problem.residual, stage_problem.x0, jac=stage_problem.residual_jac,
        x_scale=step, bounds=(stage_problem.x0 - MAX_PARAMETER_CHANGE * step,
                              stage_problem.x0 + MAX_PARAMETER_CHANGE * step),
        max_nfev=max_nfev, ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor
    )
    x = stage_problem.embed(result.x)
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
    verbose=not ci_smoke, raise_on_max_iterations=True)

### Print, plot and save ######################################################

final_total = report("final", final_equilibrium)["QS total"]
print(f"\nQS total {final_total:.3e}")
if USE_TRIAL_STABILITY:
    final_s = np.linspace(0.0, 1.0, int(final_input.ns_array[-1]))[2:-1]
    keep = final_s >= STABILITY_MIN_S
    final_dmerc = np.asarray(opt.trial_pressure_d_merc_state(
        final_equilibrium.solution, final_equilibrium.solver_context, beta=TRIAL_BETA))[2:-1]
    final_dr = np.asarray(opt.trial_pressure_glasser_d_r_state(
        final_equilibrium.solution, final_equilibrium.solver_context, beta=TRIAL_BETA,
        shear_epsilon=1.0e-8))[2:-1]
    print(f"Trial-pressure proxy on s >= {STABILITY_MIN_S:.1f}: "
          f"min DMerc = {final_dmerc[keep].min():.3e}, max DR = {final_dr[keep].max():.3e}")

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}\nWrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
