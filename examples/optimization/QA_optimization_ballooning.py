#!/usr/bin/env python
"""Optimize a quasi-axisymmetric boundary against the infinite-n ballooning bound.

The seed (``input.nfp2_QA_finite_beta``, beta = 2.7 %) is Mercier-STABLE and
ballooning-UNSTABLE, which is the case a ballooning objective is for: the
interchange criteria see nothing to fix.

The optimized quantity is ``ballooning_growth_rate``, a smooth softmax upper
bound on the growth rate over the sampled field lines, so driving it below zero
is a sufficient condition for every sampled line to be stable. The hard maximum
is what gets reported. The bound sits above the hard maximum by at most
TEMPERATURE * log(number of lines).
"""

import os
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.stability import ballooning_growth_rate, ballooning_lambda

# The finite-beta seed deck:
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "input.nfp2_QA_finite_beta"

# Flux surfaces the quasisymmetry residual is evaluated on:
SURFACES = np.linspace(0.1, 1.0, 6)

# Mode ladder: highest boundary mode number varied in each stage, and the
# residual evaluations each stage may spend:
MAX_MODES = [1, 2]
MAX_NFEV = [12, 20]

# Ballooning field lines. lambda is least stable at a configuration-dependent
# zeta0 (Gaur et al., J. Plasma Phys. 89 (2023), footnote 2): on this seed the
# single-point default misses 26 % of it, 3.27e-3 at zeta0 = 0 against 4.42e-3
# over the scan, so a zeta0 = 0 objective would optimize the wrong bound:
ZETA0S = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 5)
LINES = dict(npoints=97, nturns=3.0, zeta0s=ZETA0S)

# Softmax temperature of the bound, and the target and weight of its residual:
TEMPERATURE = 0.002
BALLOONING_TARGET = 0.0
BALLOONING_WEIGHT = 200.0

# Targets:
ASPECT_TARGET = 6.0  # Radial grid the optimizer trials are solved on, and the finer one the
# certificate is solved on. The difference is not small: a full run reaches
# max lambda 2.2e-4 on the stage grid and 9.1e-4 when the same boundary is
# re-solved at FINAL_NS, so quote the resolved number:
STAGE_NS = 25
STAGE_FTOL = 1.0e-11
STAGE_NITER = 4000
FINAL_NS = 71  # Step control. One scaled variable moves a low-order coefficient by
# PARAMETER_STEP metres, and a stage may move it MAX_PARAMETER_CHANGE steps:
PARAMETER_STEP = 0.02
MAX_PARAMETER_CHANGE = 4.0
ESS_ALPHA = 1.2                   # smaller values let high Fourier modes move more

# Equilibrium resolution: poloidal and toroidal mode numbers are max_mode + 2,
# but never below MINIMUM_MPOL:
MINIMUM_MPOL = 5

# Verification solve of the optimized boundary:
FINAL_FTOL = 1.0e-13
FINAL_NITER = 8000

# Every output file name contains this:
OUTPUT_NAME = "QA_ballooning_optimized"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODES, MAX_NFEV = [1], [3]
    STAGE_NS, FINAL_NS = 15, 15
    FINAL_FTOL = 1.0e-11

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

inp = replace(vj.VmecInput.from_file(INPUT_FILE),
              ns_array=np.array([STAGE_NS]), ftol_array=np.array([STAGE_FTOL]),
              niter_array=np.array([STAGE_NITER]))

### Set up the objective ######################################################

# The optimizable: a smooth upper bound on max lambda over all sampled lines.
ballooning = partial(ballooning_growth_rate, temperature=TEMPERATURE, **LINES)


def worst_lambda(equilibrium):
    """Hard max lambda over the sampled lines -- the number worth quoting."""
    growth = ballooning_lambda(equilibrium.solution, equilibrium.solver_context, **LINES)
    return float(np.max(np.asarray(growth)))


# Each term is (function, target, weight).
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
objective_function_terms = [
    (qs, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (ballooning, BALLOONING_TARGET, BALLOONING_WEIGHT),
]

report = opt.EquilibriumReporter(
    ("QS total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"),
    ("ballooning bound", ballooning, ".4e"))
monitor = opt.OptimizationMonitor()

### Run the optimization ######################################################

equilibrium = opt.solve_equilibrium(inp, verbose=not ci_smoke)
seed_lambda = worst_lambda(equilibrium)
seed_dmerc = float(np.min(np.asarray(equilibrium.wout.DMerc)[2:-1]))
print(f"\nseed: beta = {float(equilibrium.wout.betatotal):.3%}, "
      f"max lambda = {seed_lambda:+.4e} (unstable), "
      f"min DMerc = {seed_dmerc:+.3e} (Mercier-stable)")

for max_mode, max_nfev in zip(MAX_MODES, MAX_NFEV):
    print(f"\n===== QA + ballooning stage, max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    problem = opt.VmecProblem.from_tuples(
        inp, objective_function_terms, max_mode=max_mode, use_ess=True,
        ess_alpha=ESS_ALPHA, restart_from=equilibrium)
    monitor.problem = problem
    step = PARAMETER_STEP * problem.scales
    result = least_squares(
        problem.residual, problem.x0, jac=problem.residual_jac, x_scale=step,
        bounds=(problem.x0 - MAX_PARAMETER_CHANGE * step,
                problem.x0 + MAX_PARAMETER_CHANGE * step),
        max_nfev=max_nfev, ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor)
    inp = problem.input_from_x(result.x)
    equilibrium = problem.equilibrium_from_x(result.x)
    report(f"mode {max_mode}", equilibrium)
    print(f"max lambda = {worst_lambda(equilibrium):+.4e}")

### Check the result ##########################################################

# The physics certificate is the resolved solve, not the optimizer's grid:
# ballooning is radially stiff, and the optimizer's number is optimistic by a
# factor of four here.  Add a stage at FINAL_NS if you need the margin.
final_input = replace(inp, ns_array=np.array([FINAL_NS]),
                      ftol_array=np.array([FINAL_FTOL]),
                      niter_array=np.array([FINAL_NITER]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution, verbose=not ci_smoke,
    raise_on_max_iterations=True)
final_lambda = worst_lambda(final_equilibrium)
final_dmerc = float(np.min(np.asarray(final_equilibrium.wout.DMerc)[2:-1]))

### Print, plot and save ######################################################

report("final", final_equilibrium)
print(f"\nmax lambda {seed_lambda:+.4e} -> {final_lambda:+.4e} "
      f"({'stable' if final_lambda < 0.0 else 'still unstable'}) at NS = {FINAL_NS}\n"
      f"min DMerc {seed_dmerc:+.3e} -> {final_dmerc:+.3e}, "
      f"beta {float(final_equilibrium.wout.betatotal):.3%}")

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}\nWrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
