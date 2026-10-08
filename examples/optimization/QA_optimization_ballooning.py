#!/usr/bin/env python
"""Optimize a quasi-axisymmetric boundary to ideal-ballooning stability at fixed beta.

The seed (``input.nfp2_QA_finite_beta``, beta = 2.7 %, aspect 5.0) is
Mercier-stable and ballooning-unstable on its outer surfaces, s >= 0.85, which
is the case a ballooning objective is for: the interchange criteria see nothing
to fix.

The objective follows DESC's ``BallooningStability`` (Gaur et al., J. Plasma
Phys. 89 (2023), doi:10.1017/S0022377823000107): every sampled field line whose growth rate
exceeds a threshold contributes its excess, and a stable line contributes
nothing, so the optimizer spends its steps on the lines that are unstable
instead of on the mean. Pressure, current and toroidal flux are held as in
DESC's example; beta, aspect ratio and Mercier stability are held by their own
rows, so ballooning cannot be bought by lowering beta or by changing the device.
"""

import os
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.stability import ballooning_lambda

# The finite-beta seed deck:
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "input.nfp2_QA_finite_beta"

# Flux surfaces the quasisymmetry residual is evaluated on:
SURFACES = np.linspace(0.1, 1.0, 6)

# Mode ladder: highest boundary mode number varied in each stage, and the
# residual evaluations each stage may spend:
MAX_MODES = [1, 2]
MAX_NFEV = [8, 12]

# Ballooning field lines and surfaces. lambda is least stable at a
# configuration-dependent zeta0 (Gaur et al. 2023, footnote 2), so zeta0 is
# scanned as in DESC. The surfaces span the stable core and the unstable edge,
# and the certificate reads the same normalized radii on the finer grid:
ZETA0S = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 5)
LINES = dict(npoints=97, nturns=3.0, zeta0s=ZETA0S)
BALLOONING_S = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]

# A line contributes max(lambda - BALLOONING_THRESHOLD, 0). The threshold sits
# below zero because lambda on the optimizer's grid under-reads the resolved one
# (4.0e-3 at NS = 25, 6.2e-3 at NS = 41 and 7.0e-3 at NS = 71 on the seed):
BALLOONING_THRESHOLD = -5.0e-4
BALLOONING_WEIGHT = 30.0

# Held quantities. The aspect ratio and beta are the seed's own; a mismatched
# aspect target moves the minor radius and, at fixed flux and pressure, beta
# with it. Mercier rows act on the dimensionless PHIEDGE**2 DMerc over
# s >= MERCIER_MIN_S (nearer the axis the finite-difference DMerc is a
# cancellation of much larger terms):
ASPECT_TARGET = 5.0
BETA_TARGET = 0.027
BETA_WEIGHT = 1.0 / BETA_TARGET
MERCIER_MARGIN = 5.0e-3
MERCIER_WEIGHT = 10.0
MERCIER_MIN_S = 0.1

# Radial grid the optimizer trials are solved on, and the finer one the
# certificate is solved on:
STAGE_NS = 41
STAGE_FTOL = 1.0e-11
STAGE_NITER = 4000
FINAL_NS = 71

# Step control. One scaled variable moves a low-order coefficient by
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
    STAGE_NS, FINAL_NS = 21, 21
    FINAL_FTOL = 1.0e-11

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

inp = replace(vj.VmecInput.from_file(INPUT_FILE),
              ns_array=np.array([STAGE_NS]), ftol_array=np.array([STAGE_FTOL]),
              niter_array=np.array([STAGE_NITER]))

### Set up the objective ######################################################

PHIEDGE = float(inp.phiedge)


def surface_indices(ns):
    """Full-mesh indices of BALLOONING_S on an ns-surface grid."""
    return [min(max(int(round(s * (ns - 1))), 2), ns - 2) for s in BALLOONING_S]


def lambdas(state, runtime):
    """Growth rate of every sampled line, shaped (surface, alpha, zeta0)."""
    ns = int(runtime.setup.s_full.shape[0])
    return ballooning_lambda(state, runtime, s_indices=surface_indices(ns), **LINES)


def ballooning_excess(state, runtime):
    """One row per line: its growth rate above the threshold, zero when below."""
    return jnp.maximum(lambdas(state, runtime) - BALLOONING_THRESHOLD, 0.0).ravel()


def worst_lambda(state, runtime):
    """Hard max lambda over the sampled lines, the number worth quoting."""
    return jnp.max(lambdas(state, runtime))


def mercier_window(state, runtime):
    """PHIEDGE**2 DMerc on the interior surfaces with s >= MERCIER_MIN_S."""
    dmerc = PHIEDGE**2 * opt.d_merc_state(state, runtime)
    s = np.linspace(0.0, 1.0, dmerc.shape[0])
    return dmerc[(s >= MERCIER_MIN_S) & (s < 1.0)]


def mercier_rows(state, runtime):
    """Hinge on the Mercier criterion below its margin (positive is stable)."""
    return jnp.maximum(MERCIER_MARGIN - mercier_window(state, runtime), 0.0)


def minimum_mercier(state, runtime):
    """Least stable PHIEDGE**2 DMerc in the window, reported not targeted."""
    return jnp.min(mercier_window(state, runtime))


# Each term is (function, target, weight).
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
objective_function_terms = [
    (qs, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (opt.volume_average_beta, BETA_TARGET, BETA_WEIGHT),
    (mercier_rows, 0.0, MERCIER_WEIGHT),
    (ballooning_excess, 0.0, BALLOONING_WEIGHT),
]

report = opt.EquilibriumReporter(
    ("QS total", qs.total, ".4e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("beta", opt.volume_average_beta, ".3%"), ("mean iota", opt.mean_iota, ".4f"),
    ("max lambda", worst_lambda, "+.3e"), ("min PHIEDGE^2 DMerc", minimum_mercier, "+.3e"))
monitor = opt.OptimizationMonitor()

### Run the optimization ######################################################

equilibrium = opt.solve_equilibrium(inp, verbose=not ci_smoke)
seed = report("seed", equilibrium)

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

### Check the result ##########################################################

# The physics certificate is the resolved solve, not the optimizer's grid:
# ballooning is radially stiff, so the same normalized radii are re-read at
# FINAL_NS.
final_input = replace(inp, ns_array=np.array([FINAL_NS]),
                      ftol_array=np.array([FINAL_FTOL]),
                      niter_array=np.array([FINAL_NITER]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution, verbose=not ci_smoke,
    raise_on_max_iterations=True)

### Print, plot and save ######################################################

final = report("final", final_equilibrium)
seed_lambda, final_lambda = seed["max lambda"], final["max lambda"]
print(f"\nmax lambda {seed_lambda:+.4e} -> {final_lambda:+.4e} "
      f"({'stable' if final_lambda < 0.0 else 'still unstable'}) at NS = {FINAL_NS}\n"
      f"min PHIEDGE^2 DMerc {seed['min PHIEDGE^2 DMerc']:+.3e} -> "
      f"{final['min PHIEDGE^2 DMerc']:+.3e}, "
      f"beta {seed['beta']:.3%} -> {final['beta']:.3%}")

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}\nWrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
