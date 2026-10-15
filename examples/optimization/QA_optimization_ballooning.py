#!/usr/bin/env python
"""Optimize a quasi-axisymmetric boundary to ideal-ballooning stability at fixed beta.

The seed (``input.nfp2_QA_finite_beta``, beta = 2.7 %, aspect 5.0) is
Mercier-stable and ballooning-unstable on its outer surfaces, s >= 0.8, which
is the case a ballooning objective is for: the interchange criteria see nothing
to fix.

The objective follows Gaur et al., Plasma Phys. Control. Fusion 67, 125015
(2025) and DESC's ``BallooningStability``: every sampled field line whose
growth rate exceeds the threshold contributes its excess as one residual row,
and a stable line contributes nothing, so the optimizer spends its steps on
the lines that are unstable. Pressure, current and toroidal flux are held;
beta, aspect ratio and Mercier stability are held by their own rows, so
ballooning cannot be bought by lowering beta or by changing the device.
"""

import os
from dataclasses import replace
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.stability import ballooning_lambda, ballooning_residual

# The finite-beta seed deck:
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "input.nfp2_QA_finite_beta"

# Flux surfaces the quasisymmetry residual is evaluated on:
SURFACES = np.linspace(0.1, 1.0, 6)

# Mode ladder: highest boundary mode number varied in each stage, and the
# residual evaluations each stage may spend:
MAX_MODES = [1, 2]
MAX_NFEV = [4, 6]
# Research setting (the README result; hours on 8 cores):
# MAX_MODES = [1, 2, 3]
# MAX_NFEV = [15, 25, 25]

# Ballooning field lines: surfaces by normalized flux s (the same surfaces on
# every radial grid), field-line labels over the symmetric half [0, pi], and
# the ballooning parameter zeta0, at which lambda peaks in a
# configuration-dependent place (Gaur et al. 2023, footnote 2). The line grid
# keeps 128 points per poloidal turn, enough to resolve the toroidal harmonics
# for the Richardson step; three turns hold an unstable mode, and the
# certificate below reads five:
BALLOONING_S = [0.7, 0.8, 0.85, 0.9, 0.95]
LINES = dict(surfaces=BALLOONING_S, alphas=np.linspace(0.0, np.pi, 8),
             zeta0s=np.linspace(-0.5 * np.pi, 0.5 * np.pi, 5),
             npoints=385, nturns=3.0)

# A line contributes max(lambda - BALLOONING_THRESHOLD, 0). Stable lines tend
# to lambda -> 0- as the line lengthens, so the threshold is zero, not a margin:
BALLOONING_THRESHOLD = 0.0
BALLOONING_WEIGHT = 30.0

# The certificate re-reads lambda on a denser set of lines and surfaces:
CHECK_LINES = dict(surfaces=np.linspace(0.5, 0.98, 17),
                   alphas=np.linspace(0.0, np.pi, 17),
                   zeta0s=np.linspace(-0.5 * np.pi, 0.5 * np.pi, 9),
                   npoints=641, nturns=5.0)

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
    LINES = dict(LINES, alphas=np.linspace(0.0, np.pi, 4), npoints=321)
    CHECK_LINES = dict(LINES, surfaces=[0.8, 0.9, 0.95])

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

inp = replace(vj.VmecInput.from_file(INPUT_FILE),
              ns_array=np.array([STAGE_NS]), ftol_array=np.array([STAGE_FTOL]),
              niter_array=np.array([STAGE_NITER]))

### Set up the objective ######################################################

PHIEDGE = float(inp.phiedge)
ballooning_rows = partial(ballooning_residual, threshold=BALLOONING_THRESHOLD, **LINES)


def worst_lambda(state, runtime):
    """Hard max lambda over the optimizer's lines, the number worth quoting."""
    return jnp.max(ballooning_lambda(state, runtime, **LINES))


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
    (ballooning_rows, 0.0, BALLOONING_WEIGHT),
]

report = opt.EquilibriumReporter(
    ("QS total", qs.total, ".4e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("beta", opt.volume_average_beta, ".3%"), ("mean iota", opt.mean_iota, ".4f"),
    ("max lambda", worst_lambda, "+.3e"), ("min PHIEDGE^2 DMerc", minimum_mercier, "+.3e"))
monitor = opt.OptimizationMonitor()

### Run the optimization ######################################################

equilibrium = opt.solve_equilibrium(inp, verbose=not ci_smoke)
seed_equilibrium = equilibrium
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
# the seed and the optimized boundary are both re-solved at FINAL_NS and read
# on the denser CHECK_LINES.
def resolved(equilibrium_input, initial):
    """Solve at FINAL_NS from a converged stage solution."""
    return opt.solve_equilibrium(
        replace(equilibrium_input, ns_array=np.array([FINAL_NS]),
                ftol_array=np.array([FINAL_FTOL]), niter_array=np.array([FINAL_NITER])),
        initial_state=initial.solution, verbose=not ci_smoke, raise_on_max_iterations=True)


final_input = replace(inp, ns_array=np.array([FINAL_NS]),
                      ftol_array=np.array([FINAL_FTOL]), niter_array=np.array([FINAL_NITER]))
final_equilibrium = resolved(inp, equilibrium)
seed_resolved = resolved(seed_equilibrium.inp, seed_equilibrium)
maps = {name: np.asarray(ballooning_lambda(eq.state, eq.runtime, **CHECK_LINES)).max(axis=2)
        for name, eq in (("seed", seed_resolved), ("final", final_equilibrium))}

### Print, plot and save ######################################################

final = report("final", final_equilibrium)
seed_check = report("seed", seed_resolved)
print(f"\nmax lambda {maps['seed'].max():+.4e} -> {maps['final'].max():+.4e} "
      f"({'stable' if maps['final'].max() < 0.0 else 'still unstable'}) at NS = {FINAL_NS} "
      f"on {np.size(maps['final']) * len(CHECK_LINES['zeta0s'])} lines\n"
      f"min PHIEDGE^2 DMerc {seed_check['min PHIEDGE^2 DMerc']:+.3e} -> "
      f"{final['min PHIEDGE^2 DMerc']:+.3e}, "
      f"beta {seed_check['beta']:.3%} -> {final['beta']:.3%}, "
      f"aspect {seed_check['aspect']:.3f} -> {final['aspect']:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True, layout="constrained")
limit = float(np.max(np.abs(maps["seed"])))
for ax, name in zip(axes, ("seed", "final")):
    image = ax.pcolormesh(CHECK_LINES["alphas"], CHECK_LINES["surfaces"], maps[name],
                          cmap="RdBu_r", vmin=-limit, vmax=limit, shading="nearest")
    ax.contour(CHECK_LINES["alphas"], CHECK_LINES["surfaces"], maps[name], [0.0], colors="k")
    ax.set(title=f"{name}: max lambda = {maps[name].max():+.2e}", xlabel="field line alpha")
axes[0].set_ylabel("normalized flux s")
fig.colorbar(image, ax=axes, label="max over zeta0 of lambda (> 0 unstable)")
fig.savefig(f"{OUTPUT_NAME}_lambda.png", dpi=150)
np.savez(f"{OUTPUT_NAME}_lambda.npz", **maps, **{k: np.asarray(v) for k, v in CHECK_LINES.items()})
print(f"Wrote {OUTPUT_NAME}_lambda.png and {OUTPUT_NAME}_lambda.npz")

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}\nWrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
