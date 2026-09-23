#!/usr/bin/env python
"""Optimize a boundary for quasi-helical symmetry at finite beta with one adjoint per gradient.

This minimizes the same weighted sum of squared residuals as the least-squares
example of the same family. The difference is algorithmic: the residual rows
are summed into one scalar before implicit differentiation, so SciPy L-BFGS-B
receives a value and a gradient from one reverse equilibrium adjoint instead of
a residual vector and its full Jacobian.

The scalar form trades objective progress per evaluation for a cheaper cold
start and lower peak memory: at a matched evaluation budget the least-squares
driver reached roughly a 3x lower objective on the same problem, so it remains
the default for objective progress.

The pressure is prescribed, p(s) = PRES_SCALE (1 - s), and the toroidal flux
sets beta.  At zero net current ideal MHD is unchanged by p -> lambda**2 p,
PHIEDGE -> lambda PHIEDGE (the field scales, the geometry does not), so beta
depends on PRES_SCALE / PHIEDGE**2 alone, and B0 = PHIEDGE / (pi a**2) with
<p> = PRES_SCALE / 2 gives

    PHIEDGE = pi a**2 sqrt(mu0 PRES_SCALE / TARGET_BETA),   a = R0 / ASPECT_TARGET.

The seed is scaled to the target aspect ratio and one solve corrects the
estimate. Beta and the aspect ratio then agree at the targets, so the beta row
only absorbs what reshaping does to <B**2> at fixed aspect ratio (10 to 20 per
cent without the row) and no per-stage pressure recalibration is needed.
Holding PHIEDGE while calibrating PRES_SCALE on the seed instead makes the two
rows disagree: at fixed flux beta scales as ASPECT**-4, so the targets are
consistent only at the seed's aspect ratio.
"""

import os
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

import vmex as vj
from vmex import optimize as opt
from vmex.core.scaling import input_minor_radius

NFP = 4
TARGET_BETA = 0.01
# Pressure p(s) = PRES_SCALE (1 - s), in pascal.  With TARGET_BETA it sets the
# field, B0 = sqrt(mu0 PRES_SCALE / TARGET_BETA); this value keeps PHIEDGE near
# the seed deck's, where the Mercier and resistive-interchange weights were
# tuned (DMerc scales as PHIEDGE**-2):
PRES_SCALE = 8.0e3
SURFACES = np.linspace(0.1, 0.9, 8)
MAX_MODES = [1, 2]
MAXITER = [15, 25]
ASPECT_TARGET = 6.0
IOTA_TARGET = -1.1
STABILITY_MIN_S = 0.2
STABILITY_WEIGHT = 1e-6
EDGE_WEIGHT_FACTOR = 10.0
PARAMETER_STEP = 0.01
MAX_PARAMETER_CHANGE = 3.0
ESS_ALPHA = 1.2                   # smaller values let high Fourier modes move more
MINIMUM_MPOL = 5
VARY_MAJOR_RADIUS = False         # True optimizes RBC(0,0) instead of fixing it
SEED_PERTURBATION = 0.12
POLISH_FORCE_BALANCE = False      # True polishes only the final saved state

# Verification solve of the optimized boundary:
FINAL_NS = 71
FINAL_FTOL = 1e-14
FINAL_NITER = 20000

# Every output file name contains this:
OUTPUT_NAME = "QH_finite_beta_scalar_optimized"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    SURFACES = np.array([0.25, 0.6, 0.9])
    MAX_MODES, MAXITER, MINIMUM_MPOL = [1], [2], 3
    FINAL_NS, FINAL_FTOL = 31, 1e-10

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"
inp = vj.VmecInput.from_file(INPUT_FILE)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
# Start at the target aspect ratio: scale the cross-section (every m >= 1
# harmonic) about the circular seed axis, which keeps the major radius.
minor_radius = float(inp.rbc[inp.ntor, 0]) / ASPECT_TARGET
scale = minor_radius / input_minor_radius(replace(inp, rbc=rbc, zbs=zbs))
rbc[:, 1:] *= scale
zbs[:, 1:] *= scale

# Beta from the toroidal flux: the closed form, then one correction solve.
mu0 = 4.0e-7 * np.pi
closed_form_phiedge = np.pi * minor_radius**2 * np.sqrt(mu0 * PRES_SCALE / TARGET_BETA)
am = np.zeros(21)
am[:2] = [1.0, -1.0]  # p(s) = PRES_SCALE * (1 - s)
inp = replace(inp, rbc=rbc, zbs=zbs, pmass_type="power_series", am=am,
              pres_scale=PRES_SCALE, phiedge=closed_form_phiedge)
equilibrium = opt.solve_equilibrium(inp)
closed_form_beta = float(equilibrium.wout.betatotal)
inp = replace(inp, phiedge=closed_form_phiedge * np.sqrt(closed_form_beta / TARGET_BETA))
equilibrium = opt.solve_equilibrium(inp, initial_state=equilibrium.solution)
print(f"PHIEDGE: closed form {closed_form_phiedge:.6f} Wb gives beta "
      f"{closed_form_beta:.4%}; corrected {inp.phiedge:.6f} Wb gives beta "
      f"{float(equilibrium.wout.betatotal):.4%}")

stability_s = np.linspace(0.0, 1.0, int(inp.ns_array[-1]))[2:-1]
stability_weights = np.where(
    stability_s >= STABILITY_MIN_S,
    STABILITY_WEIGHT * (1.0 + (EDGE_WEIGHT_FACTOR - 1.0) * stability_s**4), 0.0)
### Set up the objective ######################################################


qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=-1)
objective_terms = [
    (qs, 0.0, 1.0), (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (opt.mean_iota, IOTA_TARGET, 1.0),
    (opt.volume_average_beta, TARGET_BETA, 1.0 / TARGET_BETA**2),
    (opt.mercier_stability_residual, 0.0, stability_weights),
    (opt.glasser_stability_residual, 0.0, stability_weights)]


report = opt.EquilibriumReporter(
    ("QS", qs.total, ".4e"), ("beta", opt.volume_average_beta, ".3%"),
    ("aspect", opt.aspect_ratio, ".3f"),
    ("mean iota", opt.mean_iota, ".3f"))
monitor = opt.OptimizationMonitor()

### Run the optimization ######################################################

for max_mode, maxiter in zip(MAX_MODES, MAXITER):
    print(f"\n===== scalar finite-beta QH stage, max_mode = {max_mode} =====")
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
