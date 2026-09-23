#!/usr/bin/env python
"""Optimize a boundary for quasi-axisymmetry at finite beta.

This is the least-squares counterpart to
``QA_optimization_finite_beta_scalar.py``: SciPy least squares receives the
full residual vector and its exact Jacobian, while VMEX supplies the implicit
equilibrium derivatives.

The pressure is prescribed, p(s) = PRES_SCALE (1 - s), and beta is set by the
toroidal flux instead of by rescaling the pressure.  At zero net current ideal
MHD is unchanged by p -> lambda**2 p, PHIEDGE -> lambda PHIEDGE (the field
scales by lambda, the geometry does not), so beta depends on
PRES_SCALE / PHIEDGE**2 alone.  With B0 = PHIEDGE / (pi a**2),
<p> = PRES_SCALE <1 - s> = PRES_SCALE / 2 and beta = 2 mu0 <p> / <B**2>,

    PHIEDGE = pi a**2 sqrt(mu0 PRES_SCALE / TARGET_BETA),   a = R0 / ASPECT_TARGET.

The seed is scaled to the target aspect ratio, one solve measures the shaping
and finite-beta error of that estimate (a few per cent), and one correction
PHIEDGE *= sqrt(beta / TARGET_BETA) removes it. Beta and the aspect ratio then
agree at the targets, so the beta row only absorbs what reshaping does to
<B**2> at fixed aspect ratio (10 to 20 per cent without the row) and no
per-stage pressure recalibration is needed. Holding PHIEDGE while calibrating
PRES_SCALE on the seed instead makes the two rows disagree: at fixed flux beta
scales as ASPECT**-4, so the targets are consistent only at the seed's aspect
ratio.
"""

import os
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.scaling import input_minor_radius

# Number of field periods, and the seed deck the boundary is shaped from:
NFP = 2
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"

# Rotating-ellipse amplitude added to the circular seed:
SEED_PERTURBATION = 0.05

# Volume-averaged beta that the toroidal flux is chosen to give at ASPECT_TARGET:
TARGET_BETA = 0.01
# Pressure p(s) = PRES_SCALE (1 - s), in pascal.  With TARGET_BETA it sets the
# field, B0 = sqrt(mu0 PRES_SCALE / TARGET_BETA); this value keeps PHIEDGE near
# the seed deck's, where the Mercier and resistive-interchange weights were
# tuned (DMerc scales as PHIEDGE**-2):
PRES_SCALE = 1.0e3

# Flux surfaces the quasisymmetry residual is evaluated on:
SURFACES = np.linspace(0.1, 0.9, 8)

# Mode ladder: highest boundary mode number varied in each stage, and the
# residual evaluations each stage may spend:
MAX_MODES = [1, 2]
MAX_NFEV = [10, 15]  # Targets:
ASPECT_TARGET = 6.0
MAGNETIC_WELL_TARGET = 0.01
IOTA_FLOOR = 0.42                 # minimum |iota| over the profile

# Mercier and resistive-interchange rows: weight on surfaces s >= STABILITY_MIN_S,
# rising to EDGE_WEIGHT_FACTOR times that at the edge:
STABILITY_MIN_S = 0.2
STABILITY_WEIGHT = 1e-6
EDGE_WEIGHT_FACTOR = 10.0

# Step control. One scaled variable moves a low-order coefficient by
# PARAMETER_STEP metres, and a stage may move it MAX_PARAMETER_CHANGE steps:
PARAMETER_STEP = 0.01
MAX_PARAMETER_CHANGE = 3.0
ESS_ALPHA = 1.2                   # smaller values let high Fourier modes move more
VARY_MAJOR_RADIUS = False         # True optimizes RBC(0,0) instead of fixing it

# Equilibrium resolution: poloidal and toroidal mode numbers are max_mode + 2,
# but never below MINIMUM_MPOL:
MINIMUM_MPOL = 5

# Verification solve of the optimized boundary:
FINAL_NS = 71
FINAL_FTOL = 1e-14
FINAL_NITER = 20000
POLISH_FORCE_BALANCE = False      # True polishes only the final saved state

# Every output file name contains this:
OUTPUT_NAME = "QA_finite_beta_optimized"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    SURFACES = np.array([0.25, 0.6, 0.9])
    MAX_MODES, MAX_NFEV, MINIMUM_MPOL = [1], [2], 3
    FINAL_NS, FINAL_FTOL = 31, 1e-10

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

# Shape the seed, then scale its cross-section (every m >= 1 harmonic) about
# the circular axis so it starts at the target aspect ratio.
inp = vj.VmecInput.from_file(INPUT_FILE)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
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

### Set up the objective ######################################################

def iota_floor(state, runtime):
    """Hinge on the profile minimum of |iota|."""
    return jnp.maximum(IOTA_FLOOR - opt.min_abs_iota(state, runtime), 0.0)


stability_s = np.linspace(0.0, 1.0, int(inp.ns_array[-1]))[2:-1]
stability_weights = np.where(
    stability_s >= STABILITY_MIN_S,
    STABILITY_WEIGHT * (1.0 + (EDGE_WEIGHT_FACTOR - 1.0) * stability_s**4), 0.0)

# Each term is (function, target, weight).
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
objective_function_terms = [
    (qs, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (iota_floor, 0.0, 10.0),
    (opt.magnetic_well, MAGNETIC_WELL_TARGET, 1.0),
    (opt.volume_average_beta, TARGET_BETA, 1.0 / TARGET_BETA**2),
    (opt.mercier_stability_residual, 0.0, stability_weights),
    (opt.glasser_stability_residual, 0.0, stability_weights),
]

report = opt.EquilibriumReporter(
    ("QS total", qs.total, ".6e"), ("beta", opt.volume_average_beta, ".3%"),
    ("aspect", opt.aspect_ratio, ".4f"), ("min |iota|", opt.min_abs_iota, ".4f"),
    ("magnetic well", opt.magnetic_well, ".4f"))
monitor = opt.OptimizationMonitor()

### Run the optimization ######################################################

# Stages that share a boundary resolution share one problem, as in
# QA_optimization.py: problem.subproblem() frees the harmonics of each stage
# and the jitted residual and Jacobian are compiled once per resolution.
resolution_of = {max_mode: max(max_mode + 2, MINIMUM_MPOL) for max_mode in MAX_MODES}
group_max_mode = {
    max_mode: max(other for other in MAX_MODES
                  if resolution_of[other] == resolution_of[max_mode])
    for max_mode in MAX_MODES}
problem, mpol = None, None
for max_mode, max_nfev in zip(MAX_MODES, MAX_NFEV):
    print(f"\n===== finite-beta QA stage, max_mode = {max_mode} =====")
    if resolution_of[max_mode] != mpol:
        mpol = resolution_of[max_mode]
        inp = replace(inp, delt=0.5).change_resolution(
            mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
        problem = opt.VmecProblem.from_tuples(
            inp, objective_function_terms, max_mode=group_max_mode[max_mode],
            vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True,
            ess_alpha=ESS_ALPHA, restart_from=equilibrium)
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
final_input = replace(
    inp, ns_array=np.array([FINAL_NS]), ftol_array=np.array([FINAL_FTOL]),
    niter_array=np.array([FINAL_NITER]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution,
    verbose=not ci_smoke, raise_on_max_iterations=True,
    polish_force_balance=POLISH_FORCE_BALANCE)

### Print, plot and save ######################################################

final_total = report("final", final_equilibrium)["QS total"]
print(f"\nQS total {final_total:.3e}")

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}\nWrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
