#!/usr/bin/env python
"""LASYM finite-beta quasi-helically symmetric boundary optimization.

The pressure is prescribed, p(s) = PRES_SCALE (1 - s), and the toroidal flux
sets beta: at zero net current beta depends on PRES_SCALE / PHIEDGE**2 alone,
so PHIEDGE = pi a**2 sqrt(mu0 PRES_SCALE / TARGET_BETA) with
a = R0 / ASPECT_TARGET.  The seed is scaled to the target aspect ratio and
one solve corrects the estimate, so the beta and aspect-ratio targets agree.
"""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.scaling import input_minor_radius

# Number of field periods, and the seed deck the boundary is shaped from:
NFP = 4
INPUT_FILE = Path(__file__).resolve().parents[2] / "data" / f"input.minimal_seed_nfp{NFP}"

SEED_PERTURBATION = 0.12
ASYMMETRY_PERTURBATION = 0.01

# Flux surfaces the quasisymmetry residual is evaluated on:
SURFACES = np.linspace(0.1, 0.9, 8)

# Mode ladder: highest boundary mode number varied in each stage, and the
# residual evaluations each stage may spend:
MAX_MODES = [1, 2]
MAX_NFEV = [10, 15]  # Targets:
TARGET_BETA = 0.01
ASPECT_TARGET = 6.0

# Mercier and resistive-interchange rows.  The magnetic axis is excluded and
# the weight rises smoothly at the edge, where stability is most difficult:
STABILITY_MIN_S = 0.2
STABILITY_WEIGHT = 1.0e-6
EDGE_WEIGHT_FACTOR = 10.0

# Step control. One scaled variable moves a low-order coefficient by
# PARAMETER_STEP metres, and a stage may move it MAX_PARAMETER_CHANGE steps:
PARAMETER_STEP = 0.01
MAX_PARAMETER_CHANGE = 3.0
ESS_ALPHA = 1.2                   # smaller values let high Fourier modes move more

# The deck's own radial ladder is used at the shipped settings:
STAGE_OVERRIDES = {}

# Forward-solve iteration cap inside an optimizer trial:
STAGE_MAX_ITERATIONS = 3000

# Equilibrium resolution: poloidal and toroidal mode numbers are max_mode + 2,
# but never below MINIMUM_MPOL:
MINIMUM_MPOL = 5

# Pressure p(s) = PRES_SCALE (1 - s), in pascal.  With TARGET_BETA it sets the
# field, B0 = sqrt(mu0 PRES_SCALE / TARGET_BETA); this value keeps PHIEDGE near
# the seed deck's, where the Mercier and resistive-interchange weights were
# tuned (DMerc scales as PHIEDGE**-2):
PRES_SCALE = 8.0e3

# Verification solve of the optimized boundary:
FINAL_NS = 71
FINAL_FTOL = 1e-14
FINAL_NITER = 20000

# Every output file name contains this; each stage also writes its own
# boundary as input.<STAGE_NAME>_max_mode_NNN:
OUTPUT_NAME = "QH_LASYM_finite_beta_optimized"
STAGE_NAME = "QH_finite_beta"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    SURFACES = np.array([0.25, 0.6, 0.9])
    MINIMUM_MPOL = 3
    MAX_MODES, MAX_NFEV = [1], [2]
    STAGE_OVERRIDES = dict(ns_array=np.array([11]), ftol_array=np.array([1e-8]),
                           niter_array=np.array([1500]))
    STAGE_MAX_ITERATIONS = 100
    PARAMETER_STEP = 0.001
    FINAL_NS, FINAL_FTOL = 31, 1e-10

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

# VmecInput is frozen, so copy its arrays before shaping the seed boundary.
inp = replace(vj.VmecInput.from_file(INPUT_FILE), **STAGE_OVERRIDES)
rbc, zbs, rbs, zbc = inp.rbc.copy(), inp.zbs.copy(), inp.rbs.copy(), inp.zbc.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
# RBS(1,1) and ZBC(1,1) open the asymmetric boundary families explicitly.
rbs[inp.ntor + 1, 1], zbc[inp.ntor + 1, 1] = ASYMMETRY_PERTURBATION, -ASYMMETRY_PERTURBATION
# Start at the target aspect ratio: scale the cross-section (every m >= 1
# harmonic) about the circular seed axis, which keeps the major radius.
minor_radius = float(inp.rbc[inp.ntor, 0]) / ASPECT_TARGET
scale = minor_radius / input_minor_radius(
    replace(inp, lasym=True, rbc=rbc, zbs=zbs, rbs=rbs, zbc=zbc))
for coefficients in (rbc, zbs, rbs, zbc):
    coefficients[:, 1:] *= scale

# Beta from the toroidal flux: the closed form, then one correction solve.
mu0 = 4.0e-7 * np.pi
closed_form_phiedge = np.pi * minor_radius**2 * np.sqrt(mu0 * PRES_SCALE / TARGET_BETA)
am = np.zeros(21)
am[:2] = [1.0, -1.0]  # p(s) = PRES_SCALE * (1 - s)
inp = replace(inp, lasym=True, rbc=rbc, zbs=zbs, rbs=rbs, zbc=zbc,
    pmass_type="power_series", am=am, pres_scale=PRES_SCALE,
    phiedge=closed_form_phiedge)
equilibrium = opt.solve_equilibrium(inp)
closed_form_beta = float(equilibrium.wout.betatotal)
inp = replace(inp, phiedge=closed_form_phiedge * np.sqrt(closed_form_beta / TARGET_BETA))
equilibrium = opt.solve_equilibrium(inp, initial_state=equilibrium.solution)
print(f"PHIEDGE: closed form {closed_form_phiedge:.6f} Wb gives beta "
      f"{closed_form_beta:.4%}; corrected {inp.phiedge:.6f} Wb gives beta "
      f"{float(equilibrium.wout.betatotal):.4%}")

### Set up the objective ######################################################

stability_s = np.linspace(0.0, 1.0, int(inp.ns_array[-1]))[2:-1]
# The singular magnetic-axis rows are excluded and the edge receives a
# smoothly increasing stability weight.
stability_weights = np.where(stability_s >= STABILITY_MIN_S,
    STABILITY_WEIGHT * (1.0 + (EDGE_WEIGHT_FACTOR - 1.0) * stability_s**4), 0.0)
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=-1)
# Each term is (function, target, weight).
objective_function_terms = [(qs, 0.0, 1.0), (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (opt.volume_average_beta, TARGET_BETA, 1.0 / TARGET_BETA**2),
    (opt.mercier_stability_residual, 0.0, stability_weights),
    (opt.glasser_stability_residual, 0.0, stability_weights)]
report = opt.EquilibriumReporter(
    ("QS", qs.total, ".4e"), ("beta", opt.volume_average_beta, ".3%"),
    ("aspect", opt.aspect_ratio, ".3f"), ("iota", opt.mean_iota, ".3f"))
monitor = opt.OptimizationMonitor()

### Run the optimization ######################################################

# A RuntimeWarning about uncertified Jacobian columns is expected once the
# optimizer leaves the seed and needs no action; see examples/README.md.
for max_mode, max_nfev in zip(MAX_MODES, MAX_NFEV):
    print(f"\n===== LASYM finite-beta QH stage, max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    problem = opt.VmecProblem.from_tuples(inp, objective_function_terms, max_mode=max_mode,
        use_ess=True, ess_alpha=ESS_ALPHA, restart_from=equilibrium,
        forward_max_iterations=STAGE_MAX_ITERATIONS,
        progress=True, evaluation_progress=True)
    print(f"dof_names = {problem.dof_names}")
    problem.compile_residual_and_jacobian()
    monitor.problem = problem
    step = PARAMETER_STEP * problem.scales
    result = least_squares(problem.residual, problem.x0, jac=problem.residual_jac,
        x_scale=step, bounds=(problem.x0 - MAX_PARAMETER_CHANGE * step,
                             problem.x0 + MAX_PARAMETER_CHANGE * step),
        max_nfev=max_nfev, ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor)
    inp, equilibrium = problem.input_from_x(result.x), problem.equilibrium_from_x(result.x)
    inp.to_indata(f"input.{STAGE_NAME}_max_mode_{max_mode:03d}")
    report(f"mode {max_mode}", equilibrium)

### Check the result ##########################################################

# The optimizer's grid is not the certificate: re-solve the optimized boundary
# on a finer radial grid to a tighter tolerance and quote that.
final_input = replace(inp, ns_array=np.array([FINAL_NS]),
    ftol_array=np.array([FINAL_FTOL]), niter_array=np.array([FINAL_NITER]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution, verbose=not ci_smoke,
    raise_on_max_iterations=True)
### Print, plot and save ######################################################

print(f"asymmetric boundary norm = "
      f"{np.linalg.norm(final_input.rbs) + np.linalg.norm(final_input.zbc):.6e}")
report("final", final_equilibrium)

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}\nWrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
