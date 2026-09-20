#!/usr/bin/env python
"""Optimize a finite-beta QH boundary against a self-consistent bootstrap current.

The current profile is not prescribed and then forgotten: a Picard loop
alternates hot-restarted VMEC solves with Redl bootstrap fits until the two
agree, and the optimization that follows varies a stage-refined current spline
alongside the boundary, with the Redl mismatch as a residual row. Mercier and
resistive-interchange rows carry a radially graded weight.

The quasi-helical class shifts the Redl isomorphism: both the Picard loop and
the mismatch residual take helicity -1 here, where the QA script takes 0.

The kinetic profiles are the Landreman-Buller-Drevlak forms, ne = n0 (1 - s^5)
and Te = Ti = T0 (1 - s); one seed solve calibrates their amplitude to the
requested volume-average beta for this field scale.
"""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.bootstrap import (ELEMENTARY_CHARGE, KineticProfiles,
                                 RedlBootstrapMismatch, self_consistent_bootstrap)

# Number of field periods, and the seed deck the boundary is shaped from:
NFP = 4
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"

# Rotating-ellipse amplitude added to the circular seed, which gives the
# optimizer a QH basin:
SEED_PERTURBATION = 0.12

# Pressure the profiles are calibrated to, and the weight of the beta residual,
# which is relative because the target is small:
TARGET_BETA = 0.025
BETA_WEIGHT = 1.0 / TARGET_BETA**2

# Flux surfaces the quasisymmetry and bootstrap residuals are evaluated on:
SURFACES = np.linspace(0.1, 0.9, 8)

# Mode ladder: highest boundary mode number varied in each stage, the residual
# evaluations each stage may spend, and the optimized I'(s) spline knots:
MAX_MODES = [2, 3]
MAX_NFEV = [15, 30]
N_CURRENT_SPLINE = [6, 8]

# Targets:
ASPECT_TARGET = 6.0

# Stability rows. VMEC's dimensional DMerc/DR values are O(1e2-1e3) for this
# seed, so the weight is small; Mercier coordinates are unreliable near the
# axis, so the weight is zero below STABILITY_MIN_S and rises toward the edge:
STABILITY_WEIGHT = 1.0e-6
EDGE_WEIGHT_FACTOR = 10.0
STABILITY_MIN_S = 0.2

# Redl isomorphism helicity of this symmetry class:
REDL_HELICITY = -1

# Picard loop that makes the seed current self-consistent:
PICARD_ITERATIONS = 8
PICARD_TOLERANCE = 1e-3
REDL_N_LAMBDA = 32

# Step control. Boundary coefficients move in metres; the current dofs are
# dimensionless, so they carry their own optimizer scale:
PARAMETER_STEP = 0.02
CURRENT_PARAMETER_STEP = 0.05
MAX_PARAMETER_CHANGE = 10.0       # per-stage box guardrail, in scaled step units
VARY_MAJOR_RADIUS = False         # True optimizes RBC(0,0) instead of fixing it

# Equilibrium resolution: poloidal and toroidal mode numbers are max_mode + 2,
# but never below MINIMUM_MPOL:
MINIMUM_MPOL = 5

# Verification solve of the optimized boundary:
FINAL_NS = 101
FINAL_FTOL = 1e-14
FINAL_NITER = 8000

# Every output file name contains this:
OUTPUT_NAME = "QH_bootstrap_optimized"
CURRENT_FIGURE = "QH_bootstrap_current.png"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    SURFACES = np.linspace(0.2, 0.8, 4)
    MAX_MODES, MAX_NFEV, N_CURRENT_SPLINE = [1], [4], [4]
    PICARD_ITERATIONS, REDL_N_LAMBDA = 2, 12
    FINAL_NS, FINAL_FTOL = 31, 1e-10

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

# VmecInput is frozen, so copy its arrays before shaping the seed boundary.
inp = vj.VmecInput.from_file(INPUT_FILE)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION

# The Landreman-Buller-Drevlak profiles. Their product gives p = 2 e ne Te;
# AM below is (1 - s)(1 - s^5), matching that shape.
n0 = 3.0e20 * (TARGET_BETA / 0.05) ** (1 / 3)
T0 = 15.0e3 * (TARGET_BETA / 0.05) ** (2 / 3)
am = np.zeros(21)
am[[0, 1, 5, 6]] = [1.0, -1.0, -1.0, 1.0]
ac = np.zeros(21)
ac[0] = 1.0
inp = replace(inp, rbc=rbc, zbs=zbs, delt=0.5, pmass_type="power_series", am=am,
              pres_scale=2 * ELEMENTARY_CHARGE * n0 * T0, ncurr=1,
              pcurr_type="power_series", ac=ac, curtor=0.0)

# One seed solve calibrates the profile amplitude to the requested beta.
seed = opt.solve_equilibrium(inp)
profile_scale = TARGET_BETA / float(seed.wout.betatotal)
n0 *= profile_scale ** (1 / 3)
T0 *= profile_scale ** (2 / 3)
inp = replace(inp, pres_scale=inp.pres_scale * profile_scale)

# These polynomials provide ne(s), Te(s) and Ti(s) to the Redl model. The
# Picard loop leaves the prescribed pressure and boundary shape unchanged and
# updates the current profile (I'(s), CURTOR) to the bootstrap response.
profiles = KineticProfiles(n0 * np.array([1, 0, 0, 0, 0, -1]),
                           T0 * np.array([1, -1]), T0 * np.array([1, -1]))
picard = self_consistent_bootstrap(inp, profiles, REDL_HELICITY,
                                   n_iter=PICARD_ITERATIONS,
                                   tol=PICARD_TOLERANCE, degree=N_CURRENT_SPLINE[0] - 1,
                                   s_eval=SURFACES, verbose=not ci_smoke)
inp, equilibrium = picard.input, picard.equilibrium

### Set up the objective ######################################################

def minimum_dmerc(equilibrium_state, solver_context):
    """Interior minimum of the Mercier criterion, reported not targeted."""
    return opt.d_merc_state(equilibrium_state, solver_context)[2:-1].min()


def maximum_dr(equilibrium_state, solver_context):
    """Interior maximum of the resistive-interchange criterion, reported not targeted."""
    return opt.glasser_d_r_state(equilibrium_state, solver_context)[2:-1].max()


# Zero weight below STABILITY_MIN_S, rising smoothly toward the edge.
stability_s = np.linspace(0.0, 1.0, int(inp.ns_array[-1]))[2:-1]
stability_weights = np.where(stability_s >= STABILITY_MIN_S,
    STABILITY_WEIGHT * (1.0 + (EDGE_WEIGHT_FACTOR - 1.0) * stability_s**4), 0.0)

# Each term is (function, target, weight).
bootstrap = RedlBootstrapMismatch(profiles, helicity_n=REDL_HELICITY, surfaces=SURFACES,
                                  n_lambda=REDL_N_LAMBDA)
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=-1)
objective_function_terms = [
    (qs, 0.0, 1.0), (bootstrap, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (opt.volume_average_beta, TARGET_BETA, BETA_WEIGHT),
    (opt.mercier_stability_residual, 0.0, stability_weights),
    (opt.glasser_stability_residual, 0.0, stability_weights),
]

report = opt.EquilibriumReporter(
    ("QS", qs.total, ".4e"), ("f_boot", bootstrap.total, ".4e"),
    ("beta", opt.volume_average_beta, ".3%"), ("aspect", opt.aspect_ratio, ".3f"),
    ("iota", opt.mean_iota, ".3f"), ("min DMerc", minimum_dmerc, ".2e"),
    ("max DR", maximum_dr, ".2e"))
monitor = opt.OptimizationMonitor()

### Run the optimization ######################################################

report("self-consistent seed", equilibrium)
for max_mode, max_nfev, n_spline in zip(MAX_MODES, MAX_NFEV, N_CURRENT_SPLINE):
    print(f"\n===== QH bootstrap stage, max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = inp.change_resolution(mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6,
                                nzeta=2 * mpol + 4)
    inp = opt.resample_current_profile(inp, n_spline)
    # A RuntimeWarning about uncertified Jacobian columns is expected once the
    # optimizer leaves the seed and needs no action; see examples/README.md.
    problem = opt.VmecProblem.from_tuples(
        inp, objective_function_terms, max_mode=max_mode,
        current_dofs=n_spline - 1, vary_major_radius=VARY_MAJOR_RADIUS,
        use_ess=True, restart_from=equilibrium, progress=not ci_smoke)
    print(f"dof_names = {problem.dof_names}")
    monitor.problem = problem
    step = PARAMETER_STEP * problem.scales
    step[-n_spline:] = CURRENT_PARAMETER_STEP  # n-1 spline shapes plus CURTOR
    result = least_squares(
        problem.residual, problem.x0, jac=problem.residual_jac, x_scale=step,
        bounds=(problem.x0 - MAX_PARAMETER_CHANGE * step,
                problem.x0 + MAX_PARAMETER_CHANGE * step),
        max_nfev=max_nfev, ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor)
    inp = problem.input_from_x(result.x)
    equilibrium = problem.equilibrium_from_x(result.x)
    report(f"mode {max_mode}", equilibrium)

### Check the result ##########################################################

# The optimizer's grid is not the certificate: re-solve the optimized boundary
# on a finer radial grid to a tighter tolerance and quote that.
final_input = replace(inp, ns_array=np.array([FINAL_NS]),
    ftol_array=np.array([FINAL_FTOL]), niter_array=np.array([FINAL_NITER]))
final_equilibrium = opt.solve_equilibrium(final_input, initial_state=equilibrium.solution,
    verbose=not ci_smoke, raise_on_max_iterations=True)

### Print, plot and save ######################################################

report("final", final_equilibrium)

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}\nWrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
vj.plot_bootstrap_current(CURRENT_FIGURE, final_equilibrium, bootstrap)
print(f"Wrote {CURRENT_FIGURE}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
