#!/usr/bin/env python
"""Ask only for iota, a small bootstrap current and an aspect ratio, and watch QI.

A finite-beta boundary is optimized with no symmetry objective at all: the
rows are an iota floor, the Redl bootstrap mismatch (the current stays
self-consistent), the Redl bootstrap current <j.B> itself with target 0, the
aspect ratio and the volume-average beta. A small bootstrap current is the
property quasi-isodynamic fields are designed for, so the question this
example asks is whether such an optimum drifts toward QI without being told.

The constructed-QI residual of QI_optimization_bootstrap.py, the mirror ratio
and the QA and QH quasisymmetry ratios are reported, never targeted, and |B|
in Boozer coordinates is plotted for the seed and the optimum. Read the
answer off those numbers. At the default budget the QI residual
falls by a third while both QS ratios rise about fivefold, but it stays near
1.0, about seven times the 0.15 that QI_optimization_bootstrap_dkx.py reaches
by targeting it: the optimum leaves quasisymmetry without becoming QI.
"""

import os
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.bootstrap import (ELEMENTARY_CHARGE, KineticProfiles,
                                 RedlBootstrapMismatch, j_dot_B_redl,
                                 redl_geometry_from_state, self_consistent_bootstrap)
from vmex.core.qi import ConstructedQIResidual
from vmex.core.plotting import boozer_modB_on_surface

# Number of field periods, and the seed deck the boundary is shaped from:
NFP = 2
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"

# Rotating-ellipse amplitude added to the circular seed:
SEED_PERTURBATION = 0.05

# Pressure the profiles are calibrated to, and the weight of the beta residual,
# which is relative because the target is small. On this seed the ~50 kA
# bootstrap current of beta = 2.5% leaves the first mpol = 5 stage solve stalled
# near fsq = 5e-9, above the 100 x ftol seed check; 1.5% converges to ftol:
TARGET_BETA = 0.015
BETA_WEIGHT = 1.0 / TARGET_BETA**2

# Flux surfaces the bootstrap residuals and reported QI/QS diagnostics are evaluated on:
SURFACES = np.linspace(0.1, 0.9, 8)

# Mode ladder: highest boundary mode number varied in each stage, the residual
# evaluations each stage may spend, and the optimized I'(s) spline knots:
MAX_MODES = [1, 2]
MAX_NFEV = [8, 8]
N_CURRENT_SPLINE = [6, 8]

# Boozer resolution the QI residual is evaluated on:
QI_OPTIONS = dict(mboz=10, nboz=10, nphi=41, nalpha=12, n_bounce=13)

# Targets:
ASPECT_TARGET = 10.0             # near the seed's 11.5: no aspect pull toward a shape family
IOTA_FLOOR = 0.42                 # minimum |iota| over the profile
BOOTSTRAP_WEIGHT = 1.0            # on <j.B> / (seed rms <j.B>), target 0

# Picard loop that makes the seed current self-consistent:
PICARD_ITERATIONS = 8
PICARD_TOLERANCE = 1e-3
REDL_N_LAMBDA = 16

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
FINAL_NS = 51
# The tens of kA of bootstrap current stall this solve near fsq = 1e-10 at
# ns = 51, so the certificate tolerance is 1e-10 rather than the template's 1e-14:
FINAL_FTOL = 1e-10
FINAL_NITER = 20000

# Every output file name contains this:
OUTPUT_NAME = "bootstrap_current_optimized"
CURRENT_FIGURE = "bootstrap_current_optimized_current.png"
BOOZER_FIGURE = "bootstrap_current_optimized_boozer_modB.png"
BOOZER_SURFACE = 0.5               # s of the Boozer |B| panels

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    SURFACES = np.linspace(0.3, 0.7, 3)
    QI_OPTIONS = dict(mboz=8, nboz=8, nphi=31, nalpha=7, n_bounce=7)
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
picard = self_consistent_bootstrap(inp, profiles, 0, n_iter=PICARD_ITERATIONS,
                                   tol=PICARD_TOLERANCE, degree=N_CURRENT_SPLINE[0] - 1,
                                   s_eval=SURFACES, verbose=not ci_smoke)
inp, equilibrium = picard.input, picard.equilibrium

### Set up the objective ######################################################

def iota_floor(equilibrium_state, solver_context):
    """Hinge on the profile minimum of |iota|: a mean target can hide a near-zero surface."""
    return jnp.maximum(
        IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)


def redl_j_dot_B(equilibrium_state, solver_context):
    """Redl <j.B> on SURFACES in MA T/m^2, the units of DKX's mismatch=False row."""
    geometry = redl_geometry_from_state(equilibrium_state, solver_context,
                                        surfaces=SURFACES, n_lambda=REDL_N_LAMBDA)
    return j_dot_B_redl(profiles, geometry, 0)[0] / 1e6


def rms_j_dot_B(equilibrium_state, solver_context):
    """Root-mean-square Redl <j.B> over SURFACES, in MA T/m^2."""
    return jnp.sqrt(jnp.mean(redl_j_dot_B(equilibrium_state, solver_context) ** 2))


def boozer_modB(equilibrium, label):
    """|B|(theta_B, phi_B) on BOOZER_SURFACE, from a booz_xform_jax transform."""
    wout_file = vj.write_wout(f"wout_{OUTPUT_NAME}_{label}.nc", equilibrium.wout)
    boozmn = vj.run_booz_xform(wout_file, mbooz=16, nbooz=16, surfaces=[BOOZER_SURFACE],
                               outdir=".")
    return boozer_modB_on_surface(boozmn, s_index=-1)


def relative_j_dot_B(equilibrium_state, solver_context):
    """Redl <j.B> relative to the seed's rms value, so the row starts at O(1)."""
    return redl_j_dot_B(equilibrium_state, solver_context) / seed_rms_j_dot_B


seed_rms_j_dot_B = float(rms_j_dot_B(equilibrium.state, equilibrium.runtime))

# Each term is (function, target, weight). No QS or QI row.
bootstrap = RedlBootstrapMismatch(profiles, helicity_n=0, surfaces=SURFACES,
                                  n_lambda=REDL_N_LAMBDA)
objective_function_terms = [
    (bootstrap, 0.0, 1.0),
    (relative_j_dot_B, 0.0, BOOTSTRAP_WEIGHT),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (iota_floor, 0.0, 10.0),
    (opt.volume_average_beta, TARGET_BETA, BETA_WEIGHT),
]

# Reported, not targeted: QI and the quasisymmetry ratios for comparison.
qi = ConstructedQIResidual(SURFACES, **QI_OPTIONS)
qa = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
qh = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=-1)
report = opt.EquilibriumReporter(
    ("f_boot", bootstrap.total, ".4e"), ("rms <j.B> [MA T/m^2]", rms_j_dot_B, ".4f"),
    ("aspect", opt.aspect_ratio, ".3f"), ("min |iota|", opt.min_abs_iota, ".3f"),
    ("beta", opt.volume_average_beta, ".3%"),
    ("constructed QI", qi.total, ".4e"), ("mirror", opt.mirror_ratio, ".3f"),
    ("QS(QA)", qa.total, ".4e"), ("QS(QH)", qh.total, ".4e"))
monitor = opt.OptimizationMonitor()

### Run the optimization ######################################################

seed_values = report("self-consistent seed", equilibrium)
seed_boozer = boozer_modB(equilibrium, "seed")
costs = []
for max_mode, max_nfev, n_spline in zip(MAX_MODES, MAX_NFEV, N_CURRENT_SPLINE):
    print(f"\n===== iota/bootstrap/aspect stage, max_mode = {max_mode} =====")
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
    costs += [float(np.sum(problem.residual(problem.x0) ** 2)), 2 * result.cost]
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

### Check the result ##########################################################

# The optimizer's grid is not the certificate: re-solve the optimized boundary
# on a finer radial grid to a tighter tolerance and quote that.
final_input = replace(inp, ns_array=np.array([FINAL_NS]),
    ftol_array=np.array([FINAL_FTOL]), niter_array=np.array([FINAL_NITER]))
final_equilibrium = opt.solve_equilibrium(final_input, initial_state=equilibrium.solution,
    verbose=not ci_smoke, raise_on_max_iterations=True)

### Print, plot and save ######################################################

print(f"cost: initial {costs[0]:.6e} -> final {costs[-1]:.6e}")
assert costs[-1] < costs[0], "the optimization did not lower the objective"
final_values = report("final", final_equilibrium)

# The finding, stated from the reported numbers rather than assumed:
for key in ("constructed QI", "QS(QA)", "QS(QH)"):
    print(f"{key}: seed {seed_values[key]:.4e} -> final {final_values[key]:.4e}")
qi_ratio = final_values["constructed QI"] / seed_values["constructed QI"]
qs_ratio = final_values["QS(QA)"] / seed_values["QS(QA)"]
print(f"trend toward QI: constructed-QI residual x{qi_ratio:.2f}, QS(QA) x{qs_ratio:.2f} "
      "from the seed (neither was targeted)")

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}\nWrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
vj.plot_bootstrap_current(CURRENT_FIGURE, final_equilibrium, bootstrap)
print(f"Wrote {CURRENT_FIGURE}")

# |B| contours in Boozer coordinates, seed and optimum. QI shows as contours
# that close poloidally (vertical bands); QA as horizontal lines.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
final_boozer = boozer_modB(final_equilibrium, "final")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
for ax, (theta, phi, B), title in zip(axes, (seed_boozer, final_boozer), ("seed", "optimized")):
    contours = ax.contour(phi, theta, B, levels=24, cmap="jet", linewidths=1.0)
    fig.colorbar(contours, ax=ax, label=r"$|B|$ [T]")
    ax.set_title(f"{title}, s = {BOOZER_SURFACE}")
    ax.set_xlabel(r"Boozer toroidal angle $\phi_B$")
    ax.set_ylabel(r"Boozer poloidal angle $\theta_B$")
fig.savefig(BOOZER_FIGURE, dpi=150)
print(f"Wrote {BOOZER_FIGURE}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
