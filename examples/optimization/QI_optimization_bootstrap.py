#!/usr/bin/env python
"""Continue a cold-started vacuum QI boundary up a beta ladder, with bootstrap-current, well, Mercier and Gamma_c options.

Run QI_optimization.py first: its result, input.QI_optimized, is the starting
boundary (its NFP and SEED_MODES choose the field-period count and the
starting basin). Each stage of STAGES then raises the volume-average beta,
recalibrates the pressure and continues from the previous stage's equilibrium,
so the QI shape found in vacuum is carried to the target beta in steps. The
aspect ratio, minimum |iota| and mirror ratio of the vacuum design are held.
Four options shape the physics, and any of them works alone:

- BOOTSTRAP = "none": the plasma is current-free. "redl" varies a
  stage-refined current spline against the Redl bootstrap current: a Picard
  loop alternates hot-restarted VMEC solves with Redl fits until the two agree
  at every new beta, and the Redl mismatch is a residual row. "dkx" swaps in
  DKX's drift-kinetic current for the last DKX_FINAL_STAGES stages (pip install
  "dkx>=2.14.0"): the row is the absolute difference between the equilibrium's
  <j.B> and DKX's, in MA T/m^2, because the self-normalized form cannot ask for
  less current when DKX's is near zero, as it is for a QI field.
- MAGNETIC_WELL_FLOOR: a hinge holding the well proxy (V'(0) - V'(1)) / V'(0)
  at or above the floor, zero while it holds; None drops it.
- MERCIER: radially graded Mercier and resistive-interchange rows; False (the
  default) drops them. With them on, NFP 3 approaches marginal Mercier
  stability (DMerc between -3 and 0 on s >= 0.2 at 5 % beta) at 1.5 to 14 times
  the effective ripple and 0.3e-3 to 1.8e-3 more constructed QI; NFP 2 never
  becomes stable.
- GAMMA_C_STAGES: the last stages add the smooth Gamma_c fast-ion proxy,
  normalized to its value at the stage start; 0 drops it.

Shape hinges hold the mirror ratio and the elongation.

Redl takes helicity 0 here, and not because this field is quasi-axisymmetric:
a quasi-isodynamic field carries no helical symmetry, so the isomorphism shift
iota -> iota - nfp * helicity_n is the identity. The formula was fitted on
quasisymmetric fields, so on a QI boundary it is an analytic estimate and not
a kinetic answer.

From three cold-started vacuum designs per field-period count and no current,
at 5 % beta, constructed QI is 1.7-2.0e-3 (NFP 2) and 1.9-2.7e-3 (NFP 3) with
the Mercier rows off, against 1.4-1.6e-3 and 1.6-2.2e-3 in vacuum; eps_eff^(3/2)
is 1.4e-4-5.7e-4 and 4.9e-4-1.4e-3, and no alpha is lost in 5 ms. With the
well and Mercier rows on, QI is 2.8-3.8e-3 and 2.8-3.0e-3 and eps_eff^(3/2)
1.7-2.3e-3 and 1.9-2.6e-3.

The kinetic profiles are the Landreman-Buller-Drevlak forms, ne = n0 (1 - s^5)
and Te = Ti = T0 (1 - s); the amplitude is recalibrated at each stage to the
stage's volume-average beta for this field scale.
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
                                 RedlBootstrapMismatch, self_consistent_bootstrap)
from vmex.core.qi import ConstructedQIResidual

# The vacuum QI boundary the ladder starts from, written by QI_optimization.py:
INPUT_FILE = Path("input.QI_optimized")

# Physics toggles:
BOOTSTRAP = "none"                # "none", "redl" or "dkx"
MAGNETIC_WELL_FLOOR = 0.01        # None drops the well hinge
MERCIER = False
GAMMA_C_STAGES = 0                # last stages that add the Gamma_c row
GAMMA_C_WEIGHT = 1.0              # cost it carries at the stage start

# Weights, which multiply squared residuals. The hinges are zero while their
# limit holds:
QI_WEIGHT = 100.0
HINGE_WEIGHT = 100.0              # iota floor, mirror ratio, elongation
MAGNETIC_WELL_WEIGHT = 1.0e3
ASPECT_WEIGHT = 2.0

# Radial grid every optimizer trial is solved on. A current-carrying seed can
# stall just above 1e-10, so the tolerance is one decade looser than the
# vacuum examples'; the verification solve below is still held to 1e-12:
STAGE_NS = 31
STAGE_FTOL = 1.0e-11
STAGE_NITER = 5500

# Beta ladder. Each stage is (volume-average beta, highest boundary mode number
# varied, residual evaluations, optimized I'(s) spline knots; used with a
# bootstrap current only). With a bootstrap current raise beta by at most a
# factor of two per stage: the current grows with it, and a stage that starts
# from 2.5 times the pressure (about 47 kA) leaves its seed solve stalled near
# fsq = 1e-7 on this grid. A repeated beta refines the boundary at that beta:
STAGES = [(0.01, 2, 8, 5), (0.02, 2, 6, 5), (0.05, 2, 6, 5)]
DKX_FINAL_STAGES = 2              # BOOTSTRAP = "dkx": stages that use DKX's row
DKX_WEIGHT = 10.0
KINETIC_SURFACES = (0.3, 0.6)

# Gamma_c surfaces, surrogate temperature and quadrature:
GAMMA_C_SURFACES = (0.35, 0.6, 0.85)
GAMMA_C_TEMPERATURE = 0.15
GAMMA_C_BUDGET = dict(nalpha=5, num_transit=3, points_per_transit=64, num_pitch=24,
                      quadrature_order=32)

# Weight of the beta residual, which is relative because the target is small:
BETA_RELATIVE_WEIGHT = 100.0

# Flux surfaces the QI and bootstrap residuals are evaluated on:
SURFACES = np.linspace(0.1, 0.9, 6)

# Boozer resolution the QI residual is evaluated on:
QI_OPTIONS = dict(mboz=10, nboz=10, nphi=41, nalpha=12, n_bounce=13)

# Targets and limits. None holds the seed's own value, with the iota floor at
# IOTA_FLOOR_FRACTION of the seed's minimum |iota|; a finite-beta stage then
# keeps the shape of the vacuum design it continues from:
ASPECT_TARGET = None
IOTA_FLOOR = None                 # minimum |iota| over the profile
IOTA_FLOOR_FRACTION = 0.9
MIRROR_LIMIT = None
ELONGATION_LIMIT = 8.0

# Stability rows. DMerc and DR are dimensional (their size follows the field
# strength and the pressure), so each stage normalizes its rows by the largest
# violation at the stage start: STABILITY_WEIGHT is the cost the rows carry
# there, against about 0.25 for a constructed-QI residual of 5e-3. Mercier
# coordinates are unreliable near the axis, so the weight is zero below
# STABILITY_MIN_S and rises toward the edge:
STABILITY_WEIGHT = 1.0
EDGE_WEIGHT_FACTOR = 10.0
STABILITY_MIN_S = 0.2

# Picard loop that makes the seed current self-consistent:
PICARD_ITERATIONS = 12
PICARD_TOLERANCE = 1e-3
PICARD_RELAX = 0.5                # under-relaxation: the current feeds back on iota
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

# Verification solve of each stage's boundary; the last one is the result:
FINAL_NS = 71
FINAL_FTOL = 1e-13
FINAL_NITER = 30000

# Every output file name contains this:
OUTPUT_NAME = "QI_bootstrap_optimized"
CURRENT_FIGURE = "QI_bootstrap_current.png"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    SURFACES = np.linspace(0.2, 0.8, 4)
    QI_OPTIONS = dict(mboz=8, nboz=8, nphi=31, nalpha=7, n_bounce=7)
    STAGES = [(0.02, 1, 4, 4)]
    KINETIC_SURFACES = (0.5,)
    PICARD_ITERATIONS, REDL_N_LAMBDA = 2, 12
    FINAL_NS, FINAL_FTOL = 31, 1e-10

###############################################################################
# End of input parameters.
###############################################################################

if BOOTSTRAP == "dkx":
    from dkx.bootstrap import KineticBootstrapMismatch

### Set up the equilibrium ####################################################

inp = vj.VmecInput.from_file(INPUT_FILE)

# The Landreman-Buller-Drevlak profiles. Their product gives p = 2 e ne Te;
# AM below is (1 - s)(1 - s^5), matching that shape. n0 and T0 start at the
# 5% scale and are recalibrated to each stage's beta.
am = np.zeros(21)
am[[0, 1, 5, 6]] = [1.0, -1.0, -1.0, 1.0]
ac = np.zeros(21)
ac[0] = 1.0
n0, T0 = 3.0e20, 15.0e3
inp = replace(inp, ns_array=np.array([STAGE_NS]), ftol_array=np.array([STAGE_FTOL]),
              niter_array=np.array([STAGE_NITER]), delt=0.5, pmass_type="power_series",
              am=am, pres_scale=2 * ELEMENTARY_CHARGE * n0 * T0, ncurr=1,
              pcurr_type="power_series", ac=ac, curtor=0.0)
# The limits below come from the vacuum design.
vacuum = opt.solve_equilibrium(replace(inp, pres_scale=0.0))
seed_state, seed_context = vacuum.solution, vacuum.solver_context
equilibrium = opt.solve_equilibrium(inp)
if ASPECT_TARGET is None:
    ASPECT_TARGET = float(opt.aspect_ratio(seed_state, seed_context))
if IOTA_FLOOR is None:
    IOTA_FLOOR = IOTA_FLOOR_FRACTION * float(opt.min_abs_iota(seed_state, seed_context))
if MIRROR_LIMIT is None:
    MIRROR_LIMIT = float(opt.mirror_ratio(seed_state, seed_context))
print(f"holding aspect {ASPECT_TARGET:.3f}, min |iota| >= {IOTA_FLOOR:.3f}, "
      f"mirror <= {MIRROR_LIMIT:.3f}")

### Set up the objective ######################################################

def iota_floor(equilibrium_state, solver_context):
    """Hinge on the profile minimum of |iota|: a mean target can hide a near-zero surface.

    opt.mean_iota targets the average instead; opt.soft_min_abs_iota is the smooth minimum.
    """
    return jnp.maximum(
        IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)


def mirror_excess(equilibrium_state, solver_context):
    """Hinge on the mirror ratio above its limit; zero while the limit holds."""
    return jnp.maximum(
        opt.mirror_ratio(equilibrium_state, solver_context) - MIRROR_LIMIT, 0.0)


def elongation_excess(equilibrium_state, solver_context):
    """Hinge on the cross-section elongation above its limit."""
    return jnp.maximum(
        opt.max_elongation(equilibrium_state, solver_context) - ELONGATION_LIMIT, 0.0)


def interior_surfaces(criterion):
    """The criterion on the weighted surfaces (s >= STABILITY_MIN_S) of whatever grid solved it."""
    values = criterion[2:-1]
    return values[np.linspace(0.0, 1.0, values.size + 3)[2:-1] >= STABILITY_MIN_S]


def minimum_dmerc(equilibrium_state, solver_context):
    """Minimum of the Mercier criterion there, reported not targeted."""
    return interior_surfaces(opt.d_merc_state(equilibrium_state, solver_context)).min()


def maximum_dr(equilibrium_state, solver_context):
    """Maximum of the resistive-interchange criterion there, reported not targeted."""
    return interior_surfaces(opt.glasser_d_r_state(equilibrium_state, solver_context)).max()


# Radial grading of the stability weights on the grid that solved them.
stability_s = np.linspace(0.0, 1.0, int(inp.ns_array[-1]))[2:-1]
stability_grading = np.where(stability_s >= STABILITY_MIN_S,
    1.0 + (EDGE_WEIGHT_FACTOR - 1.0) * stability_s**4, 0.0)


def stability_weights(criterion):
    """Graded weights that give the rows STABILITY_WEIGHT of cost at this violation."""
    scale = np.abs(np.asarray(interior_surfaces(criterion))).max()
    return STABILITY_WEIGHT * stability_grading / (stability_grading.sum() * scale**2)


qi = ConstructedQIResidual(SURFACES, **QI_OPTIONS)
gamma_c = opt.GammaCSmooth(GAMMA_C_SURFACES, temperature=GAMMA_C_TEMPERATURE, **GAMMA_C_BUDGET)
monitor = opt.OptimizationMonitor()

### Run the optimization ######################################################

for stage, (beta, max_mode, max_nfev, n_spline) in enumerate(STAGES):
    print(f"\n===== QI stage {stage}: beta = {beta:.1%}, max_mode = {max_mode} =====")
    # Recalibrate the pressure to this stage's beta: at fixed flux beta is
    # proportional to the pressure, and n0 T0 carries it.
    scale = beta / float(equilibrium.wout.betatotal)
    n0, T0 = n0 * scale ** (1 / 3), T0 * scale ** (2 / 3)
    inp = replace(inp, pres_scale=inp.pres_scale * scale)
    profiles = KineticProfiles(n0 * np.array([1, 0, 0, 0, 0, -1]),
                               T0 * np.array([1, -1]), T0 * np.array([1, -1]))
    bootstrap = RedlBootstrapMismatch(profiles, helicity_n=0, surfaces=SURFACES,
                                      n_lambda=REDL_N_LAMBDA)
    # Solve at the new pressure before anything reads the equilibrium: a
    # stage's hot start from the old pressure is not a converged state, and the
    # Redl current of an unconverged state is wrong.
    equilibrium = opt.solve_equilibrium(inp, initial_state=equilibrium.solution)
    use_dkx = BOOTSTRAP == "dkx" and stage >= len(STAGES) - DKX_FINAL_STAGES
    if BOOTSTRAP != "none" and not (use_dkx and stage > 0):
        # The Picard loop leaves the prescribed pressure and boundary shape
        # unchanged and updates the current profile (I'(s), CURTOR) to the
        # Redl bootstrap response at the new pressure. A DKX stage keeps the
        # current it inherits and lets the DKX row move it.
        picard = self_consistent_bootstrap(
            inp, profiles, 0, n_iter=PICARD_ITERATIONS, tol=PICARD_TOLERANCE,
            relax=PICARD_RELAX, degree=n_spline - 1, s_eval=SURFACES,
            initial_state=equilibrium.solution, verbose=not ci_smoke)
        inp, equilibrium = picard.input, picard.equilibrium

    # Each term is (function, target, weight).
    if use_dkx:
        kinetic = KineticBootstrapMismatch(profiles, KINETIC_SURFACES)

        def current_row(equilibrium_state, solver_context):
            """Equilibrium minus DKX <j.B> on the kinetic surfaces, in MA T/m^2."""
            _, j_equilibrium, j_kinetic = kinetic.current_profiles(
                equilibrium_state, solver_context)
            return (j_equilibrium - j_kinetic) / 1e6
    else:
        current_row = bootstrap
    terms = [(qi, 0.0, QI_WEIGHT),
             (opt.aspect_ratio, ASPECT_TARGET, ASPECT_WEIGHT),
             (iota_floor, 0.0, HINGE_WEIGHT),
             (mirror_excess, 0.0, HINGE_WEIGHT),
             (elongation_excess, 0.0, HINGE_WEIGHT),
             (opt.volume_average_beta, beta, BETA_RELATIVE_WEIGHT / beta**2)]
    if BOOTSTRAP != "none":
        terms.append((current_row, 0.0, DKX_WEIGHT if use_dkx else 1.0))
    if MAGNETIC_WELL_FLOOR is not None:
        terms.append((opt.Hinge(opt.magnetic_well, floor=MAGNETIC_WELL_FLOOR), 0.0,
                      MAGNETIC_WELL_WEIGHT))
    if MERCIER:
        state, context = equilibrium.solution, equilibrium.solver_context
        terms += [(opt.mercier_stability_residual, 0.0,
                   stability_weights(opt.d_merc_state(state, context))),
                  (opt.glasser_stability_residual, 0.0,
                   stability_weights(opt.glasser_d_r_state(state, context)))]
    if stage >= len(STAGES) - GAMMA_C_STAGES:
        gamma_c_total = float(gamma_c.total_state(equilibrium.solution, equilibrium.solver_context))
        terms.append((gamma_c, 0.0, GAMMA_C_WEIGHT / gamma_c_total))
    report = opt.EquilibriumReporter(
        ("constructed QI", qi.total, ".4e"), ("f_boot", bootstrap.total, ".4e"),
        ("beta", opt.volume_average_beta, ".3%"), ("aspect", opt.aspect_ratio, ".3f"),
        ("min |iota|", opt.min_abs_iota, ".3f"), ("mirror", opt.mirror_ratio, ".3f"),
        ("well", opt.magnetic_well, ".4f"),
        ("min DMerc", minimum_dmerc, ".2e"), ("max DR", maximum_dr, ".2e"),
        ("Gamma_c", gamma_c.total, ".3e"))
    report(f"stage {stage} start", equilibrium)

    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = inp.change_resolution(mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6,
                                nzeta=2 * mpol + 4)
    current_dofs = 0
    if BOOTSTRAP != "none":
        inp = opt.resample_current_profile(inp, n_spline)
        equilibrium = opt.solve_equilibrium(inp, initial_state=equilibrium.solution)
        current_dofs = n_spline - 1
    # A RuntimeWarning about uncertified Jacobian columns is expected once the
    # optimizer leaves the seed and needs no action; see examples/README.md.
    problem = opt.VmecProblem.from_tuples(
        inp, terms, max_mode=max_mode, current_dofs=current_dofs,
        vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True, restart_from=equilibrium,
        progress=not ci_smoke, evaluation_progress=not ci_smoke)
    print(f"dof_names = {problem.dof_names}")
    monitor.problem = problem
    step = PARAMETER_STEP * problem.scales
    if BOOTSTRAP != "none":
        step[-n_spline:] = CURRENT_PARAMETER_STEP  # n-1 spline shapes plus CURTOR
    result = least_squares(
        problem.residual, problem.x0, jac=problem.residual_jac, x_scale=step,
        bounds=(problem.x0 - MAX_PARAMETER_CHANGE * step,
                problem.x0 + MAX_PARAMETER_CHANGE * step),
        max_nfev=max_nfev, ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor)
    inp = problem.input_from_x(result.x)
    equilibrium = problem.equilibrium_from_x(result.x)
    report(f"stage {stage} end", equilibrium)

    # Certify every stage: the optimizer's grid is not the certificate, so
    # re-solve the stage's boundary on a finer radial grid to a tighter
    # tolerance, and save it so each beta of the ladder has its own deck.
    final_input = replace(inp, ns_array=np.array([FINAL_NS]),
        ftol_array=np.array([FINAL_FTOL]), niter_array=np.array([FINAL_NITER]))
    final_equilibrium = opt.solve_equilibrium(
        final_input, initial_state=equilibrium.solution, verbose=not ci_smoke,
        raise_on_max_iterations=False)
    # A stage with a current can stall above FINAL_FTOL (the NFP 3 deck at 5 %
    # reaches 4e-12 in 30000 iterations), so the residual is reported, not required.
    print(f"verification solve: ns = {FINAL_NS}, fsq = {float(final_equilibrium.wout.fsqr):.1e}, "
          f"ftol = {FINAL_FTOL:.0e}")
    report("final" if stage == len(STAGES) - 1 else f"stage {stage} final", final_equilibrium)
    name = f"{OUTPUT_NAME}_stage{stage}"
    print(f"Wrote {final_input.to_indata(f'input.{name}')}")
    print(f"Wrote {vj.write_wout(f'wout_{name}.nc', final_equilibrium.wout)}")

### Print, plot and save ######################################################

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}\nWrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
vj.plot_bootstrap_current(CURRENT_FIGURE, final_equilibrium, bootstrap)
print(f"Wrote {CURRENT_FIGURE}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
