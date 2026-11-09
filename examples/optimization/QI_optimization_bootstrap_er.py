#!/usr/bin/env python
"""Optimize a finite-beta QI boundary whose bootstrap current is self-consistent with DKX at the ambipolar E_r.

QI_optimization_bootstrap_dkx.py with a nonzero radial electric field and a
fixed-point structure. In a quasi-isodynamic field the bootstrap current
depends strongly on E_r, so the kinetic <j.B> of each kinetic surface is
evaluated at its own ambipolar ion root (dkx.bootstrap.KineticBootstrapMismatch
with ambipolar=True; pip install "dkx>=2.12.0"). The shape and the current are
alternated:

1. Picard step: the equilibrium is solved, DKX gives <j.B> at the ambipolar E_r
   on the kinetic surfaces, and the Redl <j.B> profile is rescaled by the
   DKX/Redl ratio interpolated between those surfaces. The enclosed current
   profile that carries that <j.B> (VMEX's section 6.2 identity) becomes the
   applied current, and the solve is repeated until it stops changing.
2. Shape stage: the boundary is optimized at that fixed current, against the
   constructed-QI residual and the aspect, rotational-transform, mirror-ratio
   and beta constraints, with one more mode number each stage.

DKX is only evaluated forward, a few times per Picard step, so the optimizer's
program holds no root finder and the ladder costs what a Redl-only QI
optimization costs. The final equilibrium is judged by the rms difference
between its own <j.B> and the DKX one, quoted at the end.
"""

import os
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.bootstrap import (ELEMENTARY_CHARGE, KineticProfiles,
                                 RedlBootstrapMismatch, j_dot_B_redl,
                                 redl_geometry_from_state,
                                 self_consistent_bootstrap)
from vmex.core.qi import ConstructedQIResidual
from dkx.bootstrap import KineticBootstrapMismatch

# The seed deck: an nfp = 2 vacuum QI boundary (constructed QI 5e-3 at zero beta):
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "input.QI_nfp2_initial"

# Radial grid every optimizer trial is solved on:
STAGE_NS = 31
STAGE_FTOL = 1.0e-12
STAGE_NITER = 5500

# Pressure the profiles are calibrated to. The target is small, so the beta
# residual is relative:
TARGET_BETA = 0.015
BETA_WEIGHT = 400.0 / TARGET_BETA**2

# Flux surfaces the QI residuals and the Picard Redl profile are evaluated on:
SURFACES = np.linspace(0.1, 0.9, 6)

# Surfaces of the DKX kinetic rows (their grid None keeps the DKX defaults),
# the first guess of the ion-root E_r in kV/m and the Newton tolerance on E_r:
KINETIC_SURFACES = (0.2, 0.4, 0.6, 0.8)
KINETIC_RESOLUTION = None
ER_INITIAL = -2.0
ER_TOLERANCE = 0.05

# Mode ladder: highest boundary mode number varied in each stage, the residual
# evaluations each stage may spend, and the I'(s) spline knots the current
# profile is carried on between stages:
MAX_MODES = [1, 3]
MAX_NFEV = [5, 12]
N_CURRENT_SPLINE = 6

# Boozer resolution the QI residual is evaluated on:
QI_OPTIONS = dict(mboz=10, nboz=10, nphi=41, nalpha=12, n_bounce=13)

# Constraints, each a one-sided hinge:
ASPECT_LIMIT = 10.0
IOTA_FLOOR = 0.5                  # minimum |iota| over the profile
MIRROR_LIMIT = 0.21
MIRROR_HINGE_FRACTION = 0.99      # the hinge starts pulling just below the limit
HINGE_WEIGHT = 1.0e4

# Picard loop: iteration cap and tolerance on the relative change of I'(s), and
# the final self-consistency requirement (rms of <j.B> VMEX - DKX over rms DKX):
PICARD_ITERATIONS = 6
PICARD_TOLERANCE = 1.0e-2
PICARD_DEGREE = 9
PICARD_SURFACES = np.linspace(0.02, 0.98, 25)   # the profile is interpolated onto this grid
CONSISTENCY_LIMIT = 0.05
REDL_N_LAMBDA = 16

# Step control. Boundary coefficients move in metres; the current dofs are
# dimensionless but are held fixed in the shape stages:
ESS_ALPHA = 1.2
MAX_PARAMETER_CHANGE = 10.0       # per-stage box guardrail, in scaled step units
VARY_MAJOR_RADIUS = False         # True optimizes RBC(0,0) instead of fixing it

# Equilibrium resolution: poloidal and toroidal mode numbers are max_mode + 2,
# but never below MINIMUM_MPOL:
MINIMUM_MPOL = 5

# Verification solve of the optimized boundary:
FINAL_NS = 31                     # the stage grid; research: 51
FINAL_FTOL = 1.0e-10
FINAL_NITER = 20000

# Every output file name contains this:
OUTPUT_NAME = "QI_bootstrap_er_optimized"
CURRENT_FIGURE = "QI_bootstrap_er_optimized_current.png"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    SURFACES = np.linspace(0.3, 0.7, 3)
    QI_OPTIONS = dict(mboz=8, nboz=8, nphi=31, nalpha=7, n_bounce=7)
    MAX_MODES, MAX_NFEV = [1], [3]
    PICARD_ITERATIONS, REDL_N_LAMBDA = 2, 12
    PICARD_SURFACES = np.linspace(0.1, 0.9, 9)
    PICARD_DEGREE = 6
    KINETIC_SURFACES = (0.5,)
    CONSISTENCY_LIMIT = 1.0

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

inp = replace(vj.VmecInput.from_file(INPUT_FILE),
              ns_array=np.array([STAGE_NS]), ftol_array=np.array([STAGE_FTOL]),
              niter_array=np.array([STAGE_NITER]), delt=0.5)
mpol = max(max(MAX_MODES) + 2, MINIMUM_MPOL)
inp = inp.change_resolution(mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)

# The Landreman-Buller-Drevlak profiles. Their product gives p = 2 e ne Te;
# AM below is (1 - s)(1 - s^5), matching that shape.
n0 = 3.0e20 * (TARGET_BETA / 0.05) ** (1 / 3)
T0 = 15.0e3 * (TARGET_BETA / 0.05) ** (2 / 3)
am = np.zeros(21)
am[[0, 1, 5, 6]] = [1.0, -1.0, -1.0, 1.0]
ac = np.zeros(21)
ac[0] = 1.0
inp = replace(inp, pmass_type="power_series", am=am,
              pres_scale=2 * ELEMENTARY_CHARGE * n0 * T0, ncurr=1,
              pcurr_type="power_series", ac=ac, curtor=0.0)

# One seed solve calibrates the profile amplitude to the requested beta.
seed = opt.solve_equilibrium(inp)
profile_scale = TARGET_BETA / float(seed.wout.betatotal)
n0 *= profile_scale ** (1 / 3)
T0 *= profile_scale ** (2 / 3)
inp = replace(inp, pres_scale=inp.pres_scale * profile_scale)

# These polynomials provide ne(s), Te(s) and Ti(s) to Redl and DKX.
profiles = KineticProfiles(n0 * np.array([1, 0, 0, 0, 0, -1]),
                           T0 * np.array([1, -1]), T0 * np.array([1, -1]))
redl = RedlBootstrapMismatch(profiles, helicity_n=0, surfaces=SURFACES,
                             n_lambda=REDL_N_LAMBDA)
kinetic = KineticBootstrapMismatch(profiles, KINETIC_SURFACES, resolution=KINETIC_RESOLUTION,
                                   minor_radius=float(seed.wout.Aminor_p),
                                   ambipolar=True, er_initial=ER_INITIAL, er_tol=ER_TOLERANCE,
                                   spectral_angles=True)
bootstrap = redl  # for the Redl overlay figure


### The self-consistent current ###############################################

_programs = {}


def dkx_rows(equilibrium):
    """(s, VMEX, DKX) <j.B> on the kinetic surfaces in MA T/m^2, and the ambipolar E_r in kV/m.

    The DKX call is jitted once per radial grid with the enclosed current of the
    runtime as its argument. A runtime differs between equilibria of one
    resolution only in that array, so every later call, at any current profile,
    reuses the compiled program instead of recompiling the drift-kinetic solve.
    """
    rt = equilibrium.runtime
    key = int(rt.setup.s_full.size)
    if key not in _programs:
        _programs[key] = jax.jit(lambda state, icurv: kinetic.current_profiles(
            state, replace(rt, setup=replace(rt.setup, icurv=icurv))))
    s, j_vmex, j_dkx = _programs[key](equilibrium.state, rt.setup.icurv)
    return np.asarray(s), np.asarray(j_vmex) / 1e6, np.asarray(j_dkx) / 1e6, kinetic.electric_field()


def kinetic_rows(equilibrium):
    """(s, VMEX, Redl, DKX) <j.B> on the kinetic surfaces in MA T/m^2 and E_r in kV/m."""
    s, j_vmex, j_dkx, e_r = dkx_rows(equilibrium)
    geometry = redl_geometry_from_state(equilibrium.state, equilibrium.runtime, surfaces=s,
                                        n_lambda=REDL_N_LAMBDA)
    return s, j_vmex, np.asarray(j_dot_B_redl(profiles, geometry, 0)[0]) / 1e6, j_dkx, e_r


def inconsistency(equilibrium):
    """rms(<j.B> VMEX - DKX) / rms(<j.B> DKX) on the kinetic surfaces."""
    _, j_vmex, j_dkx, _ = dkx_rows(equilibrium)
    return float(np.sqrt(np.mean((j_vmex - j_dkx) ** 2) / np.mean(j_dkx**2)))


def kinetic_current(equilibrium, s_eval, j_redl):
    """DKX <j.B> on s_eval: monotone cubic through the kinetic surfaces, zero at s = 0 and 1.

    Redl's formula is not a model of a QI field, so the profile is the kinetic
    one and Redl's ``j_redl`` is ignored.
    """
    s, _, j_dkx, _ = dkx_rows(equilibrium)
    return 1e6 * PchipInterpolator(np.r_[0.0, s, 1.0], np.r_[0.0, j_dkx, 0.0])(s_eval)


def make_consistent(deck, equilibrium):
    """Picard iteration of the applied current to the DKX <j.B>; returns (deck, equilibrium)."""
    picard = self_consistent_bootstrap(
        deck, profiles, 0, n_iter=PICARD_ITERATIONS, tol=PICARD_TOLERANCE,
        degree=PICARD_DEGREE, s_eval=PICARD_SURFACES, current_model=kinetic_current,
        initial_state=None if equilibrium is None else equilibrium.solution,
        verbose=not ci_smoke)
    deck = picard.input
    return deck, opt.solve_equilibrium(deck, initial_state=picard.equilibrium.solution)


### Set up the objective ######################################################

def iota_floor(equilibrium_state, solver_context):
    """Hinge on the profile minimum of |iota|: a mean target can hide a near-zero surface."""
    return jnp.maximum(
        IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)


def aspect_excess(equilibrium_state, solver_context):
    """Hinge on the aspect ratio above its limit."""
    return jnp.maximum(opt.aspect_ratio(equilibrium_state, solver_context) - ASPECT_LIMIT, 0.0)


def mirror_excess(equilibrium_state, solver_context):
    """Hinge on the mirror ratio, pulling from MIRROR_HINGE_FRACTION of the limit."""
    return jnp.maximum(
        opt.mirror_ratio(equilibrium_state, solver_context)
        - MIRROR_HINGE_FRACTION * MIRROR_LIMIT, 0.0)


# Each term is (function, target, weight).
qi = ConstructedQIResidual(SURFACES, **QI_OPTIONS)
objective_function_terms = [
    (qi, 0.0, 100.0),
    (aspect_excess, 0.0, HINGE_WEIGHT),
    (iota_floor, 0.0, HINGE_WEIGHT),
    (mirror_excess, 0.0, HINGE_WEIGHT),
    (opt.volume_average_beta, TARGET_BETA, BETA_WEIGHT),
]

report = opt.EquilibriumReporter(
    ("constructed QI", qi.total, ".4e"), ("f_boot Redl", redl.total, ".4e"),
    ("beta", opt.volume_average_beta, ".3%"), ("aspect", opt.aspect_ratio, ".3f"),
    ("min |iota|", opt.min_abs_iota, ".3f"), ("mirror", opt.mirror_ratio, ".3f"))
monitor = opt.OptimizationMonitor()

### Run the optimization ######################################################

print("===== seed: Picard current at the ambipolar E_r =====")
inp, equilibrium = make_consistent(inp, None)
seed_equilibrium = equilibrium
report("self-consistent seed", equilibrium)
print(f"seed <j.B> inconsistency (rms, VMEX vs DKX): {inconsistency(equilibrium):.3f}")

# One problem serves every stage: the boundary dofs of the largest stage plus
# the current spline, of which a stage frees the boundary only, so every
# jitted residual and Jacobian is compiled once and the Picard steps between
# stages change only the current entries of x.
inp = opt.resample_current_profile(inp, N_CURRENT_SPLINE)
problem = opt.VmecProblem.from_tuples(
    inp, objective_function_terms, max_mode=max(MAX_MODES),
    current_dofs=N_CURRENT_SPLINE - 1, vary_major_radius=VARY_MAJOR_RADIUS,
    use_ess=True, ess_alpha=ESS_ALPHA, progress=not ci_smoke)
monitor.problem = problem
x = problem.x0
costs = []
for max_mode, max_nfev in zip(MAX_MODES, MAX_NFEV):
    print(f"\n===== QI stage, max_mode = {max_mode} =====")
    names = problem.stage_dof_names(max_mode)[:-N_CURRENT_SPLINE]  # the boundary only
    stage = problem.subproblem(active=names, x=x)
    print(f"dof_names = {stage.dof_names}")
    cost_start = float(np.sum(stage.residual(stage.x0) ** 2))
    result = least_squares(
        stage.residual, stage.x0, jac=stage.residual_jac, x_scale=stage.scales,
        bounds=(stage.x0 - MAX_PARAMETER_CHANGE * stage.scales,
                stage.x0 + MAX_PARAMETER_CHANGE * stage.scales),
        max_nfev=max_nfev, ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor)
    costs += [cost_start, 2 * result.cost]
    x = stage.embed(result.x)
    inp = problem.input_from_x(x)
    equilibrium = problem.equilibrium_from_x(x)
    report(f"mode {max_mode}", equilibrium)
    print(f"--- Picard step after mode {max_mode} ---")
    inp, equilibrium = make_consistent(inp, equilibrium)
    report(f"mode {max_mode}, consistent current", equilibrium)
    x = problem.x_from_input(opt.resample_current_profile(inp, N_CURRENT_SPLINE))

### Check the result ##########################################################

# The optimizer's grid is not the certificate: re-solve the optimized boundary
# on the verification grid and judge the current against DKX there.
final_input = replace(inp, ns_array=np.array([FINAL_NS]),
    ftol_array=np.array([FINAL_FTOL]), niter_array=np.array([FINAL_NITER]))
final_equilibrium = opt.solve_equilibrium(final_input, initial_state=equilibrium.solution,
    verbose=not ci_smoke, raise_on_max_iterations=True)

### Print, plot and save ######################################################

print(f"cost: initial {costs[0]:.6e} -> final {costs[-1]:.6e}")
report("final", final_equilibrium)
error = inconsistency(final_equilibrium)
for row in zip(*kinetic_rows(final_equilibrium)):
    print(f"[final] s = {row[0]:.3f}: <j.B> VMEX {row[1]:+.4f}, Redl {row[2]:+.4f}, "
          f"DKX {row[3]:+.4f} MA T/m^2, ambipolar E_r {row[4]:+.2f} kV/m")
print(f"[final] <j.B> inconsistency (rms, VMEX vs DKX): {error:.3f}")
assert costs[-1] < costs[0], "the optimization did not lower the objective"
assert error < CONSISTENCY_LIMIT, "the equilibrium current does not match DKX"

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}\nWrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
vj.plot_bootstrap_current(CURRENT_FIGURE, final_equilibrium, bootstrap)
print(f"Wrote {CURRENT_FIGURE}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
