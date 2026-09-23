#!/usr/bin/env python
"""Screen a vacuum QA candidate with frozen-geometry pressure stability proxies.

The mode ladder optimizes quasisymmetry, aspect ratio, a transform floor and a
magnetic well on a vacuum equilibrium, adding trial-pressure Mercier and
resistive-interchange residuals after the first stage: those evaluate the
stability criteria a small pressure *would* produce on the frozen geometry,
without solving at finite pressure.

A vacuum DMerc is only the formal zero-pressure limit, so outside the smoke
pass the script then adds 0.1 % pressure and polishes the actual finite-beta
DMerc and DR from coarse to resolved radial grids.  That resolved solve, not
the proxy, is the stability certificate.

Outputs are named ``QA_DMerc_*`` so they do not overwrite the ones
``QA_optimization.py`` writes with the same working directory.
"""

import os
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt

# Number of field periods, and the seed deck the boundary is shaped from:
NFP = 2
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"

# Rotating-ellipse amplitude added to the circular seed. The exactly circular
# torus has zero first-order iota sensitivity; this gives the optimizer a QA basin:
SEED_PERTURBATION = 0.05

# Flux surfaces the quasisymmetry residual is evaluated on:
SURFACES = np.linspace(0.1, 1.0, 10)

# Mode ladder: highest boundary mode number varied in each stage, and the
# residual evaluations each stage may spend:
MAX_MODES = [1, 2, 2, 2]
MAX_NFEV = [10, 15, 10, 15]  # Targets:
ASPECT_TARGET = 6.0
MAGNETIC_WELL_TARGET = 0.01
IOTA_FLOOR = 0.42                 # minimum |iota| over the profile

# Trial-pressure stability screen, added from the second stage on. TRIAL_BETA
# is the pressure proxy: 0.1 % beta, p(s) proportional to 1 - s. Its cost scale
# is multiplied by ten per stage, so the screen tightens as the shape settles:
USE_TRIAL_STABILITY = True
TRIAL_BETA = 0.001
STABILITY_COST_PER_SURFACE = 1.0e-2
EDGE_WEIGHT_FACTOR = 10.0         # how much harder the edge is weighted than s = 0.2
STABILITY_MIN_S = 0.2             # the singular core is excluded below this
STABILITY_MARGIN = 1.0e-3

# The finite-pressure certificate that follows the vacuum ladder: radial grids
# it is polished on, and the residual evaluations each may spend. Skipped in
# the smoke pass:
CERTIFICATE_RESOLUTIONS = [31, 71]
CERTIFICATE_MAX_NFEV = [10, 10]
CERTIFICATE_MARGIN = 5e-4
CERTIFICATE_SMOOTHING = 1e-5
CERTIFICATE_WEIGHT = 5.0
CERTIFICATE_STEP = 0.01
CERTIFICATE_MAX_CHANGE = 8.0
CALIBRATION_PRES_SCALE = 10.0     # arbitrary; one solve rescales it to TRIAL_BETA
CONTINUATION_FRACTIONS = np.linspace(0.25, 1.0, 4)
SHEAR_EPSILON = 1e-8

# Step control. One scaled variable moves a low-order coefficient by
# PARAMETER_STEP metres, and a stage may move it MAX_PARAMETER_CHANGE steps:
PARAMETER_STEP = 0.02
MAX_PARAMETER_CHANGE = 5.0
ESS_ALPHA = 1.2                   # smaller values let high Fourier modes move more
VARY_MAJOR_RADIUS = False         # True optimizes RBC(0,0) instead of fixing it

# Equilibrium resolution: poloidal and toroidal mode numbers are max_mode + 2,
# but never below MINIMUM_MPOL:
MINIMUM_MPOL = 5

# Verification solve of the optimized vacuum boundary:
FINAL_NS = 71
FINAL_FTOL = 1.0e-14
FINAL_NITER = 8000

# Output stems. The vacuum optimum and the finite-pressure certificate are
# written separately, and neither collides with QA_optimization.py:
VACUUM_NAME = "QA_DMerc_vacuum"
CERTIFICATE_NAME = "QA_DMerc_optimized"
OUTPUT_NAME = "QA_DMerc"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs. It stops
# after the vacuum ladder: the certificate below is a sequence of full solves.
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODES, MAX_NFEV = [1], [4]
    FINAL_NS, FINAL_FTOL = 31, 1.0e-10

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

# VmecInput is frozen, so copy its arrays before shaping the seed boundary.
inp = vj.VmecInput.from_file(INPUT_FILE)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
inp = replace(inp, rbc=rbc, zbs=zbs)

### Set up the objective ######################################################

def trial_dmerc(equilibrium_state, solver_context):
    """Mercier residual a TRIAL_BETA pressure would produce on this geometry."""
    return opt.trial_pressure_mercier_stability_residual(
        equilibrium_state, solver_context, beta=TRIAL_BETA,
        margin=STABILITY_MARGIN)

def trial_dr(equilibrium_state, solver_context):
    """Resistive-interchange counterpart of trial_dmerc."""
    return opt.trial_pressure_glasser_stability_residual(
        equilibrium_state, solver_context, beta=TRIAL_BETA,
        margin=STABILITY_MARGIN)

# The first QA stage excludes stability. Later stages omit the singular core
# and emphasize the difficult edge smoothly.
stability_s = np.linspace(0.0, 1.0, int(inp.ns_array[-1]))[2:-1]
stability_shape = np.where(stability_s >= STABILITY_MIN_S,
    1.0 + (EDGE_WEIGHT_FACTOR - 1.0) * stability_s**4, 0.0)

def iota_floor(equilibrium_state, solver_context):
    """Hinge on the profile minimum of |iota|: a mean target can hide a near-zero surface.

    opt.mean_iota targets the average instead; opt.soft_min_abs_iota is the smooth minimum.
    """
    return jnp.maximum(
        IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)


# Each term is (function, target, weight).
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
objective_function_terms = [
         (qs, 0.0, 1.0),
         (opt.aspect_ratio, ASPECT_TARGET, 1.0),
         (iota_floor, 0.0, 10.0),
         (opt.magnetic_well, MAGNETIC_WELL_TARGET, 1.0),
         ]

report = opt.EquilibriumReporter(
    ("QS total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("min |iota|", opt.min_abs_iota, ".4f"), ("magnetic well", opt.magnetic_well, ".4f"))
monitor = opt.OptimizationMonitor()

### Run the optimization ######################################################

# Optimize for QA first, then add the pressure-stability proxy locally.
equilibrium = opt.solve_equilibrium(inp)
for stage, (max_mode, max_nfev) in enumerate(zip(MAX_MODES, MAX_NFEV)):
    print(f"\n===== QA stage, max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    stage_terms = objective_function_terms
    if USE_TRIAL_STABILITY and stage > 0:
        # Normalize each dimensional row at the established QA seed, so the
        # Mercier rows cannot overwhelm quasisymmetry.
        dmerc0 = np.asarray(trial_dmerc(equilibrium.solution, equilibrium.solver_context))
        dr0 = np.asarray(trial_dr(equilibrium.solution, equilibrium.solver_context))
        stability_scale = np.maximum.reduce((np.abs(dmerc0), np.abs(dr0), np.ones_like(dmerc0)))
        stage_cost = STABILITY_COST_PER_SURFACE * 10.0**(stage - 1)
        stability_weights = stage_cost * stability_shape / stability_scale**2
        stage_terms = [*objective_function_terms,
            (trial_dmerc, 0.0, stability_weights),
            (trial_dr, 0.0, stability_weights)
        ]
        print(f"Adding trial-pressure stability on s >= {STABILITY_MIN_S:.1f}; "
              f"weights rise smoothly toward the edge (cost scale {stage_cost:g}).")
    # A RuntimeWarning about uncertified Jacobian columns is expected once the
    # optimizer leaves the seed and needs no action; see examples/README.md.
    problem = opt.VmecProblem.from_tuples(inp, stage_terms, max_mode=max_mode,
        vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True, ess_alpha=ESS_ALPHA,
        restart_from=equilibrium)
    print(f"dof_names = {problem.dof_names}")
    monitor.problem = problem
    if not ci_smoke:
        problem.compile_residual_and_jacobian()
    step = PARAMETER_STEP * problem.scales
    result = least_squares(
        problem.residual, problem.x0, jac=problem.residual_jac,
        x_scale=step, bounds=(problem.x0 - MAX_PARAMETER_CHANGE * step,
                             problem.x0 + MAX_PARAMETER_CHANGE * step), max_nfev=max_nfev,
        ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor
    )
    inp = problem.input_from_x(result.x)
    equilibrium = problem.equilibrium_from_x(result.x)
    report(f"mode {max_mode}", equilibrium)

### Check the result ##########################################################

# The optimizer's grid is not the certificate: re-solve the optimized boundary
# on a finer radial grid to a tighter tolerance and quote that.
final_input = replace(inp,
    ns_array=np.array([FINAL_NS]), ftol_array=np.array([FINAL_FTOL]),
    niter_array=np.array([FINAL_NITER]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution,
    verbose=not ci_smoke, raise_on_max_iterations=True)
final_total = report("final", final_equilibrium)["QS total"]
print(f"\nQS total {final_total:.3e}")
if USE_TRIAL_STABILITY:
    final_s = np.linspace(0.0, 1.0, int(final_input.ns_array[-1]))[2:-1]
    keep = final_s >= STABILITY_MIN_S
    final_dmerc = np.asarray(opt.trial_pressure_d_merc_state(
        final_equilibrium.solution, final_equilibrium.solver_context, beta=TRIAL_BETA))[2:-1]
    final_dr = np.asarray(opt.trial_pressure_glasser_d_r_state(
        final_equilibrium.solution, final_equilibrium.solver_context, beta=TRIAL_BETA,
        shear_epsilon=SHEAR_EPSILON))[2:-1]
    print(f"Trial-pressure proxy on s >= {STABILITY_MIN_S:.1f}: "
          f"min DMerc = {final_dmerc[keep].min():.3e}, max DR = {final_dr[keep].max():.3e}")

### Print, plot and save ######################################################

# The optimized vacuum equilibrium is preserved separately. With trial-pressure
# stability on, the primary output is the resolved certificate below, whose
# plotted DMerc and DR carry their finite-pressure meaning.
vacuum_input_path = final_input.to_indata(f"input.{VACUUM_NAME}")
vacuum_wout_path = vj.write_wout(f"wout_{VACUUM_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {vacuum_input_path}\nWrote {vacuum_wout_path}")

# A vacuum DMerc is only the formal zero-pressure limit.  Add TRIAL_BETA of
# pressure, then polish the actual finite-beta DMerc and DR from coarse to
# resolved radial grids.  That is a sequence of full solves, so the smoke pass
# stops above.
certificate_wout_path = None
if not ci_smoke:
    am = np.zeros(21)
    am[:2] = [1.0, -1.0]  # p(s) = PRES_SCALE * (1 - s)
    calibration_input = replace(inp, pmass_type="power_series", am=am,
        pres_scale=CALIBRATION_PRES_SCALE,
        ns_array=np.array([CERTIFICATE_RESOLUTIONS[0]]),
        ftol_array=np.array([1.0e-12]), niter_array=np.array([8000]))
    certificate = opt.solve_equilibrium(calibration_input, initial_state=equilibrium.solution)
    pressure_scale = calibration_input.pres_scale * TRIAL_BETA / float(certificate.wout.betatotal)
    for fraction in CONTINUATION_FRACTIONS:
        continuation_input = replace(calibration_input, pres_scale=fraction * pressure_scale)
        certificate = opt.solve_equilibrium(continuation_input, initial_state=certificate.solution)
    for ns, max_nfev in zip(CERTIFICATE_RESOLUTIONS, CERTIFICATE_MAX_NFEV):
        certificate_input = replace(inp, pmass_type="power_series", am=am,
            pres_scale=pressure_scale, ns_array=np.array([ns]), ftol_array=np.array([1e-12]),
            niter_array=np.array([16000]))
        certificate = opt.solve_equilibrium(certificate_input, initial_state=certificate.solution,
            raise_on_max_iterations=True)
        s = np.linspace(0.0, 1.0, ns)[2:-1]
        dmerc = np.asarray(opt.d_merc_state(certificate.solution, certificate.solver_context))[2:-1]
        dr = np.asarray(opt.glasser_d_r_state(
            certificate.solution, certificate.solver_context,
            shear_epsilon=SHEAR_EPSILON))[2:-1]
        shape = np.where(s >= STABILITY_MIN_S,
            1.0 + (EDGE_WEIGHT_FACTOR - 1.0) * s**4, 0.0)
        scale = np.maximum.reduce(
            (np.abs(dmerc), np.abs(dr), np.full_like(dmerc, CERTIFICATE_MARGIN)))

        def finite_beta_dmerc(equilibrium_state, solver_context):
            """Mercier residual of the actual finite-pressure equilibrium."""
            return opt.mercier_stability_residual(
                equilibrium_state, solver_context, margin=CERTIFICATE_MARGIN,
                smoothing=CERTIFICATE_SMOOTHING)

        def finite_beta_dr(equilibrium_state, solver_context):
            """Resistive-interchange counterpart of finite_beta_dmerc."""
            return opt.glasser_stability_residual(
                equilibrium_state, solver_context, margin=CERTIFICATE_MARGIN,
                smoothing=CERTIFICATE_SMOOTHING)

        print(f"Polishing the physical 0.1%-beta stability certificate at NS={ns}; "
              f"the core s < {STABILITY_MIN_S:.1f} is excluded and edge weights increase smoothly.")
        certificate_terms = [*objective_function_terms,
            (finite_beta_dmerc, 0.0, CERTIFICATE_WEIGHT * shape / scale**2),
            (finite_beta_dr, 0.0, CERTIFICATE_WEIGHT * shape / scale**2)]
        certificate_problem = opt.VmecProblem.from_tuples(certificate_input, certificate_terms,
            max_mode=MAX_MODES[-1], use_ess=True, ess_alpha=ESS_ALPHA,
            restart_from=certificate, progress=True, evaluation_progress=True)
        step = CERTIFICATE_STEP * certificate_problem.scales
        certificate_result = least_squares(certificate_problem.residual, certificate_problem.x0,
            jac=certificate_problem.residual_jac, x_scale=step,
            bounds=(certificate_problem.x0 - CERTIFICATE_MAX_CHANGE * step,
                    certificate_problem.x0 + CERTIFICATE_MAX_CHANGE * step),
            max_nfev=max_nfev, ftol=1e-7, xtol=1e-10, verbose=2)
        certificate_input = certificate_problem.input_from_x(certificate_result.x)
        certificate = certificate_problem.equilibrium_from_x(certificate_result.x)
    certificate_input = replace(certificate_input, ftol_array=np.array([1e-14]))
    certificate = opt.solve_equilibrium(certificate_input, initial_state=certificate.solution,
        verbose=True, raise_on_max_iterations=True)
    certificate_input_path = certificate_input.to_indata(f"input.{CERTIFICATE_NAME}")
    certificate_wout_path = vj.write_wout(f"wout_{CERTIFICATE_NAME}.nc", certificate.wout)
    certificate_s = np.linspace(0.0, 1.0, int(certificate.wout.ns))[2:-1]
    keep = certificate_s >= STABILITY_MIN_S
    certificate_dmerc = np.asarray(certificate.wout.DMerc)[2:-1]
    certificate_dr = np.asarray(opt.glasser_d_r_state(
        certificate.solution, certificate.solver_context,
        shear_epsilon=SHEAR_EPSILON))[2:-1]
    print(f"0.1%-beta certificate: beta={certificate.wout.betatotal:.4e}, "
          f"min DMerc={certificate_dmerc[keep].min():.3e}, "
          f"max DR={certificate_dr[keep].max():.3e} on s >= {STABILITY_MIN_S:.1f}")
    print(f"Wrote {certificate_input_path}\nWrote {certificate_wout_path}")

print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
for path in vj.plot_wout(vacuum_wout_path, ".").values():
    print(f"Wrote {path}")
if certificate_wout_path is not None:
    for path in vj.plot_wout(certificate_wout_path, ".").values():
        print(f"Wrote {path}")
