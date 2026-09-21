#!/usr/bin/env python
"""Optimize a boundary for quasi-axisymmetry at finite beta.

This is the vector-residual counterpart to
``QA_optimization_finite_beta_scalar.py``. SciPy least squares receives the
full residual vector and its exact Jacobian, while VMEX supplies the implicit
equilibrium derivatives.

The pressure is a prescribed p(s) = PRES_SCALE (1 - s); one calibration solve
rescales its amplitude to TARGET_BETA before the ladder starts.
"""

import os
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex.core.errors import VmecError
from vmex import optimize as opt

NFP = 2
TARGET_BETA = 0.01
PHIEDGE = 1.0                    # fixed toroidal-flux normalization [Wb]
CALIBRATION_PRES_SCALE = 100.0
CALIBRATION_TOL = 1.0e-4
CALIBRATION_ATTEMPTS = 32
CALIBRATION_MAX_RATIO = 1.5
SURFACES = np.linspace(0.1, 0.9, 8)
MAX_MODES = [ 1,  1,  1,  2,  2]
MAX_NFEV  = [10, 10, 20, 20, 20]
ASPECT_TARGET = 6.0
IOTA_FLOOR = 0.42                 # minimum |iota| over the profile
MAGNETIC_WELL_TARGET = 0.01
STABILITY_MIN_S = 0.2
STABILITY_WEIGHT = 1e-6
EDGE_WEIGHT_FACTOR = 10.0
PARAMETER_STEP = 0.01
MAX_PARAMETER_CHANGE = 3.0
ESS_ALPHA = 1.2                   # smaller values let high Fourier modes move more
MINIMUM_MPOL = 5
VARY_MAJOR_RADIUS = False         # True optimizes RBC(0,0) instead of fixing it
SEED_PERTURBATION = 0.05
POLISH_FORCE_BALANCE = False      # True polishes only the final saved state

# Verification solve of the optimized boundary:
FINAL_NS = 35
FINAL_FTOL = 1e-14
FINAL_NITER = 20000

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

INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"
inp = vj.VmecInput.from_file(INPUT_FILE)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
am = np.zeros(21)
am[:2] = [1.0, -1.0]  # p(s) = PRES_SCALE * (1-s)
inp = replace(inp, rbc=rbc, zbs=zbs, pmass_type="power_series", am=am,
              phiedge=PHIEDGE, pres_scale=CALIBRATION_PRES_SCALE)


def calibrate_pressure(input_data, initial_state=None):
    """Continuation-rescale pressure until this boundary reaches TARGET_BETA."""
    pressure_scale = float(input_data.pres_scale)
    state = initial_state
    for _attempt in range(CALIBRATION_ATTEMPTS):
        candidate = replace(input_data, phiedge=PHIEDGE, pres_scale=pressure_scale)
        restart = None if state is None else getattr(state, "solution", state)
        try:
            solved = opt.solve_equilibrium(candidate, initial_state=restart)
        except VmecError:
            if restart is None:
                raise
            # A changed boundary can invalidate a hot restart. Retry this
            # pressure point from VMEC's own seed before changing pressure.
            solved = opt.solve_equilibrium(candidate)
        state = solved
        beta = float(state.wout.betatotal)
        if abs(beta - TARGET_BETA) <= CALIBRATION_TOL:
            return candidate, state
        ratio = TARGET_BETA / max(beta, 1.0e-12)
        ratio = np.clip(ratio, 1.0 / CALIBRATION_MAX_RATIO, CALIBRATION_MAX_RATIO)
        pressure_scale *= float(ratio)
    raise RuntimeError(
        f"Could not calibrate beta at aspect {float(state.wout.aspect):.6f}: "
        f"target={TARGET_BETA:.6e}, achieved={beta:.6e}, "
        f"phiedge={PHIEDGE:.6e}")


inp, equilibrium = calibrate_pressure(inp)

stability_s = np.linspace(0.0, 1.0, int(inp.ns_array[-1]))[2:-1]
stability_weights = np.where(
    stability_s >= STABILITY_MIN_S,
    STABILITY_WEIGHT * (1.0 + (EDGE_WEIGHT_FACTOR - 1.0) * stability_s**4), 0.0)


def iota_floor(state, runtime):
    return jnp.maximum(IOTA_FLOOR - opt.min_abs_iota(state, runtime), 0.0)


### Set up the objective ######################################################


qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
objective_function_terms = [
    (qs, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (iota_floor, 0.0, 10.0),
    (opt.magnetic_well, MAGNETIC_WELL_TARGET, 1.0),
    (opt.volume_average_beta, TARGET_BETA, 1.0 / TARGET_BETA**2),
    (opt.mercier_stability_residual, 0.0, stability_weights),
    (opt.glasser_stability_residual, 0.0, stability_weights)]


report = opt.EquilibriumReporter(
    ("QS", qs.total, ".4e"), ("beta", opt.volume_average_beta, ".3%"),
    ("aspect", opt.aspect_ratio, ".3f"),
    ("min |iota|", opt.min_abs_iota, ".3f"))
monitor = opt.OptimizationMonitor(stream=None)

### Run the optimization ######################################################

previous_mpol = None
for max_mode, max_nfev in zip(MAX_MODES, MAX_NFEV):
    print(f"\n===== finite-beta QA stage, max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    if previous_mpol != mpol:
        inp, equilibrium = calibrate_pressure(inp)
    else:
        print("reusing beta-calibrated state at unchanged resolution")
    previous_mpol = mpol
    print(f"calibrated: aspect={float(equilibrium.wout.aspect):.6f}, "
          f"phiedge={PHIEDGE:.6e}, pres_scale={inp.pres_scale:.6e}, "
          f"beta={float(equilibrium.wout.betatotal):.6e}")
    problem = opt.VmecProblem.from_tuples(
        inp, objective_function_terms, max_mode=max_mode,
        vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True,
        ess_alpha=ESS_ALPHA, restart_from=equilibrium,
        progress=not ci_smoke, evaluation_progress=not ci_smoke)
    print(f"dof_names = {problem.dof_names}")
    monitor.problem = problem
    if not ci_smoke:
        problem.compile_residual_and_jacobian()

    step = PARAMETER_STEP * problem.scales
    result = least_squares(
        problem.residual, problem.x0, jac=problem.residual_jac,
        x_scale=step, max_nfev=max_nfev,
        bounds=(problem.x0 - MAX_PARAMETER_CHANGE * step,
                problem.x0 + MAX_PARAMETER_CHANGE * step),
        ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor)
    inp = problem.input_from_x(result.x)
    equilibrium = problem.equilibrium_from_x(result.x)
    inp, equilibrium = calibrate_pressure(inp, equilibrium.solution)
    print(f"consistent: aspect={float(equilibrium.wout.aspect):.6f}, "
          f"phiedge={PHIEDGE:.6e}, pres_scale={inp.pres_scale:.6e}, "
          f"beta={float(equilibrium.wout.betatotal):.6e}")
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
