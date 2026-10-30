"""Reduce transport from a nearly circular NFP4 vacuum torus.

The mapped-field variance, aspect ratio and iota floor define the objective.
NEO independently screens transport; this objective does not certify pwO or
particle confinement. Install vmex[neoclassical]; edit the inputs below.
"""
import os
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex._compat import require_optional
from vmex.core.monitoring import FeasibleCandidateMonitor
from vmex.core.neoclassical import screen_transport, transport_improves
from vmex.core.pwo import MappedPwOResidual
from vmex.core.implicit import iota_profile

INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "input.minimal_seed_nfp4"
OUTPUT_DIR = Path("low_transport")
SEED_PERTURBATION = .005          # 5% of the minor radius, with positive transform
MAX_MODES, MAX_PARAMETER_CHANGE = [1, 2], 5.
# Fixed calibration: preserve flux, length scale and B_REF across continuations.
SURFACES = (.25, 16.5 / 34, .75)
B_REF = np.array([2.7123510440502594, 2.699696279828112, 2.6832357706580665])
B_MAX_REF = np.array([2.9284215849767965, 2.979692353278536, 3.008858221344288])
RAW_SCALE = .004813522516264787
ASPECT_TARGET, IOTA_TARGET = 10., .42
MAX_ITERATIONS, NITER = 15, 2000
NS, MPOL, NTOR, FTOL = 35, 5, 5, 1e-12
OPTIMIZER_FTOL = 1e-3

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_ITERATIONS = 1

def geometry_residuals(state, runtime):
    """Aspect and signed full-profile iota; an average can hide a low-transform surface."""
    full = iota_profile(state, runtime)
    return jnp.concatenate((jnp.atleast_1d((opt.aspect_ratio(state, runtime)-ASPECT_TARGET)/.1),
                            jnp.sqrt(10/full.size)*jnp.maximum((IOTA_TARGET-full)/.01, 0)))


def objective(state, runtime):
    residual = jnp.sqrt(2)*jnp.concatenate((
        mapped.residuals_state(state, runtime)/jnp.sqrt(RAW_SCALE),
        geometry_residuals(state, runtime)))
    return residual


def inspect_candidate(eq):
    residual, fields = diagnostics(eq.state, eq.runtime)
    metrics = mapped.physical_metrics(eq, bmax_ref=1.1*B_MAX_REF,
                                     aspect_bounds=(9.9, 10.1), iota_floor=.41, fields=fields)
    return np.asarray(residual), metrics


vj.configure()
require_optional("neo-jax", "low-transport NEO screen")
print("[setup] near-circular vacuum torus, boundary modes 1 then 2", flush=True)
inp = vj.VmecInput.from_file(INPUT_FILE)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor, 1] = zbs[inp.ntor, 1] = .1
rbc[inp.ntor+1, 1], zbs[inp.ntor+1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
inp = replace(inp, rbc=rbc, zbs=zbs)
inp = replace(inp, ns_array=np.array([NS]), ftol_array=np.array([FTOL]),
              niter_array=np.array([NITER]), delt=.5).change_resolution(
                  mpol=MPOL, ntor=NTOR, ntheta=16, nzeta=14)
seed = opt.solve_equilibrium(inp, raise_on_max_iterations=True)
mapped = MappedPwOResidual(SURFACES, b_ref=B_REF, mboz=8, nboz=8, oversample=2)
problem = opt.VmecProblem.from_tuples(
    inp, [(objective, 0., 1.)], max_mode=2, vary_major_radius=False,
    use_ess=True, ess_alpha=1.2, restart_from=seed, forward_ftol=FTOL,
    forward_max_iterations=NITER, refine_tol=FTOL, adjoint_tol=1e-8, max_fsq_ratio=1.)
# JIT shares the Boozer transform between the residual and independent field checks.
diagnostics = jax.jit(lambda state, runtime: (objective(state, runtime),
                                             mapped.diagnostics_state(state, runtime)))
monitor = FeasibleCandidateMonitor(problem, inspect_candidate, OUTPUT_DIR,
    maxiter=MAX_ITERATIONS, optimizer_ftol=OPTIMIZER_FTOL, particle_qualified=False,
    input=str(INPUT_FILE), beta_target=0., seed_perturbation_m=SEED_PERTURBATION,
    settings=dict(ns=NS, mpol=MPOL, ntor=NTOR, ftol=FTOL, niter=NITER))
with monitor:
    x = problem.x0.copy()
    monitor(x)
    problem.input_from_x(x).to_indata(OUTPUT_DIR / "input.seed")
    vj.write_wout(OUTPUT_DIR / "wout_seed.nc", problem.equilibrium_from_x(x).wout)

    # Reuse one equilibrium/derivative compilation while exposing boundary modes in stages.
    for mode in MAX_MODES:
        print(f"[optimize] max_mode={mode}, at most {MAX_ITERATIONS} iterations", flush=True)
        stage = problem.subproblem(max_mode=mode, x=x)
        step = .02*np.asarray(stage.scales)
        result = least_squares(lambda y: monitor(stage.embed(y)), stage.x0,
            jac=stage.residual_jac, x_scale=step,
            bounds=(stage.x0-MAX_PARAMETER_CHANGE*step, stage.x0+MAX_PARAMETER_CHANGE*step),
            max_nfev=5*MAX_ITERATIONS+5, ftol=OPTIMIZER_FTOL, xtol=1e-10, gtol=1e-7,
            callback=monitor.callback(mode), verbose=2)
        x = stage.embed(result.x)
        monitor(x)
        problem.input_from_x(x).to_indata(OUTPUT_DIR / f"input.mode{mode}")

    monitor.save(stopping_reason=str(result.message),
                 transport_screen_passed=False)
    print("[verify] matched NEO screen of the best feasible candidate", flush=True)
    if monitor.best is not None:
        before = screen_transport(vj.read_wout(OUTPUT_DIR / "wout_seed.nc"), SURFACES)
        after = screen_transport(vj.read_wout(OUTPUT_DIR / "wout_trial.nc"), SURFACES)
        monitor.save(baseline_Q=before.tolist(), trial_Q=after.tolist(),
                     transport_screen_passed=transport_improves(before, after))
        print(f"NEO mean epsilon_eff^(3/2): {before.mean():.6g} -> {after.mean():.6g}")
        for path in vj.plot_wout(OUTPUT_DIR / "wout_trial.nc", OUTPUT_DIR).values():
            print(f"Wrote {path}")
    monitor.save(status="completed")
    print(f"Wrote {OUTPUT_DIR / 'summary.json'}. Transport screen passed: "
          f"{monitor.report['transport_screen_passed']}. Particle tracing is a separate check.")
