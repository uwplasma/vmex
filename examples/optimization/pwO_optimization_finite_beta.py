"""Continue the vacuum pwO equilibrium at beta=0.5%, with p proportional to 1-s.

Run pwO_optimization.py first. Pressure is calibrated at fixed toroidal
flux, then the boundary is optimized and checked with NEO."""
import json
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
from vmex.core.pwo import MappedPwOResidual, transport_backtrack
from vmex.core.implicit import iota_profile
from vmex.core.restart import restart_state

INPUT_FILE = Path("low_transport/input.trial")
OUTPUT_DIR = Path("low_transport_finite_beta")
TARGET_BETA = .005
MAX_MODES, MAX_PARAMETER_CHANGE = [2], .25
SURFACES = jnp.linspace(.25, .75, 3)
ASPECT_TARGET, IOTA_TARGET = 6., .42
MIRROR_LIMIT, MIRROR_WEIGHT = .2, 100.
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
    fields = mapped.diagnostics_state(state, runtime)
    residual = jnp.sqrt(2)*jnp.concatenate((
        fields[0]["residuals1d"]/jnp.sqrt(initial_scale),
        geometry_residuals(state, runtime),
        jnp.sqrt(MIRROR_WEIGHT)*jnp.maximum(fields[3]/(.9*MIRROR_LIMIT)-1, 0)))
    return jnp.concatenate((residual, jnp.atleast_1d(
        opt.volume_average_beta(state, runtime)/TARGET_BETA-1)))


def inspect_candidate(eq):
    residual, fields = diagnostics(eq.state, eq.runtime)
    metrics = mapped.physical_metrics(eq, mirror_ceiling=MIRROR_LIMIT,
        aspect_bounds=(ASPECT_TARGET-.1, ASPECT_TARGET+.1), iota_floor=.41, fields=fields)
    metrics["feasible"] &= .9*TARGET_BETA <= metrics["beta"] <= 1.1*TARGET_BETA
    return np.asarray(residual), metrics


vj.configure()
require_optional("neo-jax", "low-transport NEO screen")
print("[setup] finite-beta continuation at fixed flux", flush=True)
vacuum_report = json.loads(INPUT_FILE.with_name("summary.json").read_text())
if (vacuum_report.get("status") != "completed"
        or vacuum_report.get("transport_screen_passed") is not True):
    raise ValueError("Use a completed vacuum pwO candidate that passes its NEO transport screen")
inp = vj.VmecInput.from_file(INPUT_FILE)
if (inp.lasym or inp.lfreeb or inp.phiedge <= 0
        or inp.rbc[inp.ntor, 0] != 1 or inp.ncurr != 1 or inp.curtor != 0
        or inp.pmass_type != "power_series" or np.any(inp.am)
        or inp.gamma != 0 or inp.bloat != 1 or inp.spres_ped != 1):
    raise ValueError("Start from the vacuum input.trial at its original flux and length scale")
inp = replace(inp, am=np.array([1., -1.]), ac=np.zeros(21), pres_scale=1.)
inp = replace(inp, ns_array=np.array([NS]), ftol_array=np.array([FTOL]),
              niter_array=np.array([NITER]), delt=.5).change_resolution(
                  mpol=MPOL, ntor=NTOR, ntheta=16, nzeta=14)
vacuum = vj.read_wout(INPUT_FILE.with_name("wout_trial.nc"))
calibration = opt.solve_equilibrium(inp, initial_state=restart_state(vacuum, inp),
                                    raise_on_max_iterations=True)
measured_beta = float(calibration.wout.betatotal)
if not np.isfinite(measured_beta) or measured_beta <= 0:
    raise ValueError("Pressure calibration must yield finite positive beta")
inp = replace(inp, pres_scale=TARGET_BETA/measured_beta)
seed = opt.solve_equilibrium(inp, initial_state=calibration.state, raise_on_max_iterations=True)
mapped = MappedPwOResidual(SURFACES, mboz=8, nboz=8, oversample=2)
# Freeze only an automatic initial cost scale; field normalization stays live.
initial_scale = float(mapped.total_state(seed.state, seed.runtime)) or 1.
actual_surfaces = np.asarray(mapped.compute_state(seed.state, seed.runtime)["s_b"])
print(f"[setup] initial pwO cost {initial_scale:.6g}; Boozer surfaces {actual_surfaces}")
problem = opt.VmecProblem.from_tuples(
    inp, [(objective, 0., 1.)], max_mode=2, vary_major_radius=False,
    use_ess=True, ess_alpha=1.2, restart_from=seed, forward_ftol=FTOL,
    forward_max_iterations=NITER, refine_tol=FTOL, adjoint_tol=1e-8, max_fsq_ratio=1.)
# JIT shares the Boozer transform between the residual and independent field checks.
diagnostics = jax.jit(lambda state, runtime: (objective(state, runtime),
                                             mapped.diagnostics_state(state, runtime)))
monitor = FeasibleCandidateMonitor(problem, inspect_candidate, OUTPUT_DIR,
    maxiter=MAX_ITERATIONS, optimizer_ftol=OPTIMIZER_FTOL, particle_qualified=False,
    input=str(INPUT_FILE), initial_scale=initial_scale, mirror_limit=MIRROR_LIMIT,
    requested_surfaces=np.asarray(SURFACES).tolist(), actual_surfaces=actual_surfaces.tolist(), beta_target=TARGET_BETA, seed_perturbation_m=None,
    settings=dict(ns=NS, mpol=MPOL, ntor=NTOR, ftol=FTOL, niter=NITER))
with monitor:
    x = problem.x0.copy()
    start = x.copy()
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
        before = screen_transport(vj.read_wout(OUTPUT_DIR / "wout_seed.nc"), actual_surfaces)
        after = screen_transport(vj.read_wout(OUTPUT_DIR / "wout_trial.nc"), actual_surfaces)
        monitor.save(baseline_Q=before.tolist(), trial_Q=after.tolist(),
                     transport_screen_passed=transport_improves(before, after))
        if not monitor.report["transport_screen_passed"]:
            transport_backtrack(monitor, start, x, before, actual_surfaces, seed=seed)
        print(f"NEO mean epsilon_eff^(3/2): {before.mean():.6g} -> "
              f"{np.mean(monitor.report['trial_Q']):.6g}")
        for path in vj.plot_wout(OUTPUT_DIR / "wout_trial.nc", OUTPUT_DIR).values():
            print(f"Wrote {path}")
    monitor.save(status="completed")
    print(f"Wrote {OUTPUT_DIR / 'summary.json'}. Transport screen passed: "
          f"{monitor.report['transport_screen_passed']}. Particle tracing is a separate check.")

if not ci_smoke and not monitor.report["transport_screen_passed"]:
    raise SystemExit("Transport did not improve; keep the input equilibrium")
