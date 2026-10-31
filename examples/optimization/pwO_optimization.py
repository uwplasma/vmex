"""Optimize pwO contour variance from a near-circular vacuum torus.

Aspect ratio, signed iota and mirror limits constrain the shape; NEO
screens the selected equilibrium. Requires vmex[neoclassical]."""
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

NFP = 2
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "input.minimal_seed_nfp4"
OUTPUT_DIR = Path("low_transport")
MAX_MODES, MAX_PARAMETER_CHANGE = [1, 2], 5.
SURFACES = jnp.linspace(.25, .75, 3)
ASPECT_TARGET, IOTA_TARGET = 6., .42
SEED_PERTURBATION = .05/ASPECT_TARGET  # 5% of the minor radius, positive transform
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
    return residual


def inspect_candidate(eq):
    residual, fields = diagnostics(eq.state, eq.runtime)
    metrics = mapped.physical_metrics(eq, mirror_ceiling=MIRROR_LIMIT,
        aspect_bounds=(ASPECT_TARGET-.1, ASPECT_TARGET+.1), iota_floor=.41, fields=fields)
    return np.asarray(residual), metrics


vj.configure()
require_optional("neo-jax", "low-transport NEO screen")
print("[setup] near-circular vacuum torus, boundary modes 1 then 2", flush=True)
inp = vj.VmecInput.from_file(INPUT_FILE)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor, 1] = zbs[inp.ntor, 1] = 1/ASPECT_TARGET
rbc[inp.ntor+1, 1], zbs[inp.ntor+1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
inp = replace(inp, nfp=NFP, rbc=rbc, zbs=zbs, phiedge=.083*(10/ASPECT_TARGET)**2)
inp = replace(inp, ns_array=np.array([NS]), ftol_array=np.array([FTOL]),
              niter_array=np.array([NITER]), delt=.5).change_resolution(
                  mpol=MPOL, ntor=NTOR, ntheta=16, nzeta=14)
seed = opt.solve_equilibrium(inp, raise_on_max_iterations=True)
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
    input=str(INPUT_FILE), nfp=NFP, initial_scale=initial_scale, mirror_limit=MIRROR_LIMIT,
    requested_surfaces=np.asarray(SURFACES).tolist(), actual_surfaces=actual_surfaces.tolist(), beta_target=0., seed_perturbation_m=SEED_PERTURBATION,
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
        before = screen_transport(vj.read_wout(OUTPUT_DIR / "wout_seed.nc"), actual_surfaces)
        after = screen_transport(vj.read_wout(OUTPUT_DIR / "wout_trial.nc"), actual_surfaces)
        monitor.save(baseline_Q=before.tolist(), trial_Q=after.tolist(),
                     transport_screen_passed=transport_improves(before, after))
        print(f"NEO mean epsilon_eff^(3/2): {before.mean():.6g} -> {after.mean():.6g}")
        for path in vj.plot_wout(OUTPUT_DIR / "wout_trial.nc", OUTPUT_DIR).values():
            print(f"Wrote {path}")
    monitor.save(status="completed")
    print(f"Wrote {OUTPUT_DIR / 'summary.json'}. Transport screen passed: "
          f"{monitor.report['transport_screen_passed']}. Particle tracing is a separate check.")

if not ci_smoke and not monitor.report["transport_screen_passed"]:
    raise SystemExit("Transport did not improve; keep the input equilibrium")
