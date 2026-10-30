"""Short continuation of the bundled non-QS low-transport equilibrium.

The mapped-field residual is a search proxy, not a pwO or confinement test.
Geometry residuals and hard full-surface field checks accompany it. NEO screens the
proposal after optimization; fresh particle tracing is required before promotion.
Run with vmex[neoclassical]. The bundled input is never overwritten.
"""
from pathlib import Path
from dataclasses import replace
import argparse
import json
import os
import time

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.omnigenity import boozer_spectrum_state
from vmex.core.pwo import mapped_pwo_residual_from_boozer
from vmex.core.statephysics import _field_chain, _iotas_half

vj.configure()

INPUT = Path(__file__).resolve().parents[1] / "data" / "input.low_transport"
SURFACES = (.25, 16.5 / 34, .75)
# Fixed calibration from the original NFP4 seed; do not reset between batches.
B_REF = jnp.array([2.7123510440502594, 2.699696279828112, 2.6832357706580665])
B_MAX_REF = jnp.array([2.9284215849767965, 2.979692353278536, 3.008858221344288])
RAW_SCALE = .004813522516264787
MAX_ITERATIONS, NITER = 10, 2000
FORWARD_FTOL, ROOT_TOL = 1e-12, 1e-12


def geometry_residuals(state, runtime):
    half = _iotas_half(state, runtime)
    full = jnp.concatenate((jnp.array([1.5*half[1]-.5*half[2]]),
                            .5*(half[1:-1]+half[2:]),
                            jnp.array([1.5*half[-1]-.5*half[-2]])))
    return jnp.concatenate((jnp.atleast_1d((opt.aspect_ratio(state, runtime)-10)/.1),
                            jnp.sqrt(10/full.size)*jnp.maximum((.42-full)/.01, 0)))


def field_terms(state, runtime):
    spectrum = boozer_spectrum_state(state, runtime, surfaces=SURFACES,
                                    mboz=8, nboz=8, oversample=2)
    mapped = mapped_pwo_residual_from_boozer(
        **{k: spectrum[k] for k in ("bmnc_b", "bmns_b", "xm_b", "xn_b", "iota_b")},
        nfp=4, b_ref=B_REF, a=.3, b=0., c=-2., eta_min=2.8, ngamma=32, neta=32)
    theta = 2*jnp.pi*jnp.arange(64)[:, None, None]/64
    zeta = 2*jnp.pi*jnp.arange(64)[None, :, None]/(4*64)
    phase = theta*spectrum["xm_b"]-zeta*spectrum["xn_b"]
    field = jnp.sum(spectrum["bmnc_b"][:, None, None, :]*jnp.cos(phase)
                    + spectrum["bmns_b"][:, None, None, :]*jnp.sin(phase), axis=-1)
    low, high = field.min(axis=(1, 2)), field.max(axis=(1, 2))
    return mapped, low, high, (high-low)/(high+low)


def residuals(state, runtime):
    mapped = field_terms(state, runtime)[0]
    return jnp.sqrt(2)*jnp.concatenate((
        mapped["residuals1d"]/jnp.sqrt(RAW_SCALE), geometry_residuals(state, runtime)))


def physical_metrics(eq):
    mapped, low, high, mirror = field_terms(eq.state, eq.runtime)
    _, jacobian, _, field, _ = _field_chain(eq.state, eq.runtime)
    signed_jacobian = np.asarray(jacobian.sqrt_g)[1:]*float(eq.runtime.setup.signgs)
    pressure = np.asarray(field.pressure)
    pressure = pressure[:, None, None] if pressure.ndim == 1 else pressure
    B2 = np.asarray(2*(field.total_pressure-pressure))[1:]
    w = eq.wout
    forces = [float(getattr(eq.result, k)) for k in ("fsqr", "fsqz", "fsql")]
    row = dict(aspect=float(w.aspect), iota_min=float(np.min(w.iotaf)),
               force_errors=forces, raw=float(mapped["total"]),
               Bmin=np.asarray(low).tolist(), Bmax=np.asarray(high).tolist(),
               mirror=np.asarray(mirror).tolist())
    row["feasible"] = bool(
        np.isfinite(forces).all() and max(forces) <= ROOT_TOL
        and 9.9 <= row["aspect"] <= 10.1 and row["iota_min"] > .41
        and np.isfinite(B2).all() and B2.min() > 0
        and np.isfinite(signed_jacobian).all() and signed_jacobian.min() > 0
        and np.asarray(mapped["valid"]).all()
        and np.all(np.asarray(high) <= 1.10*np.asarray(B_MAX_REF))
        and np.all(np.asarray(mirror) >= .05))
    return row


def transport(wout):
    """Matched forward NEO screen; not a converged transport calculation."""
    from booz_xform_jax import Booz_xform
    from neo_jax import NeoConfig, run_neo
    from vmex.core.neoclassical import _neo_boozer_data

    bx = Booz_xform(verbose=0, mboz=24, nboz=24)
    bx.read_wout_data(wout)
    indices = [int(np.argmin(abs(np.asarray(bx.s_in)-s))) for s in SURFACES]
    if not np.allclose(np.asarray(bx.s_in)[indices], SURFACES, rtol=0, atol=1e-12):
        raise ValueError("The NEO screen requires the fixed NS35 half-mesh radii")
    bx.compute_surfs = indices
    bx.run(jit=True)
    config = NeoConfig(theta_n=192, phi_n=192, npart=80, multra=2, nstep_per=80,
                       nstep_min=800, nstep_max=2000, acc_req=.005, ref_swi=2,
                       max_rational_field_periods=100000, rational_surface_policy="error")
    results = run_neo(_neo_boozer_data(bx, indices), config=config, use_jax=True,
                      progress=False, jax_surface_scan=False)
    values = []
    for result in results:
        diag, value = result.diagnostics, float(result.epsilon_effective)
        if (not np.isfinite(value) or value <= 0 or not diag.get("rational_correction_valid")
                or diag.get("approximation_used") or diag.get("rational_work_limit_exceeded")):
            raise ValueError(f"Invalid NEO screen at s={float(result.s):.6g}, Q={value}; keep the incumbent")
        values.append(value)
    return np.asarray(values)


def transport_improves(baseline, trial):
    """Require a resolved mean gain and no radial tradeoff in the coarse screen."""
    a, b = np.asarray(baseline), np.asarray(trial)
    return bool(a.shape == b.shape == (3,) and np.isfinite(a).all()
                and np.isfinite(b).all() and np.all(a > 0) and np.all(b > 0)
                and np.all(b <= a) and b.mean() < .99*a.mean())


def main(input_file=INPUT, outdir=Path("low_transport"), maxiter=MAX_ITERATIONS):
    if not 1 <= maxiter <= 30:
        raise ValueError("maxiter must be between 1 and 30")
    start = time.monotonic()
    outdir = Path(outdir)
    if outdir.resolve() == INPUT.parent.resolve():
        raise ValueError("Write proposals outside the bundled input directory")
    if (outdir/"summary.json").exists():
        raise ValueError("Use a new output directory for each batch")
    outdir.mkdir(parents=True, exist_ok=True)
    inp = vj.VmecInput.from_file(input_file)
    if (inp.nfp != 4 or inp.lasym or inp.lfreeb or inp.phiedge != .083
            or inp.rbc[inp.ntor, 0] != 1 or inp.curtor != 0
            or inp.ncurr != 1 or inp.pmass_type != "power_series" or np.any(inp.am != 0)):
        raise ValueError("This fixed calibration requires the bundled NFP4 vacuum configuration")
    inp = replace(inp, ns_array=np.array([35]), ftol_array=np.array([FORWARD_FTOL]),
                  niter_array=np.array([NITER]), delt=.5).change_resolution(
                      mpol=5, ntor=5, ntheta=16, nzeta=14)
    seed = opt.solve_equilibrium(inp, raise_on_max_iterations=True)
    problem = opt.VmecProblem.from_tuples(
        inp, [(residuals, 0., 1.)], max_mode=2, vary_major_radius=False,
        use_ess=True, ess_alpha=1.2, restart_from=seed, forward_ftol=FORWARD_FTOL,
        forward_max_iterations=NITER, refine_tol=ROOT_TOL, adjoint_tol=1e-8, max_fsq_ratio=1.)
    x0 = problem.x0.copy()
    step = .02*np.asarray(problem.scales)
    history, best, cached = [], None, {}
    report = dict(status="running", iterations=[], particle_qualified=False,
                  optimizer_ftol=1e-3, maxiter=maxiter, input=str(input_file))

    def save():
        report.update(evaluations=history, seconds=time.monotonic()-start)
        path = outdir/"summary.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        temporary.replace(path)

    def evaluate(x):
        nonlocal best
        key = np.asarray(x).tobytes()
        if key in cached:
            return cached[key]
        r = np.asarray(problem.residual(x))
        eq = problem.equilibrium_from_x(x)
        if not np.allclose(r, np.asarray(residuals(eq.state, eq.runtime)), rtol=1e-8, atol=1e-10):
            raise RuntimeError("Exported equilibrium does not reproduce the optimized residual")
        row = dict(evaluation=len(history), cost=.5*float(r@r), **physical_metrics(eq))
        history.append(row)
        if row["feasible"] and (best is None or row["raw"] < best[0]):
            best = (row["raw"], np.asarray(x).copy())
            problem.input_from_x(x).to_indata(outdir/"input.trial")
            vj.write_wout(outdir/"wout_trial.nc", eq.wout)
            report["selected_evaluation"] = row["evaluation"]
        cached[key] = r
        save()
        print(f"evaluation {row['evaluation']}: proxy={row['raw']:.6g}, feasible={row['feasible']}", flush=True)
        return r

    def callback(intermediate_result):
        report["iterations"].append(dict(iteration=int(intermediate_result.nit),
                                          nfev=int(intermediate_result.nfev), cost=float(intermediate_result.fun@intermediate_result.fun/2)))
        save()
        if intermediate_result.nit >= maxiter:
            raise StopIteration

    evaluate(x0)
    if best is None:
        raise RuntimeError("The incumbent failed physical checks")
    baseline = problem.equilibrium_from_x(x0).wout
    vj.write_wout(outdir/"wout_incumbent.nc", baseline)
    try:
        result = least_squares(evaluate, x0, jac=problem.residual_jac, x_scale=step,
                               bounds=(x0-.25*step, x0+.25*step), max_nfev=5*maxiter+5,
                               ftol=1e-3, xtol=1e-10, gtol=1e-7, callback=callback)
        evaluate(result.x)
        before = transport(vj.read_wout(outdir/"wout_incumbent.nc"))
        after = transport(vj.read_wout(outdir/"wout_trial.nc"))
        passed = transport_improves(before, after)
        report.update(status="completed", stopping_reason=str(result.message), nfev=int(result.nfev),
                      baseline_Q=before.tolist(), trial_Q=after.tolist(), transport_screen_passed=passed,
                      decision="Particle qualification required" if passed else "Keep incumbent")
    except Exception as error:
        report.update(status="failed", error=str(error), decision="Keep incumbent")
        save()
        raise
    save()
    print(f"NEO mean Q: {before.mean():.6g} -> {after.mean():.6g}; {report['decision']}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--outdir", type=Path, default=Path("low_transport"))
    parser.add_argument("--maxiter", type=int, default=1 if os.getenv("VMEX_EXAMPLES_CI") == "1" else MAX_ITERATIONS)
    args = parser.parse_args()
    main(args.input, args.outdir, args.maxiter)
