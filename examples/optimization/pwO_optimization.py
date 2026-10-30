"""Optimize low transport from a near-circular vacuum torus at fixed aspect 10.

The mapped-field residual is a search proxy, not a pwO or confinement test.
Geometry residuals and hard full-surface field checks accompany it. NEO screens the
proposal after optimization; fresh particle tracing is required before promotion.
Shape boundary modes 1 then 2; --input continues a supplied configuration.
Optional --beta applies p proportional to 1-s at zero toroidal current.
Run with vmex[neoclassical]. Existing input files are never overwritten.
"""
from pathlib import Path
from dataclasses import replace
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.omnigenity import boozer_spectrum_state
from vmex.core.pwo import mapped_pwo_residual_from_boozer
from vmex.core.statephysics import _field_chain, _iotas_half

INPUT = Path(__file__).resolve().parents[1] / "data" / "input.minimal_seed_nfp4"
SURFACES = (.25, 16.5 / 34, .75)
# Fixed calibration from the original NFP4 seed; do not reset between batches.
B_REF = np.array([2.7123510440502594, 2.699696279828112, 2.6832357706580665])
B_MAX_REF = np.array([2.9284215849767965, 2.979692353278536, 3.008858221344288])
RAW_SCALE = .004813522516264787
MAX_ITERATIONS, NITER = 15, 2000
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


def physical_metrics(eq, fields=None):
    mapped, low, high, mirror = field_terms(eq.state, eq.runtime) if fields is None else fields
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


def near_circular_input():
    """A10 circular torus with a 5%-of-minor-radius helical perturbation.

    The perturbation selects positive transform; an exactly circular vacuum
    torus has zero first-order transform sensitivity. Its initial transform is
    about 0.01, so the transform constraint is part of the actual optimization.
    """
    inp = vj.VmecInput.from_file(INPUT)
    rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
    rbc[inp.ntor, 1] = zbs[inp.ntor, 1] = .1
    rbc[inp.ntor+1, 1], zbs[inp.ntor+1, 1] = -.005, .005
    return replace(inp, rbc=rbc, zbs=zbs)


def main(input_file=None, outdir=Path("low_transport"), maxiter=MAX_ITERATIONS, beta=0.):
    vj.configure()
    if not 1 <= maxiter <= 30:
        raise ValueError("maxiter must be between 1 and 30 per mode stage")
    if not np.isfinite(beta) or not 0 <= beta <= .005:
        raise ValueError("beta must be a volume-average fraction between 0 and .005")
    start = time.monotonic()
    outdir = Path(outdir)
    if outdir.resolve() == INPUT.parent.resolve() or (input_file is not None
            and outdir.resolve() == Path(input_file).resolve().parent):
        raise ValueError("Write proposals outside the source input directory")
    if (outdir/"summary.json").exists():
        raise ValueError("Use a new output directory for each batch")
    if beta and input_file is None:
        # The weak-transform seed did not converge at 0.5% pressure on this
        # solve budget. Shape its vacuum field first, then raise pressure.
        vacuum = main(None, outdir/"vacuum", maxiter, 0.)
        if "selected_evaluation" not in vacuum:
            raise RuntimeError("Vacuum shaping needs a feasible proposal before pressure continuation")
        report = main(outdir/"vacuum"/"input.trial", outdir, maxiter, beta)
        report["vacuum_start"] = str(outdir/"vacuum"/"summary.json")
        report["pressure_continuation_seconds"] = report["seconds"]
        report["vacuum_seconds"] = vacuum["seconds"]
        report["seconds"] = time.monotonic()-start
        (outdir/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        return report
    inp = near_circular_input() if input_file is None else vj.VmecInput.from_file(input_file)
    if (inp.nfp != 4 or inp.lasym or inp.lfreeb or inp.phiedge != .083
            or inp.rbc[inp.ntor, 0] != 1 or inp.curtor != 0
            or inp.ncurr != 1 or inp.pmass_type != "power_series"
            or (beta and (inp.gamma != 0 or inp.bloat != 1 or inp.spres_ped != 1))):
        raise ValueError("Use an NFP4 zero-current vacuum or linear finite-beta input with GAMMA=0, BLOAT=SPRES_PED=1 and the fixed flux and scale")
    pressure = np.asarray(inp.am)
    vacuum = not np.any(pressure != 0)
    linear_pressure = (pressure.size >= 2 and np.array_equal(pressure[:2], [1., -1.])
                       and not np.any(pressure[2:] != 0))
    if not vacuum and not (beta and linear_pressure):
        raise ValueError("Use a vacuum input or the declared finite-beta pressure p proportional to 1-s")
    outdir.mkdir(parents=True, exist_ok=True)
    inp = replace(inp, ns_array=np.array([35]), ftol_array=np.array([FORWARD_FTOL]),
                  niter_array=np.array([NITER]), delt=.5).change_resolution(
                      mpol=5, ntor=5, ntheta=16, nzeta=14)
    if beta:
        if vacuum:
            inp = replace(inp, am=np.array([1., -1.]), ac=np.zeros(21), pres_scale=1.)
        calibration = opt.solve_equilibrium(inp, raise_on_max_iterations=True)
        measured_beta = float(calibration.wout.betatotal)
        if not np.isfinite(measured_beta) or measured_beta <= 0:
            raise ValueError("Pressure calibration must yield finite positive beta")
        inp = replace(inp, pres_scale=inp.pres_scale*beta/measured_beta)
    seed = opt.solve_equilibrium(inp, raise_on_max_iterations=True)

    def objective(state, runtime):
        r = residuals(state, runtime)
        return (jnp.concatenate((r, jnp.atleast_1d(opt.volume_average_beta(state, runtime)/beta-1)))
                if beta else r)

    problem = opt.VmecProblem.from_tuples(
        inp, [(objective, 0., 1.)], max_mode=2, vary_major_radius=False,
        use_ess=True, ess_alpha=1.2, restart_from=seed, forward_ftol=FORWARD_FTOL,
        forward_max_iterations=NITER, refine_tol=ROOT_TOL, adjoint_tol=1e-8, max_fsq_ratio=1.)
    diagnostics = jax.jit(lambda state, runtime: (objective(state, runtime), field_terms(state, runtime)))
    x = problem.x0.copy()
    history, best, cached = [], None, {}
    report = dict(status="running", iterations=[], particle_qualified=False,
                  optimizer_ftol=1e-3, maxiter_per_stage=maxiter,
                  input=str(input_file) if input_file is not None else "near-circular NFP4",
                  beta_target=beta, seed_perturbation_m=.005 if input_file is None else None,
                  normalization="Fixed original NFP4 calibration, shared by every stage",
                  settings=dict(ns=35, mpol=5, ntor=5, ftol=FORWARD_FTOL, niter=NITER))

    def save():
        report.update(evaluations=history, seconds=time.monotonic()-start)
        path = outdir/"summary.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        temporary.replace(path)

    def evaluate(y):
        nonlocal best
        key = np.asarray(y).tobytes()
        if key in cached:
            return cached[key]
        r = np.asarray(problem.residual(y))
        eq = problem.equilibrium_from_x(y)
        recomputed, fields = diagnostics(eq.state, eq.runtime)
        if not np.allclose(r, np.asarray(recomputed), rtol=1e-8, atol=1e-10):
            raise RuntimeError("Exported equilibrium does not reproduce the optimized residual")
        row = dict(evaluation=len(history), cost=.5*float(r@r), **physical_metrics(eq, fields))
        row["beta"] = float(eq.wout.betatotal)
        row["feasible"] &= not beta or .9*beta <= row["beta"] <= 1.1*beta
        history.append(row)
        if row["feasible"] and (best is None or row["raw"] < best[0]):
            best = (row["raw"], np.asarray(y).copy())
            problem.input_from_x(y).to_indata(outdir/"input.trial")
            vj.write_wout(outdir/"wout_trial.nc", eq.wout)
            report["selected_evaluation"] = row["evaluation"]
        cached[key] = r
        save()
        print(f"evaluation {row['evaluation']}: proxy={row['raw']:.6g}, iota={row['iota_min']:.6g}, feasible={row['feasible']}", flush=True)
        return r

    evaluate(x)
    problem.input_from_x(x).to_indata(outdir/"input.seed")
    vj.write_wout(outdir/"wout_seed.nc", problem.equilibrium_from_x(x).wout)
    # An explicit input continues its incumbent; a circular start shapes modes
    # 1 then 2, reusing the same equilibrium and derivative executable shapes.
    modes = [1, 2] if input_file is None else [2]
    try:
        for mode in modes:
            stage = problem.subproblem(max_mode=mode, x=x)
            step = .02*np.asarray(stage.scales)
            stage_iterations = 0

            def callback(intermediate_result):
                nonlocal stage_iterations
                stage_iterations += 1
                report["iterations"].append(dict(mode=mode, iteration=stage_iterations,
                    nfev=int(intermediate_result.nfev), cost=float(intermediate_result.fun@intermediate_result.fun/2)))
                save()
                if stage_iterations >= maxiter:
                    raise StopIteration

            result = least_squares(lambda y: evaluate(stage.embed(y)), stage.x0,
                jac=stage.residual_jac, x_scale=step,
                bounds=(stage.x0-5*step, stage.x0+5*step) if input_file is None else
                       (stage.x0-.25*step, stage.x0+.25*step),
                max_nfev=5*maxiter+5, ftol=1e-3, xtol=1e-10, gtol=1e-7, callback=callback)
            x = stage.embed(result.x)
            evaluate(x)
            problem.input_from_x(x).to_indata(outdir/f"input.mode{mode}")
        report.update(status="completed", stopping_reason=str(result.message), nfev=len(history))
        if best is None:
            report.update(transport_screen_passed=False, decision="No physically feasible proposal; keep the qualified candidate")
        else:
            before = transport(vj.read_wout(outdir/"wout_seed.nc"))
            after = transport(vj.read_wout(outdir/"wout_trial.nc"))
            passed = transport_improves(before, after)
            report.update(baseline_Q=before.tolist(), trial_Q=after.tolist(), transport_screen_passed=passed,
                          decision="Particle qualification required; keep the qualified candidate" if passed else "Keep the qualified candidate")
            print(f"NEO mean Q: {before.mean():.6g} -> {after.mean():.6g}", flush=True)
    except Exception as error:
        report.update(status="failed", error=str(error), decision="Keep the qualified candidate")
        save()
        raise
    save()
    print(report["decision"], flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Continue a saved input instead of the near-circular start")
    parser.add_argument("--outdir", type=Path, default=Path("low_transport"))
    parser.add_argument("--maxiter", type=int, default=1 if os.getenv("VMEX_EXAMPLES_CI") == "1" else MAX_ITERATIONS)
    parser.add_argument("--beta", type=float, default=0., help="Volume-average beta fraction (0 to .005); p proportional to 1-s, zero toroidal current")
    args = parser.parse_args()
    main(args.input, args.outdir, args.maxiter, args.beta)
