#!/usr/bin/env python
"""Stage-1 vacuum fitting to canonical or squeezed-mapping pwO constructions.

Set VMEX_PWO_FAMILY=canonical or mapped. The fixed target/map is a shaping
objective, not a certificate of branchwise action invariance or alpha retention.
The canonical periodic completion is documented by vmex.core.pwo; this example
starts from the bundled QI seed and is not a published-equilibrium reproduction.

The scalar and least-squares methods minimize the same residuals. Scalar is the
low-memory default. A final finer equilibrium and angular grid rescore both the
original seed and the optimized boundary at the same physical surface s=0.5.
"""

import json
import os
from dataclasses import replace
from pathlib import Path
import time

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares, minimize

import vmex as vj
from vmex import optimize as opt
from vmex.core.pwo import CanonicalPwOResidual, MappedPwOResidual
from vmex.core.restart import restart_state

INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "input.QI_nfp2_initial"
FAMILY = os.environ.get("VMEX_PWO_FAMILY", "canonical")
METHOD = os.environ.get("VMEX_PWO_METHOD", "scalar")
SURFACES = [0.5]
MAX_MODES, MAXITER = [1, 2, 3], [8, 8, 12]
STAGE_NS, FINAL_NS = 16, 32  # even ns puts s=0.5 on both half meshes
STAGE_FTOL, FINAL_FTOL = 1e-12, 1e-13
NITER = 12000
MPOL, FINAL_MPOL = 5, 7
PWO_OPTIONS = dict(mboz=10, nboz=10, oversample=2)
VALIDATION_OPTIONS = dict(mboz=14, nboz=14, oversample=3)
GRID, VALIDATION_GRID = 24, 39
IOTA_FLOOR, MIRROR_FLOOR, MIRROR_LIMIT, ELONGATION_LIMIT = 0.40, 0.05, 0.35, 8.0
ASPECT_TOLERANCE, SIZE_TOLERANCE, FIELD_TOLERANCE = 0.03, 0.03, 0.05
BETA_TOLERANCE = 0.10
PARAMETER_STEP, PARAMETER_BOUND = 0.01, 3.0
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODES, MAXITER = [1], [1]
    PWO_OPTIONS = dict(mboz=4, nboz=4, oversample=1)
    VALIDATION_OPTIONS = dict(mboz=5, nboz=5, oversample=2)
    GRID, VALIDATION_GRID = 12, 17
    FINAL_NS, FINAL_MPOL, FINAL_FTOL = 16, 5, 1e-12



def make_target(surfaces, *, family, bmin, bmax, b_ref, options, grid, iota_sign=1.0):
    """Build the fixed physical target shared by the stage-1 and coil examples."""
    if iota_sign not in (-1.0, 1.0):
        raise ValueError("iota_sign must be fixed at +1 or -1 from the seed")
    if family == "canonical":
        return CanonicalPwOResidual(surfaces, bmin=bmin, bmax=bmax, t1=-0.3 * iota_sign,
                                    t2=3.0 * iota_sign, p=2, ntheta=grid, nzeta=grid, **options)
    if family == "mapped":
        return MappedPwOResidual(surfaces, b_ref=b_ref, a=0.3, b=0.0, c=-2.0,
                                 eta_min=2.8, ngamma=grid, neta=grid, **options)
    raise ValueError("family must be canonical or mapped")


def run(*, beta_targets=(0.0,), output_name="pwO_optimized"):
    """Optimize with fixed flux, R00, linear edge-zero pressure and zero current.

    beta_targets is a pressure-continuation ladder in volume-average beta.
    Its pressure scale is estimated from a unit-pressure solve of the original
    seed, then held fixed during each optimization rung. The actual beta is
    constrained and measured; no bootstrap-current closure is implied.
    Boundary derivatives use VMEX's fixed-point convention with its released
    m=1 gauge held fixed; cold re-solve differences require a separate audit.
    """
    if FAMILY not in ("canonical", "mapped") or METHOD not in ("scalar", "least_squares"):
        raise ValueError("require canonical|mapped family and scalar|least_squares method")
    beta_targets = tuple(float(beta) for beta in beta_targets)
    if not beta_targets or beta_targets[0] != 0 or any(
            not np.isfinite(beta) or beta < 0 for beta in beta_targets) or any(
                right < left for left, right in zip(beta_targets, beta_targets[1:])):
        raise ValueError("beta_targets must start at vacuum and increase through finite nonnegative values")
    started = time.perf_counter()
    am = np.zeros(21); am[:2] = [1.0, -1.0]
    seed_input = replace(
        vj.VmecInput.from_file(INPUT_FILE), pres_scale=0.0, pmass_type="power_series", am=am,
        ncurr=1, curtor=0.0, pcurr_type="power_series", ac=np.zeros(21), delt=0.5,
        ns_array=np.array([STAGE_NS]), ftol_array=np.array([STAGE_FTOL]),
        niter_array=np.array([NITER])).change_resolution(
            mpol=MPOL, ntor=MPOL, ntheta=2 * MPOL + 6, nzeta=2 * MPOL + 4)
    inp = seed_input
    equilibrium = opt.solve_equilibrium(inp, raise_on_max_iterations=True)
    seed_equilibrium = equilibrium
    aspect_ref = float(equilibrium.wout.aspect)
    radius_ref = float(equilibrium.wout.Aminor_p)
    field_ref = float(equilibrium.wout.volavgB)
    seed_iota = float(opt.mean_iota(equilibrium.state, equilibrium.runtime))
    if not np.isfinite(seed_iota) or seed_iota == 0:
        raise ValueError("the seed must have finite nonzero rotational transform")
    iota_sign = float(np.sign(seed_iota))
    # Calibrate physical target bounds once, never from an optimizer trial.
    probe = CanonicalPwOResidual(SURFACES, bmin=1.0, bmax=2.0,
                                 t1=-0.3 * iota_sign, t2=3.0 * iota_sign, ntheta=GRID, nzeta=GRID, **PWO_OPTIONS)
    sampled = np.asarray(probe.compute_state(equilibrium.state, equilibrium.runtime)["B"])
    bmin, bmax, b_ref = float(sampled.min()), float(sampled.max()), float(sampled.mean())

    def construction(options, grid, surfaces=SURFACES):
        return make_target(surfaces, family=FAMILY, bmin=bmin, bmax=bmax,
                           b_ref=b_ref, options=options, grid=grid, iota_sign=iota_sign)

    target = construction(PWO_OPTIONS, GRID)
    field_key = "B" if FAMILY == "canonical" else "B_mapped"
    initial_target = target.compute_state(equilibrium.state, equilibrium.runtime)
    surface_field_ref = np.asarray(initial_target[field_key]).mean(axis=(1, 2))

    def physics_rows(state, runtime):
        # A single live transform supplies the fit and local anti-collapse controls.
        out = target.compute_state(state, runtime)
        B = out[field_key]
        local_mean = jnp.mean(B, axis=(1, 2))
        local_mirror = (jnp.max(B, axis=(1, 2)) - jnp.min(B, axis=(1, 2))) / (
            jnp.max(B, axis=(1, 2)) + jnp.min(B, axis=(1, 2)))
        return jnp.concatenate((out["residuals1d"],
            (local_mean / jnp.asarray(surface_field_ref) - 1) / FIELD_TOLERANCE,
            jnp.maximum(MIRROR_FLOOR - local_mirror, 0) / MIRROR_FLOOR))

    unit_beta = None
    if beta_targets[-1] > 0:
        unit = opt.solve_equilibrium(replace(seed_input, pres_scale=1.0),
                                     initial_state=equilibrium.solution, raise_on_max_iterations=True)
        unit_beta = float(unit.wout.betatotal)
        if not np.isfinite(unit_beta) or unit_beta <= 0:
            raise RuntimeError("unit-pressure calibration did not produce positive finite beta")
    monitor = opt.OptimizationMonitor()
    stages = []
    for beta_target in beta_targets:
        inp = replace(inp, pres_scale=0.0 if beta_target == 0 else beta_target / unit_beta)

        def constraints(state, runtime):
            mirror = opt.mirror_ratio(state, runtime)
            rows = [
                (opt.aspect_ratio(state, runtime) / aspect_ref - 1) / ASPECT_TOLERANCE,
                jnp.maximum(IOTA_FLOOR - opt.min_abs_iota(state, runtime), 0) / IOTA_FLOOR,
                jnp.maximum(IOTA_FLOOR - iota_sign * opt.mean_iota(state, runtime), 0) / IOTA_FLOOR,
                jnp.maximum(MIRROR_FLOOR - mirror, 0) / MIRROR_FLOOR,
                jnp.maximum(mirror - MIRROR_LIMIT, 0) / MIRROR_LIMIT,
                jnp.maximum(opt.max_elongation(state, runtime) - ELONGATION_LIMIT, 0)
                / ELONGATION_LIMIT]
            if beta_target > 0:
                rows.append((opt.volume_average_beta(state, runtime) / beta_target - 1)
                            / BETA_TOLERANCE)
            return jnp.stack(rows)

        terms = [(physics_rows, 0.0, 10.0), (constraints, 0.0, 10.0)]

        def loss(state, runtime):
            rows = opt.residuals_from_tuples(state, runtime, terms)
            return 0.5 * jnp.vdot(rows, rows)

        factory = opt.VmecProblem.from_loss if METHOD == "scalar" else opt.VmecProblem.from_tuples
        problem = factory(
            inp, loss if METHOD == "scalar" else terms, max_mode=max(MAX_MODES),
            vary_major_radius=False, restart_from=equilibrium, max_fsq_ratio=1.0,
            adjoint_tol=1e-8, progress=not ci_smoke, evaluation_progress=not ci_smoke,
            jacobian_batch_size=1)
        full_x = problem.x0.copy()
        for max_mode, budget in zip(MAX_MODES, MAXITER):
            stage = problem.subproblem(max_mode=max_mode, x=full_x)
            monitor.problem = stage
            x0, step = stage.x0, PARAMETER_STEP * stage.scales
            if METHOD == "scalar":
                def value_and_gradient(y):
                    value, gradient = stage.value_and_grad(x0 + step * y)
                    monitor.cache_evaluation(x0 + step * y, value, gradient)
                    return value, step * gradient

                def record(intermediate_result):
                    monitor({"x": x0 + step * intermediate_result.x, "fun": intermediate_result.fun})

                initial = float(value_and_gradient(np.zeros_like(x0))[0])
                result = minimize(
                    value_and_gradient, np.zeros_like(x0), jac=True, method="L-BFGS-B",
                    bounds=[(-PARAMETER_BOUND, PARAMETER_BOUND)] * x0.size, callback=record,
                    options={"maxiter": budget, "maxfun": 5 * budget + 5, "maxls": 15,
                             "ftol": 1e-12, "gtol": 1e-7})
                x, final = x0 + step * result.x, float(result.fun)
            else:
                bounds = (x0 - PARAMETER_BOUND * step, x0 + PARAMETER_BOUND * step)
                initial = float(stage.fun(x0))
                result = least_squares(stage.residual, x0, jac=stage.residual_jac,
                                       bounds=bounds, x_scale=stage.scales, max_nfev=budget + 1,
                                       ftol=1e-7, xtol=1e-10, callback=monitor)
                x, final = result.x, float(result.cost)
            inp = stage.input_from_x(x)
            equilibrium = stage.equilibrium_from_x(x)
            full_x = problem.x_from_input(inp)
            stages.append(dict(beta_target=beta_target, mode=max_mode, initial_cost=initial,
                               final_cost=final, status=int(result.status), message=str(result.message)))
            print(f"beta {beta_target:.3%}, mode {max_mode}: cost {initial:.6e} -> {final:.6e}", flush=True)

    def refined(deck, source):
        deck = replace(deck, ns_array=np.array([FINAL_NS]), ftol_array=np.array([FINAL_FTOL]),
                       niter_array=np.array([NITER])).change_resolution(
                           mpol=FINAL_MPOL, ntor=FINAL_MPOL,
                           ntheta=2 * FINAL_MPOL + 10, nzeta=2 * FINAL_MPOL + 8)
        eq = opt.solve_equilibrium(deck, initial_state=restart_state(source.wout, deck),
                                  raise_on_max_iterations=True, verbose=not ci_smoke)
        return deck, eq

    final_input, final_equilibrium = refined(inp, equilibrium)
    reference_input = replace(seed_input, pres_scale=inp.pres_scale)
    _, reference_equilibrium = refined(reference_input, seed_equilibrium)
    fine = construction(VALIDATION_OPTIONS, VALIDATION_GRID)
    initial_score = fine.compute_state(reference_equilibrium.state, reference_equilibrium.runtime)
    final_score = fine.compute_state(final_equilibrium.state, final_equilibrium.runtime)
    train_score = target.compute_state(final_equilibrium.state, final_equilibrium.runtime)
    metrics = opt.EquilibriumReporter(
        ("aspect", opt.aspect_ratio, ".5f"), ("min_abs_iota", opt.min_abs_iota, ".5f"),
        ("mirror", opt.mirror_ratio, ".5f"), ("elongation", opt.max_elongation, ".5f"),
        ("beta", opt.volume_average_beta, ".4%"))("final", final_equilibrium)
    local_B = np.asarray(final_score[field_key])
    local_mean = local_B.mean(axis=(1, 2))
    local_mirror = (local_B.max(axis=(1, 2)) - local_B.min(axis=(1, 2))) / (
        local_B.max(axis=(1, 2)) + local_B.min(axis=(1, 2)))
    metrics.update(local_mirror=local_mirror.tolist(), local_mean_T=local_mean.tolist(),
                   minor_radius_m=float(final_equilibrium.wout.Aminor_p),
                   volavgB_T=float(final_equilibrium.wout.volavgB),
                   initial_fine=float(initial_score["total"]), fine=float(final_score["total"]),
                   training=float(train_score["total"]))
    checks = {
        "fine-grid objective improved": metrics["fine"] < metrics["initial_fine"],
        "angular refinement within 5%": abs(metrics["fine"] - metrics["training"])
        <= 0.05 * max(abs(metrics["fine"]), 1e-12),
        "valid target/map": bool(np.all(np.asarray(final_score["valid"]))),
        "iota floor": metrics["min_abs_iota"] >= IOTA_FLOOR,
        "training-surface mirror floor": bool(np.all(local_mirror >= MIRROR_FLOOR)),
        "training-surface field held": bool(np.all(abs(local_mean / surface_field_ref - 1) <= FIELD_TOLERANCE)),
        "mirror interval": MIRROR_FLOOR <= metrics["mirror"] <= MIRROR_LIMIT,
        "elongation limit": metrics["elongation"] <= ELONGATION_LIMIT,
        "aspect held": abs(metrics["aspect"] / aspect_ref - 1) <= ASPECT_TOLERANCE,
        "size held": abs(metrics["minor_radius_m"] / radius_ref - 1) <= SIZE_TOLERANCE,
        "field held": abs(metrics["volavgB_T"] / field_ref - 1) <= FIELD_TOLERANCE,
        "converged": bool(np.all(np.asarray(final_equilibrium.result.converged))),
    }
    if beta_targets[-1] > 0:
        checks["beta held"] = abs(metrics["beta"] / beta_targets[-1] - 1) <= BETA_TOLERANCE
    unmet = [name for name, met in checks.items() if not met]
    construction_diagnostics = {
        name: np.asarray(final_score[name]).tolist()
        for name in ("eta_fraction", "mapped_area_fraction", "jacobian_lower_bound", "corner_margin", "width")
        if name in final_score}
    summary = dict(family=FAMILY, method=METHOD, smoke=ci_smoke, seconds=time.perf_counter() - started,
                   stages=stages, metrics=metrics, actual_s=np.asarray(final_score["s_b"]).tolist(),
                   target=dict(p=2, bmin_T=bmin, bmax_T=bmax, b_ref_T=b_ref,
                               surface_field_ref_T=surface_field_ref.tolist(), iota_sign=iota_sign,
                               t1=-0.3 * iota_sign, t2=3.0 * iota_sign,
                               a=0.3, b=0.0, c=-2.0, eta_min=2.8),
                   settings=dict(stage_ns=STAGE_NS, stage_ftol=STAGE_FTOL, mpol=MPOL,
                                 final_ns=FINAL_NS, final_ftol=FINAL_FTOL, final_mpol=FINAL_MPOL,
                                 niter=NITER, grid=GRID, validation_grid=VALIDATION_GRID,
                                 transform=PWO_OPTIONS, validation_transform=VALIDATION_OPTIONS,
                                 mode_ladder=MAX_MODES, iteration_budgets=MAXITER),
                   phiedge_Wb=float(inp.phiedge), pres_scale_Pa=float(inp.pres_scale),
                   pressure="p(s)=PRES_SCALE*(1-s)", current="zero enclosed toroidal current",
                   checks=checks, unmet=unmet, construction_diagnostics=construction_diagnostics,
                   research_validation_complete=False)
    # JSON rejects nonfinite diagnostic values instead of writing a misleading result.
    Path(f"{output_name}_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    final_input.to_indata(f"input.{output_name}")
    vj.write_wout(f"wout_{output_name}.nc", final_equilibrium.wout)
    vj.write_wout(f"wout_{output_name}_seed.nc", reference_equilibrium.wout)
    monitor.save(f"{output_name}_objectives.csv")
    print(json.dumps(metrics, indent=2))
    print("Unmet checks: " + (", ".join(unmet) if unmet else "none"))
    if unmet and not ci_smoke:
        raise SystemExit(1)
    return summary


if __name__ == "__main__":
    run()
