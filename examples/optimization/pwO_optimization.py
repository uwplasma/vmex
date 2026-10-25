#!/usr/bin/env python
"""Fit canonical or mapped pwO targets, then rescore at higher resolution.

VMEX_PWO_SEED=near_circular runs QP conditioning before the fit; the default
uses the bundled QI seed. VMEX_PWO_FAMILY selects canonical or mapped.
Target calibration stays fixed; final validation requires local mirror within
10% of the seed. Scalar and least-squares solvers minimize the same residuals.
A lower target residual does not establish action invariance or confinement.
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

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEED = os.environ.get("VMEX_PWO_SEED", "qi")
QS_INPUT = os.environ.get("VMEX_PWO_QS_INPUT")  # saved QP boundary skips preconditioning
SEED_PERTURBATION = 0.005  # each added Fourier coefficient is 5% of a=0.1 m
QS_MAX_MODES, QS_MAXITER = [1, 2, 3], [20, 20, 20]
QS_ASPECT_TARGET = 6.0
QS_IOTA_TARGET, QS_IOTA_TOLERANCE = 0.42, 0.02
MIRROR_TRAIN_TARGET, MIRROR_TRAIN_TOLERANCE = 0.34, 0.03  # margin for the independent gate
QS_PARAMETER_STEP, QS_PARAMETER_BOUND = 0.02, 5.0
FAMILY = os.environ.get("VMEX_PWO_FAMILY", "canonical")
METHOD = os.environ.get("VMEX_PWO_METHOD", "scalar")
PENALTY = os.environ.get("VMEX_PWO_PENALTY", "interval")
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
MIRROR_RELATIVE_TOLERANCE = 0.10
PARAMETER_STEP, PARAMETER_BOUND = 0.01, 3.0
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    QS_MAX_MODES, QS_MAXITER = [1], [1]
    MAX_MODES, MAXITER = [1], [1]
    PWO_OPTIONS = dict(mboz=4, nboz=4, oversample=1)
    VALIDATION_OPTIONS = dict(mboz=5, nboz=5, oversample=2)
    GRID, VALIDATION_GRID = 12, 17
    FINAL_NS, FINAL_MPOL, FINAL_FTOL = 16, 5, 1e-12


def make_target(surfaces, *, family, bmin, bmax, b_ref, options, grid, iota_sign=1.0, theta_center=0.0, zeta_center=0.0):
    """Build the fixed physical target shared by the stage-1 and coil examples."""
    if iota_sign not in (-1.0, 1.0):
        raise ValueError("iota_sign must be fixed at +1 or -1 from the seed")
    if family == "canonical":
        return CanonicalPwOResidual(surfaces, bmin=bmin, bmax=bmax, t1=-0.3 * iota_sign,
                                    t2=3.0 * iota_sign, theta_center=theta_center, zeta_center=zeta_center, p=2, ntheta=grid, nzeta=grid, **options)
    if family == "mapped":
        return MappedPwOResidual(surfaces, b_ref=b_ref, a=0.3, b=0.0, c=-2.0,
                                 eta_min=2.8, ngamma=grid, neta=grid, **options)
    raise ValueError("family must be canonical or mapped")


def canonical_seed_phase(field, target, nfp):
    """Choose once among the four stellarator-symmetric angular origins."""
    if any(size % 2 for size in target.shape[-2:]):
        raise ValueError("seed phase selection requires even angular grids")
    shifts = [(i * target.shape[-2] // 2, j * target.shape[-1] // 2) for i in range(2) for j in range(2)]
    candidates = [np.roll(target, shift, axis=(-2, -1)) for shift in shifts]
    costs = [float(np.mean(((field - candidate) / candidate)**2)) for candidate in candidates]
    chosen = int(np.argmin(costs))
    return float((chosen // 2) * np.pi), float((chosen % 2) * np.pi / nfp), costs


def hold_ratio(value, reference, tolerance, penalty):
    """Use the same physical tolerance for an equality or interval penalty."""
    deviation = value / reference - 1
    if penalty == "interval":
        deviation = jnp.maximum(jnp.abs(deviation) - tolerance, 0)
    return deviation / tolerance


def save_seed_checkpoint(deck, equilibrium, output_name, label):
    """Save each independently solved boundary before a later stage can fail."""
    deck.to_indata(f"input.{output_name}_{label}")
    vj.write_wout(f"wout_{output_name}_{label}.nc", equilibrium.wout)


def qp_gate_failures(metrics):
    """Check the refined QP boundary before pwO target calibration."""
    failed = []
    if not np.isfinite(metrics["qs_total"]):
        failed.append("QP objective finite")
    if (not np.isfinite(metrics["aspect"]) or
            abs(metrics["aspect"] / QS_ASPECT_TARGET - 1) > ASPECT_TOLERANCE):
        failed.append("aspect target")
    if not np.isfinite(metrics["min_abs_iota"]) or metrics["min_abs_iota"] < IOTA_FLOOR:
        failed.append("iota floor")
    if not MIRROR_FLOOR <= metrics["mirror"] <= MIRROR_LIMIT:
        failed.append("mirror interval")
    if not np.isfinite(metrics["elongation"]) or metrics["elongation"] > ELONGATION_LIMIT:
        failed.append("elongation limit")
    return failed


def run(*, beta_targets=(0.0,), output_name="pwO_optimized"):
    """Optimize with fixed flux, R00, linear edge-zero pressure and zero current.

    beta_targets is a pressure-continuation ladder in volume-average beta.
    Pressure is calibrated from a unit-pressure seed solve and held fixed per
    stage. Current is prescribed, not bootstrap-consistent. Derivatives hold
    the solver's released m=1 gauge fixed.
    """
    if (FAMILY not in ("canonical", "mapped") or METHOD not in ("scalar", "least_squares")
            or PENALTY not in ("equality", "interval")):
        raise ValueError("require canonical|mapped family, scalar|least_squares method, and equality|interval penalty")
    beta_targets = tuple(float(beta) for beta in beta_targets)
    if not beta_targets or beta_targets[0] != 0 or any(
            not np.isfinite(beta) or beta < 0 for beta in beta_targets) or any(
                right < left for left, right in zip(beta_targets, beta_targets[1:])):
        raise ValueError("beta_targets must start at vacuum and increase through finite nonnegative values")
    if SEED not in ("qi", "near_circular"):
        raise ValueError("VMEX_PWO_SEED must be qi or near_circular")
    seed_route = "near_circular" if QS_INPUT else SEED
    print(f"Starting {seed_route} -> {FAMILY} pwO ({METHOD}); beta ladder {beta_targets}", flush=True)
    started = time.perf_counter()
    am = np.zeros(21); am[:2] = [1.0, -1.0]

    def prepare(deck):
        return replace(
            deck, pres_scale=0.0, pmass_type="power_series", am=am,
            ncurr=1, curtor=0.0, pcurr_type="power_series", ac=np.zeros(21), delt=0.5,
            ns_array=np.array([STAGE_NS]), ftol_array=np.array([STAGE_FTOL]),
            niter_array=np.array([NITER])).change_resolution(
                mpol=MPOL, ntor=MPOL, ntheta=2 * MPOL + 6, nzeta=2 * MPOL + 4)

    def refined(deck, source):
        deck = replace(deck, ns_array=np.array([FINAL_NS]), ftol_array=np.array([FINAL_FTOL]),
                       niter_array=np.array([NITER])).change_resolution(
                           mpol=FINAL_MPOL, ntor=FINAL_MPOL,
                           ntheta=2 * FINAL_MPOL + 10, nzeta=2 * FINAL_MPOL + 8)
        eq = opt.solve_equilibrium(deck, initial_state=restart_state(source.wout, deck),
                                  raise_on_max_iterations=True, verbose=not ci_smoke)
        return deck, eq

    qs_stages, qs_metrics = [], None
    if SEED == "qi" and not QS_INPUT:
        seed_input = prepare(vj.VmecInput.from_file(DATA_DIR / "input.QI_nfp2_initial"))
        equilibrium = opt.solve_equilibrium(seed_input, raise_on_max_iterations=True)
    else:
        near_circular_input = vj.VmecInput.from_file(DATA_DIR / "input.minimal_seed_nfp2")
        rbc, zbs = near_circular_input.rbc.copy(), near_circular_input.zbs.copy()
        rbc[near_circular_input.ntor - 1, 1] = -SEED_PERTURBATION
        zbs[near_circular_input.ntor - 1, 1] = SEED_PERTURBATION
        near_circular_input = replace(near_circular_input, rbc=rbc, zbs=zbs)
        near_circular_input = prepare(near_circular_input)
        qs = opt.QuasisymmetryRatioResidual([0.5, 0.7, 0.9], helicity_m=0, helicity_n=1)
        qs_raw_metrics = None

        def qp_metrics(eq):
            return dict(qs_total=float(qs.total_state(eq.state, eq.runtime)),
                        aspect=float(opt.aspect_ratio(eq.state, eq.runtime)),
                        min_abs_iota=float(opt.min_abs_iota(eq.state, eq.runtime)),
                        mirror=float(opt.mirror_ratio(eq.state, eq.runtime)),
                        elongation=float(opt.max_elongation(eq.state, eq.runtime)),
                        minor_radius_m=float(eq.wout.Aminor_p), volavgB_T=float(eq.wout.volavgB))

        def save_qs_summary(training_metrics=None, refined_metrics=None):
            unmet = None if refined_metrics is None else qp_gate_failures(refined_metrics)
            status = "running" if unmet is None else ("failed" if unmet else "passed")
            Path(f"{output_name}_qs_summary.json").write_text(json.dumps(
                dict(source=QS_INPUT or "near_circular", perturbation_m=SEED_PERTURBATION,
                     helicity_m=0, helicity_n=1, stages=qs_stages,
                     settings=dict(aspect_target=QS_ASPECT_TARGET, iota_target=QS_IOTA_TARGET,
                                   mirror_target=MIRROR_TRAIN_TARGET, iota_tolerance=QS_IOTA_TOLERANCE,
                                   mirror_tolerance=MIRROR_TRAIN_TOLERANCE,
                                   parameter_step=QS_PARAMETER_STEP, parameter_bound=QS_PARAMETER_BOUND),
                     raw_metrics=qs_raw_metrics, training_metrics=training_metrics,
                     metrics=refined_metrics, status=status, unmet=unmet), indent=2, allow_nan=False) + "\n")

        if QS_INPUT:
            seed_input = prepare(vj.VmecInput.from_file(Path(QS_INPUT)))
            if (seed_input.nfp != near_circular_input.nfp or
                    not np.isclose(seed_input.phiedge, near_circular_input.phiedge) or
                    not np.isclose(seed_input.rbc[seed_input.ntor, 0],
                                   near_circular_input.rbc[near_circular_input.ntor, 0])):
                raise ValueError("QP input must retain nfp, toroidal flux and major radius")
            equilibrium = opt.solve_equilibrium(seed_input, raise_on_max_iterations=True)
        else:
            seed_input = near_circular_input
            equilibrium = opt.solve_equilibrium(seed_input, raise_on_max_iterations=True)
            save_seed_checkpoint(seed_input, equilibrium, output_name, "near_circular")
            qs_raw_metrics = qp_metrics(equilibrium)
            save_qs_summary()

            def qs_constraints(state, runtime):
                mirror = opt.mirror_ratio(state, runtime)
                return jnp.stack([
                    (opt.aspect_ratio(state, runtime) / QS_ASPECT_TARGET - 1) / ASPECT_TOLERANCE,
                    jnp.maximum(QS_IOTA_TARGET - opt.min_abs_iota(state, runtime), 0) / QS_IOTA_TOLERANCE,
                    jnp.maximum(MIRROR_FLOOR - mirror, 0) / MIRROR_TRAIN_TOLERANCE,
                    jnp.maximum(mirror - MIRROR_TRAIN_TARGET, 0) / MIRROR_TRAIN_TOLERANCE,
                    jnp.maximum(opt.max_elongation(state, runtime) - ELONGATION_LIMIT, 0)
                    / ELONGATION_LIMIT])

            qs_terms = [(qs, 0.0, 1.0), (qs_constraints, 0.0, 10.0)]

            def qs_loss(state, runtime):
                rows = opt.residuals_from_tuples(state, runtime, qs_terms)
                return 0.5 * jnp.vdot(rows, rows)

            qs_problem = opt.VmecProblem.from_loss(
                seed_input, qs_loss, max_mode=max(QS_MAX_MODES), vary_major_radius=False,
                restart_from=equilibrium, max_fsq_ratio=1.0, adjoint_tol=1e-8,
                progress=not ci_smoke, evaluation_progress=not ci_smoke, jacobian_batch_size=1)
            full_x = qs_problem.x0.copy()
            for max_mode, budget in zip(QS_MAX_MODES, QS_MAXITER):
                stage = qs_problem.subproblem(max_mode=max_mode, x=full_x)
                x0, step = stage.x0, QS_PARAMETER_STEP * stage.scales

                def qs_value_and_gradient(y):
                    value, gradient = stage.value_and_grad(x0 + step * y)
                    return value, step * gradient

                def qs_record(intermediate_result):
                    print(f"QP mode {max_mode}: accepted cost {intermediate_result.fun:.6e}", flush=True)

                initial = float(qs_value_and_gradient(np.zeros_like(x0))[0])
                result = minimize(
                    qs_value_and_gradient, np.zeros_like(x0), jac=True, method="L-BFGS-B",
                    bounds=[(-QS_PARAMETER_BOUND, QS_PARAMETER_BOUND)] * x0.size, callback=qs_record,
                    options={"maxiter": budget, "maxfun": 5 * budget + 5, "maxls": 15,
                             "ftol": 1e-12, "gtol": 1e-7})
                x = x0 + step * result.x
                seed_input = stage.input_from_x(x)
                equilibrium = stage.equilibrium_from_x(x)
                full_x = qs_problem.x_from_input(seed_input)
                qs_stages.append(dict(mode=max_mode, initial_cost=initial,
                                      final_cost=float(result.fun), status=int(result.status),
                                      message=str(result.message)))
                save_seed_checkpoint(seed_input, equilibrium, output_name, "qs")
                save_qs_summary(training_metrics=qp_metrics(equilibrium))
                print(f"QP mode {max_mode}: cost {initial:.6e} -> {result.fun:.6e}", flush=True)

        qs_training_metrics = qp_metrics(equilibrium)
        qs_refined_input, qs_refined_equilibrium = refined(seed_input, equilibrium)
        save_seed_checkpoint(qs_refined_input, qs_refined_equilibrium, output_name, "qs_refined")
        qs_metrics = qp_metrics(qs_refined_equilibrium)
        save_qs_summary(qs_training_metrics, qs_metrics)
        qs_failed = qp_gate_failures(qs_metrics)
        if qs_failed:
            raise RuntimeError("QP preconditioning failed " + ", ".join(qs_failed) + "; checkpoint saved")
    inp = seed_input
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
                                 t1=-0.3 * iota_sign, t2=3.0 * iota_sign, zeta_center=0.0, ntheta=GRID, nzeta=GRID, **PWO_OPTIONS)
    probe_data = probe.compute_state(equilibrium.state, equilibrium.runtime)
    sampled = np.asarray(probe_data["B"])
    bmin, bmax, b_ref = float(sampled.min()), float(sampled.max()), float(sampled.mean())
    calibration, theta_center, zeta_center = None, 0.0, 0.0
    if FAMILY == "canonical":
        unit_shape = np.asarray(probe_data["B_target"]) - 1.0
        shape_mean = float(unit_shape.mean())
        mirror_ref = (bmax - bmin) / (bmax + bmin)
        midpoint = b_ref / (1 + mirror_ref * (2 * shape_mean - 1))
        raw_bounds = [bmin, bmax]
        bmin, bmax = midpoint * (1 - mirror_ref), midpoint * (1 + mirror_ref)
        calibrated = bmin + (bmax - bmin) * unit_shape
        attained_mirror = float(np.ptp(calibrated) / (calibrated.max() + calibrated.min()))
        if not (0 < bmin < bmax and np.isfinite(bmax) and
                abs(calibrated.mean() / b_ref - 1) < 1e-12 and
                abs(attained_mirror / mirror_ref - 1) < 1e-12):
            raise RuntimeError("Canonical fixed mean/mirror calibration failed")
        theta_center, zeta_center, phase_costs = canonical_seed_phase(sampled, calibrated, seed_input.nfp)
        calibration = dict(shape_mean=shape_mean, seed_mean_T=b_ref, seed_mirror=mirror_ref,
                           phase_candidates_rad=[[float(i * np.pi), float(j * np.pi / seed_input.nfp)]
                                                 for i in range(2) for j in range(2)], seed_phase_costs=phase_costs,
                           raw_bounds_T=raw_bounds, target_mean_T=float(calibrated.mean()),
                           attained_mirror=attained_mirror, unit_min=float(unit_shape.min()),
                           unit_max=float(unit_shape.max()))

    def construction(options, grid, surfaces=SURFACES):
        return make_target(surfaces, family=FAMILY, bmin=bmin, bmax=bmax,
                           b_ref=b_ref, options=options, grid=grid, iota_sign=iota_sign,
                           theta_center=theta_center, zeta_center=zeta_center)

    target = construction(PWO_OPTIONS, GRID)
    field_key = "B" if FAMILY == "canonical" else "B_mapped"
    initial_target = target.compute_state(equilibrium.state, equilibrium.runtime)
    surface_field_ref = np.asarray(initial_target[field_key]).mean(axis=(1, 2))
    surface_mirror_ref = np.asarray(initial_target["mirror"])

    def physics_rows(state, runtime):
        # A single live transform supplies the fit and local anti-collapse controls.
        out = target.compute_state(state, runtime)
        B = out[field_key]
        local_mean = jnp.mean(B, axis=(1, 2))
        local_mirror = out["mirror"]
        rows = [out["residuals1d"],
            hold_ratio(local_mean, jnp.asarray(surface_field_ref), FIELD_TOLERANCE, PENALTY),
            jnp.maximum(MIRROR_FLOOR - local_mirror, 0) / MIRROR_FLOOR]
        rows.append(hold_ratio(local_mirror, jnp.asarray(surface_mirror_ref),
                               MIRROR_RELATIVE_TOLERANCE, PENALTY))
        return jnp.concatenate(rows)

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
        print(f"Building operating point beta={beta_target:.3%}, modes={MAX_MODES}", flush=True)
        inp = replace(inp, pres_scale=0.0 if beta_target == 0 else beta_target / unit_beta)

        def constraints(state, runtime):
            mirror = opt.mirror_ratio(state, runtime)
            rows = [
                hold_ratio(opt.aspect_ratio(state, runtime), aspect_ref, ASPECT_TOLERANCE, PENALTY),
                jnp.maximum(IOTA_FLOOR - opt.min_abs_iota(state, runtime), 0) / IOTA_FLOOR,
                jnp.maximum(IOTA_FLOOR - iota_sign * opt.mean_iota(state, runtime), 0) / IOTA_FLOOR,
                jnp.maximum(MIRROR_FLOOR - mirror, 0) / MIRROR_FLOOR,
                jnp.maximum(mirror - MIRROR_TRAIN_TARGET, 0) / MIRROR_TRAIN_TOLERANCE,
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

    final_input, final_equilibrium = refined(inp, equilibrium)
    reference_input = replace(seed_input, pres_scale=inp.pres_scale)
    _, reference_equilibrium = refined(reference_input, seed_equilibrium)
    fine = construction(VALIDATION_OPTIONS, VALIDATION_GRID)
    initial_score = fine.compute_state(reference_equilibrium.state, reference_equilibrium.runtime)
    final_score = fine.compute_state(final_equilibrium.state, final_equilibrium.runtime)
    train_score = target.compute_state(final_equilibrium.state, final_equilibrium.runtime)
    metrics = opt.EquilibriumReporter(
        ("aspect", opt.aspect_ratio, ".5f"), ("min_abs_iota", opt.min_abs_iota, ".5f"),
        ("mean_iota", opt.mean_iota, ".5f"), ("mirror", opt.mirror_ratio, ".5f"),
        ("elongation", opt.max_elongation, ".5f"),
        ("beta", opt.volume_average_beta, ".4%"))("final", final_equilibrium)
    local_B = np.asarray(final_score[field_key])
    local_mean = local_B.mean(axis=(1, 2))
    local_mirror = np.asarray(final_score["mirror"])
    metrics.update(local_mirror=local_mirror.tolist(), local_mean_T=local_mean.tolist(),
                   initial_local_mirror=np.asarray(initial_score["mirror"]).tolist(),
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
        "iota sign": iota_sign * metrics["mean_iota"] >= IOTA_FLOOR,
        "training-surface mirror floor": bool(np.all(local_mirror >= MIRROR_FLOOR)),
        "training-surface field held": bool(np.all(abs(local_mean / surface_field_ref - 1) <= FIELD_TOLERANCE)),
        "mirror interval": MIRROR_FLOOR <= metrics["mirror"] <= MIRROR_LIMIT,
        "elongation limit": metrics["elongation"] <= ELONGATION_LIMIT,
        "aspect held": abs(metrics["aspect"] / aspect_ref - 1) <= ASPECT_TOLERANCE,
        "size held": abs(metrics["minor_radius_m"] / radius_ref - 1) <= SIZE_TOLERANCE,
        "field held": abs(metrics["volavgB_T"] / field_ref - 1) <= FIELD_TOLERANCE,
        "converged": bool(np.all(np.asarray(final_equilibrium.result.converged))),
    }
    checks["training-surface mirror held"] = bool(np.all(
        abs(local_mirror / surface_mirror_ref - 1) <= MIRROR_RELATIVE_TOLERANCE))
    if beta_targets[-1] > 0:
        checks["beta held"] = abs(metrics["beta"] / beta_targets[-1] - 1) <= BETA_TOLERANCE
    unmet = [name for name, met in checks.items() if not met]
    construction_diagnostics = {
        name: np.asarray(final_score[name]).tolist()
        for name in ("eta_fraction", "mapped_area_fraction", "jacobian_lower_bound", "corner_margin", "width")
        if name in final_score}
    summary = dict(family=FAMILY, method=METHOD, smoke=ci_smoke, seconds=time.perf_counter() - started,
                   seed_route=seed_route,
                   preconditioning=(None if qs_metrics is None else
                                    dict(branch="QP", source=QS_INPUT or "near_circular",
                                         seed_perturbation_m=SEED_PERTURBATION,
                                         stages=qs_stages, raw_metrics=qs_raw_metrics,
                                         training_metrics=qs_training_metrics,
                                         metrics=qs_metrics)),
                   stages=stages, metrics=metrics, actual_s=np.asarray(final_score["s_b"]).tolist(),
                   target=dict(calibration=calibration, theta_center_rad=theta_center, zeta_center_rad=zeta_center, p=2, bmin_T=bmin, bmax_T=bmax, b_ref_T=b_ref,
                               surface_field_ref_T=surface_field_ref.tolist(), iota_sign=iota_sign,
                               surface_mirror_ref=surface_mirror_ref.tolist(),
                               t1=-0.3 * iota_sign, t2=3.0 * iota_sign,
                               a=0.3, b=0.0, c=-2.0, eta_min=2.8),
                   settings=dict(stage_ns=STAGE_NS, stage_ftol=STAGE_FTOL, mpol=MPOL,
                                 penalty=PENALTY, mirror_train_target=MIRROR_TRAIN_TARGET,
                                 mirror_train_tolerance=MIRROR_TRAIN_TOLERANCE,
                                 final_ns=FINAL_NS, final_ftol=FINAL_FTOL, final_mpol=FINAL_MPOL,
                                 niter=NITER, grid=GRID, validation_grid=VALIDATION_GRID,
                                 transform=PWO_OPTIONS, validation_transform=VALIDATION_OPTIONS,
                                 mode_ladder=MAX_MODES, iteration_budgets=MAXITER,
                                 mirror_relative_tolerance=MIRROR_RELATIVE_TOLERANCE),
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
