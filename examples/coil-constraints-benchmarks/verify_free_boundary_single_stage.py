#!/usr/bin/env python
"""Run the shared free-boundary scalar derivative tests independently of production.

Prepare/freeze stage-two coils, certify the initial equilibrium, compare total
objective/constraint derivatives against independent finite differences, and
compare matrix-free gradients and a predictor with dense LU. No single-stage optimization is
started here. A passing report and its two artifacts can seed production.
"""

from pathlib import Path
import signal
import time

import free_boundary_single_stage_optimization_scalar as example

GRADIENT_CHECK_FTOL = 1e-20
GRADIENT_RTOL = 1e-3
GRADIENT_ATOL = 1e-7
LINEARIZATION_PARITY_RTOL = 1e-6
QUALIFICATION_SECONDS = 3600
GRADIENT_STEPS = (3e-4, 1e-4, 3e-5, 1e-5)


def verify_problem(stage, output):
    """Check the exact shared loss/constraints without promoting any FD trial."""
    import numpy as np

    problem, scales = stage.problem, stage.chart.scales
    anchor = problem.accepted
    _, gradient = problem.value_and_grad(problem.x0)
    direction = np.random.default_rng(0).normal(size=problem.x0.size)
    direction /= np.linalg.norm(direction)
    delta = scales*direction
    analytic = np.r_[gradient @ delta, stage.constraint_transform @ problem.constraint_jac(problem.x0) @ delta,
                     stage.coil_constraint.jac(problem.x0) @ delta]
    checks = []
    policy = dict(seed=0, predictor=False, force_tolerance=GRADIENT_CHECK_FTOL,
                  rtol=GRADIENT_RTOL, atol=GRADIENT_ATOL,
                  error_rule="abs(FD-AD) <= atol + rtol*abs(AD)",
                  initial_root_residual=float(anchor.root_residual_norm))
    example.write_json(output / "gradient_check.json", dict(passed=False, checks=checks, **policy))
    consecutive_passes = 0
    for h in GRADIENT_STEPS:
        rows = []
        endpoints = []
        for sign in (1, -1):
            print(f"Independent derivative endpoint: h={h:g}, sign={sign:+d}", flush=True)
            candidate, values = problem.evaluate_trial(sign*h*delta, predict=False, ftol=GRADIENT_CHECK_FTOL)
            endpoints.append(dict(sign=sign, root_residual=float(candidate.root_residual_norm),
                **{name: float(getattr(candidate.result, name)) for name in ("fsqr", "fsqz", "fsql", "fedge")}))
            rows.append(np.r_[values[0], stage.inequalities(values[1:]),
                              stage.coil_constraint.fun(problem.x0+sign*h*delta)])
        fd = (rows[0]-rows[1])/(2*h)
        errors = np.abs(fd-analytic)/np.maximum(np.maximum(np.abs(fd), np.abs(analytic)), 1e-8)
        absolute_errors = np.abs(fd-analytic)
        allowed_errors = GRADIENT_ATOL + GRADIENT_RTOL*np.abs(analytic)
        passed = bool(np.all(np.isfinite(fd)) and np.all(absolute_errors <= allowed_errors))
        checks.append(dict(h=h, passed=passed, analytic=analytic.tolist(), finite_difference=fd.tolist(),
            relative_errors=errors.tolist(), absolute_errors=absolute_errors.tolist(),
            allowed_absolute_errors=allowed_errors.tolist(), endpoints=endpoints))
        example.write_json(output / "gradient_check.json", dict(passed=False, checks=checks, **policy))
        print(f"Derivative h={h:g}: passed={passed}; max relative error={errors.max():.6g}", flush=True)
        consecutive_passes = consecutive_passes + 1 if passed else 0
        if consecutive_passes >= 2:
            break
    if consecutive_passes < 2:
        raise RuntimeError("independent objective/constraint derivative check failed to pass two successive refined steps")
    if problem.accepted is not anchor:
        raise RuntimeError("qualification changed the accepted equilibrium")
    example.write_json(output / "gradient_check.json", dict(passed=True, checks=checks, **policy))
    parity = problem.enable_matrix_free(delta*1e-3, rtol=example.MATRIXFREE_RTOL,
        restart=example.MATRIXFREE_RESTART, max_restarts=example.MATRIXFREE_MAX_CYCLES,
        rhs_batch_size=example.MATRIXFREE_RHS_BATCH_SIZE, parity_rtol=LINEARIZATION_PARITY_RTOL)
    example.write_json(output / "matrixfree_check.json", parity)
    return dict(finite_difference=dict(passed=True, **policy),
                matrix_free=parity)


def main(argv=None):
    """Write a qualification bundle, preserving failure evidence on any gate failure."""
    args = example.parse_args(argv)
    if args.qualification is not None:
        raise ValueError("qualification prepares a fresh seed; omit --qualification")
    if args.dry_run:
        import json
        print(json.dumps(dict(arguments=vars(args), gradient_force_tolerance=GRADIENT_CHECK_FTOL,
                              gradient_rtol=GRADIENT_RTOL, gradient_atol=GRADIENT_ATOL,
                              parity_rtol=LINEARIZATION_PARITY_RTOL), indent=2, default=str))
        return 0
    output = example.setup_run(args)
    (output / Path(__file__).name).write_text(Path(__file__).read_text())

    def timeout(*_):
        raise TimeoutError("qualification wall-time budget reached")

    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(QUALIFICATION_SECONDS)
    problem = None
    started = time.perf_counter()
    report = dict(schema="vmex.single-stage-qualification/v1", passed=False,
                  verification_source_sha256=example.sha(__file__))
    try:
        stage = example.build_problem(args)
        problem = stage.problem
        checkpoint = problem.save_checkpoint(output / "qualified_initial.npz")
        report.update(contract=stage.contract, configuration=dict(
            input=str(args.input.resolve()), wout=None if args.wout is None else str(args.wout.resolve()),
            resolution=args.resolution, grid=args.grid, ftol=args.ftol, device=args.device),
            artifacts=dict(coils=dict(file="coils.stage2.json", sha256=example.sha(output / "coils.stage2.json")),
                           checkpoint=dict(file="qualified_initial.npz", sha256=checkpoint["sha256"])))
        example.write_json(output / "qualification.json", report)
        report["checks"] = verify_problem(stage, output)
        report.update(passed=True, seconds=time.perf_counter()-started)
        example.write_json(output / "qualification.json", report)
        print(f"Qualification passed. Production can reuse {output / 'qualification.json'}", flush=True)
        return 0
    except Exception as error:
        report.update(passed=False, failure=f"{type(error).__name__}: {error}", seconds=time.perf_counter()-started)
        example.write_json(output / "qualification.json", report)
        raise
    finally:
        if problem is not None:
            problem.close()
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
