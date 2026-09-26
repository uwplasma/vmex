#!/usr/bin/env python
"""Check fixed-boundary scalar constraints without starting an optimization.

The adjoint is compared with independent equilibrium re-solves. A mismatch
is recorded and fails the check; changing frozen coordinates can contribute
to it, so solver tolerance alone does not establish derivative accuracy.
"""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from single_stage_support import common
GRADIENT_CHECK_FTOL = 1e-22
GRADIENT_RTOL = 1e-3
MAX_MODE, ESS_ALPHA, PARAMETER_STEP = 3, 1.2, 0.1


def check_constraints(problem, values, jacobian, *, steps=(3e-3, 1e-3)):
    """Check three inequality rows along one reproducible boundary direction."""
    import numpy as np

    direction = np.random.default_rng(0).normal(size=problem.x0.size)
    direction *= PARAMETER_STEP * problem.scales / np.linalg.norm(direction)
    analytic = np.asarray(jacobian(problem.x0)) @ direction
    checks, passed = [], True

    def finite_list(values):
        return [float(value) if np.isfinite(value) else None for value in values]

    for h in steps:
        fd = (values(problem.x0 + h*direction) - values(problem.x0 - h*direction))/(2*h)
        errors = np.abs(fd-analytic)/np.maximum(np.maximum(np.abs(fd), np.abs(analytic)), 1e-8)
        passed = passed and bool(np.all(np.isfinite(errors) & (errors < GRADIENT_RTOL)))
        checks.append(dict(h=h, analytic=finite_list(analytic), finite_difference=finite_list(fd),
                           relative_errors=finite_list(errors)))
    return dict(passed=bool(passed), checks=checks, seed=0, rtol=GRADIENT_RTOL,
                reference="independent equilibrium re-solves")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=common.DATA / "input.rotating_ellipse")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--ftol", type=float, default=GRADIENT_CHECK_FTOL)
    parser.add_argument("--max-mode", type=int, default=MAX_MODE)
    parser.add_argument("--wall-seconds", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.ftol < 1 or args.max_mode < 1 or args.wall_seconds < 1:
        parser.error("positive finite force tolerance, mode limit and time budget required")
    if args.dry_run:
        print(json.dumps(vars(args), indent=2, default=str))
        return 0
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("TMPDIR", "XDG_CACHE_HOME", "MPLCONFIGDIR", "CUDA_CACHE_PATH"):
        directory = output / "cache" / name
        directory.mkdir(parents=True)
        os.environ[name] = str(directory)
    os.environ.update(JAX_ENABLE_X64="1", JAX_PLATFORMS="cuda,cpu" if args.device == "gpu" else "cpu",
                      VMEX_COMPILATION_CACHE="disabled", JAX_ENABLE_COMPILATION_CACHE="false")
    sys.path.insert(0, str(HERE.parents[1]))
    import jax
    import jax.numpy as jnp
    import numpy as np
    import vmex as vj
    from vmex import optimize as opt
    from single_stage_support.constraints import PhysicalConstraints
    limits = PhysicalConstraints()

    if jax.default_backend() != args.device or not jax.config.x64_enabled:
        raise RuntimeError("requested device and float64 precision are required")

    def timeout(*_):
        raise TimeoutError("constraint verification time budget reached")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.wall_seconds)
    report = dict(passed=False, configuration=vars(args), command=sys.argv, python=sys.version)
    try:
        report["contract"] = opt.OptimizationQualification.signature(
            parameters=dict(max_mode=args.max_mode, ess_alpha=ESS_ALPHA,
                            parameter_step=PARAMETER_STEP, gradient_rtol=GRADIENT_RTOL),
            input_path=args.input, ftol=args.ftol, device=args.device,
            sources=[__file__, Path(common.__file__).with_name("constraints.py")])
        inp = replace(vj.VmecInput.from_file(args.input), ftol_array=np.array([args.ftol]))
        problem = opt.VmecProblem.from_tuples(inp,
            [(opt.min_abs_iota, 0., 1.), (opt.major_radius, 0., 1.)],
            max_mode=args.max_mode, vary_major_radius=True, use_ess=True, ess_alpha=ESS_ALPHA,
            implicit_jacobian_method="reverse_adjoint", device=args.device)

        def values(x):
            eq = opt.solve_equilibrium(problem.input_from_x(x), device=args.device,
                                       raise_on_max_iterations=True)
            return np.asarray(limits.inequalities(limits.physical_values(eq.state, eq.runtime)))

        def jacobian(x):
            physical = problem.residual(x)
            transform = jax.jacfwd(limits.inequalities)(jnp.asarray(physical))
            return np.asarray(transform) @ problem.residual_jac(x)

        report.update(check_constraints(problem, values, jacobian))
        print(f"Constraint derivative checks passed: {report['passed']}")
        return 0 if report["passed"] else 1
    except Exception as error:
        report.update(passed=False, failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        (output / "constraint_gradient_check.json").write_text(
            json.dumps(report, indent=2, default=str, allow_nan=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
