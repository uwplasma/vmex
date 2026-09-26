"""Standalone fixed-boundary derivative qualification using production builders."""
import json
import os
from pathlib import Path
import signal
import time
from . import common

GRADIENT_RTOL, GRADIENT_ATOL = 1e-3, 1e-7
FD_STEPS = (1e-3, 3e-4, 1e-4)
QUALIFICATION_SECONDS = 3600

def check_direction(values, analytic, point, direction, *, output):
    """Require two refined central differences to agree with all analytic rows."""
    import numpy as np

    checks = []
    report = dict(passed=False, checks=checks, rtol=GRADIENT_RTOL,
                  atol=GRADIENT_ATOL, seed=16, reference="independent equilibrium re-solves")

    def finite(values):
        return [float(v) if np.isfinite(v) else None for v in values]

    for h in FD_STEPS:
        fd = (values(point+h*direction)-values(point-h*direction))/(2*h)
        errors = np.abs(fd-analytic)
        passed = bool(np.all(np.isfinite(fd)) and np.all(np.isfinite(analytic))
                      and np.all(errors <= GRADIENT_ATOL+GRADIENT_RTOL*np.abs(analytic)))
        checks.append(dict(h=h, passed=passed, analytic=finite(analytic),
                           finite_difference=finite(fd), absolute_errors=finite(errors)))
        report['passed'] = len(checks) == len(FD_STEPS) and all(c['passed'] for c in checks[-2:])
        (output/'gradient_check.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def verify_problem(stage, output):
    """Re-solve every FD endpoint from the same public accepted equilibrium."""
    import jax.numpy as jnp
    import numpy as np
    from vmex import optimize as opt

    point = np.asarray(stage.joint_problem.x0)
    direction = np.random.default_rng(16).normal(size=point.size)
    direction /= np.linalg.norm(direction)
    derivatives = [stage.joint_problem.grad(point) @ direction]
    if stage.constraint_jacobian is not None:
        derivatives.extend(stage.constraint_jacobian(point) @ direction)
    if stage.coil_constraint_jacobian is not None:
        derivatives.extend(np.asarray(stage.coil_constraint_jacobian(point)) @ direction)
    analytic = np.asarray(derivatives)

    def independent_rows(u):
        x = stage.x0 + stage.scales*u
        # The accepted-state view resets the seed and exact-key caches on every call.
        eq = stage.plasma_problem.equilibrium_from_x(x[:stage.x_boundary0.size])
        rows = [float(stage.plasma_loss(eq.state, eq.runtime) + jnp.sum(stage.coil_costs(jnp.asarray(x))))]
        if stage.constraint_jacobian is not None:
            rows.extend([float(opt.min_abs_iota(eq.state, eq.runtime)), float(opt.major_radius(eq.state, eq.runtime))])
        if stage.coil_constraint_values is not None:
            rows.extend(np.asarray(stage.coil_constraint_values(jnp.asarray(u))))
        return np.asarray(rows)

    return check_direction(independent_rows, analytic, point, direction, output=output)


def main(example, argv=None):
    """Write the separate qualification report, including any failure evidence."""
    args = example.parse_args(argv)
    if args.dry_run:
        print(json.dumps(dict(arguments=vars(args), steps=FD_STEPS,
            rtol=GRADIENT_RTOL, atol=GRADIENT_ATOL, optimize=False), indent=2, default=str))
        return 0
    previous_directory = Path.cwd()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"verification output already exists: {output}")
    started = time.monotonic()
    report = dict(passed=False, checks_rerun=True, optimization_started=False)

    def timeout(*_):
        raise TimeoutError("fixed-boundary qualification time budget reached")

    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(QUALIFICATION_SECONDS)
    try:
        common.setup_run(args)
        os.chdir(output)
        stage = example.build_problem(args)
        report['provenance'] = json.loads((output/'provenance.json').read_text())
        report['checks'] = verify_problem(stage, output)
        report['passed'] = report['checks']['passed']
        print(f"Fixed-boundary derivative qualification passed: {report['passed']}", flush=True)
        return 0 if report['passed'] else 1
    except Exception as error:
        report['failure'] = f"{type(error).__name__}: {error}"
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        os.chdir(previous_directory)
        report['seconds'] = time.monotonic()-started
        if output.is_dir():
            (output/'fixed_qualification.json').write_text(json.dumps(report, indent=2, default=str)+'\n')
