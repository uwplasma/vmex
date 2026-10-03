#!/usr/bin/env python
"""Free-boundary single-stage coil optimization with hard coil constraints.

Only the coils vary. Each trial solves the vacuum free-boundary equilibrium of
those coils, predicted from the accepted root and certified before use, and
SLSQP minimizes quasisymmetry (or, for ``COIL_CASE=qi``, the constructed QI
residual) subject to hard inequalities: minimum |iota|, major radius, aspect
ratio, the QI case's mirror ratio, coil-to-plasma clearance, and per-coil
length, curvature, mean squared curvature and coil separation
(``parameters.py``).

The coil currents share one free factor. In vacuum only the flux per ampere
sets the plasma size, so fixing both the currents and PHIEDGE pins it: at
iota >= 0.41 that holds the aspect ratio at its 4.9 floor (QA ~0.02), while the
shared factor lets SLSQP reach aspect 5.1 (QA ~0.004). SHARED_CURRENT = False
restores fixed currents.

    python free_boundary_single_stage_optimization.py --steps 5 --output runs/free
    COIL_CASE=qh python free_boundary_single_stage_optimization.py --steps 5 --output runs/free-qh

Every accepted step appends one line to ``metrics.jsonl``. Coils and a WOUT
are saved every ``--save-every`` steps and at the end; restart from them with
``--coils <out>/coils.json --wout <out>/wout.nc``. This case is vacuum only.
"""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import parameters as P  # noqa: E402

# Numerical controls of the equilibrium, adjoint and matrix-free solves.
ROOT_TOLERANCE, ROOT_POLISH_TOLERANCE = 2e-6, 1e-12
ROOT_POLISH_STEPS = 10             # damped Newton steps may be needed from a 1e-9 ordinary-solve residual
OPTIMIZER_FTOL = 1e-10
ADJOINT_RESIDUAL_RTOL, ADJOINT_BATCH_SIZE, ADJOINT_MAX_DOFS = 1e-9, 32, 20000
MATRIXFREE = dict(rtol=1e-11, restart=100, max_restarts=3, rhs_batch_size=3)
LU_REFRESH_HORIZON = 10
NEWTON_STEPS = 8  # Newton-correct predicted trials on the seed LU before any ordinary solve
SHARED_CURRENT = True  # all coil currents vary by one common factor (a free flux per ampere)
CURRENT_STEP = 0.05  # coordinate scale of that relative current factor
DENSE_DERIVATIVES = True  # every derivative a dense solve whose LU seeds the next step's trials (~2x faster)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="new run directory")
    parser.add_argument("--steps", type=int, default=100, help="accepted SLSQP steps")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--coils", type=Path, default=P.COILS_FILE)
    parser.add_argument("--wout", type=Path, help="restart the initial solve from this WOUT")
    parser.add_argument("--max-seconds", type=float, default=float("inf"), help="optimization wall-time budget")
    parser.add_argument("--save-every", type=int, default=25)
    return parser.parse_args(argv)


def resize_coils(coils, order, n_segments):
    """Zero-pad higher Fourier modes without changing the curves or currents."""
    import jax.numpy as jnp
    from essos.coils import Coils, Curves

    if order < coils.order:
        raise ValueError(f"cannot truncate order-{coils.order} coils to order {order}")
    old = coils.curves
    raw = jnp.pad(old.dofs / old.scaling, ((0, 0), (0, 0), (0, 2 * (order - coils.order))))
    curves = Curves(raw, n_segments=n_segments, nfp=coils.nfp, stellsym=coils.stellsym,
                    scaling_type=old.scaling_type, scaling_factor=old.scaling_factor, scale_fixed=old.scale_fixed)
    return Coils(curves, coils.dofs_currents_raw, currents_scale=coils.currents_scale)


def seed_input(vj):
    """The case's fixed-boundary seed at the optimization resolution.

    With ``P.SEED = (nfp, aspect, ratio)`` the boundary is the rotating ellipse
    R = R0 + a cos(theta) - b cos(theta + nfp phi), Z = a sin(theta) + b sin(theta + nfp phi)
    with a^2 - b^2 = (R0 / aspect)^2, b = ratio R0 / aspect and PHIEDGE for B0 ~ 1 T.
    """
    import numpy as np

    mpol, ntor, ns = P.RESOLUTION
    inp = vj.VmecInput.from_file(HERE / "input.rotating_ellipse")
    if P.SEED is not None:
        inp = replace(inp, nfp=P.SEED[0])
    inp = inp.change_resolution(mpol=mpol, ntor=ntor, ntheta=P.GRID[0], nzeta=P.GRID[1])
    if P.SEED is not None:
        _, aspect, ratio = P.SEED
        minor = P.RADIUS_TARGET / aspect
        rbc, zbs = np.zeros_like(inp.rbc), np.zeros_like(inp.zbs)
        rbc[ntor, 0] = P.RADIUS_TARGET
        rbc[ntor, 1] = zbs[ntor, 1] = minor * np.sqrt(1.0 + ratio**2)
        rbc[ntor - 1, 1], zbs[ntor - 1, 1] = -ratio * minor, ratio * minor
        inp = replace(inp, rbc=rbc, zbs=zbs, phiedge=np.pi * minor**2)
    return replace(inp, ns_array=np.array([ns]), ftol_array=np.array([P.EQUILIBRIUM_FTOL]), lfreeb=False)


def target_residual():
    """Residual vector of the case's target: quasisymmetry of ``P.HELICITY``, or constructed QI."""
    import numpy as np
    from vmex import optimize as opt
    from vmex.core.qi import ConstructedQIResidual

    if P.HELICITY is None:
        return ConstructedQIResidual(np.asarray(P.QI_SURFACES), **P.QI_OPTIONS)
    return opt.QuasisymmetryRatioResidual(np.asarray(P.QA_SURFACES), *P.HELICITY)


def main(argv=None):
    args = parse_args(argv)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["JAX_ENABLE_X64"] = "1"
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu" if args.device == "gpu" else "cpu")

    import jax
    import jax.numpy as jnp
    import numpy as np
    from essos.coils import Coils
    from essos.surfaces import surfacerzfourier_from_boundary
    import vmex as vj
    from vmex import optimize as opt
    from vmex.core import implicit as im
    import _coil_constraints as coil_limits

    started = time.monotonic()
    mpol, ntor, ns = P.RESOLUTION
    inp = seed_input(vj)
    if args.wout is not None:
        seed = vj.state_from_wout(vj.read_wout(args.wout), inp=inp, ns=ns)
    else:
        seed = opt.solve_equilibrium(inp, device=args.device, raise_on_max_iterations=True,
                                     polish_force_balance=False).state
    inp = replace(inp, lfreeb=True, mgrid_file="direct ESSOS field")

    # Reload the saved coils so a restart builds a bit-identical coordinate chart.
    resize_coils(Coils.from_json(str(args.coils)), P.COIL_ORDER, P.N_SEGMENTS).to_json(str(out / "coils.initial.json"))
    coils0 = Coils.from_json(str(out / "coils.initial.json"))
    current_dofs = tuple(range(len(coils0.dofs_currents_raw))) if SHARED_CURRENT else ()
    scales = np.r_[np.full(len(current_dofs), CURRENT_STEP),
                   P.COIL_STEP / np.broadcast_to(np.asarray(coils0.curves.scaling), coils0.dofs_curves.shape).ravel()]
    chart = opt.CoilParameters.from_coils(coils0, current_dofs=current_dofs, scales=scales)
    qs = target_residual()

    def boundary(state, runtime, grid):
        rmnc, _, _, zmns = im._edge_physical(state, runtime)
        rows, cols = np.asarray(runtime.modes.n) + ntor, np.asarray(runtime.modes.m)
        rbc = jnp.zeros((2 * ntor + 1, mpol)).at[rows, cols].set(rmnc)
        zbs = jnp.zeros((2 * ntor + 1, mpol)).at[rows, cols].set(zmns)
        return surfacerzfourier_from_boundary(rbc, zbs, inp.nfp, nphi=grid[0], ntheta=grid[1])

    aspect_lower, aspect_upper = P.ASPECT_RANGE
    aspect_scale = 0.5 * (aspect_upper - aspect_lower)
    width = P.RADIUS_TOLERANCE - P.RADIUS_MARGIN

    def clearance(state, runtime, coils):
        return coil_limits.surface_distance(coils, boundary(state, runtime, coil_limits.SURFACE_GRID))

    def qa(state, runtime):
        residuals = qs.residuals_state(state, runtime)
        return jnp.vdot(residuals, residuals)

    def loss(state, runtime, coils):
        return 0.5 * qa(state, runtime)

    def aspect(state, runtime, coils):
        return opt.aspect_ratio(state, runtime)

    seconds = {}

    def record(name, **data):
        if name == "proposal":
            seconds["trials"] = seconds.get("trials", 0) + 1
        if "seconds" in data:
            seconds[name] = seconds.get(name, 0.0) + float(data["seconds"])

    from jax import monitoring
    # Time actually spent tracing and compiling; the cache's "compile_time_saved" is not.
    monitoring.register_event_duration_secs_listener(lambda event, duration, **_: record(
        "compile", seconds=duration) if event.startswith("/jax/core/compile/") else None)

    mirror = (opt.mirror_ratio,) if P.MIRROR_LIMIT else ()
    problem = opt.FreeBoundaryProblem.from_loss(
        inp, loss, quantities=(opt.min_abs_iota, opt.major_radius, *mirror), coil_quantities=(clearance, aspect),
        parameterization=chart, restart_from=seed, root_residual_atol=ROOT_TOLERANCE, event=record,
        deadline=started + args.max_seconds,
        solver_options=dict(device=args.device, ftol=P.EQUILIBRIUM_FTOL, edge_force_tolerance=P.EQUILIBRIUM_FTOL,
                            max_iterations=int(inp.niter_array[-1]), adjoint_dense_batch_size=ADJOINT_BATCH_SIZE,
                            adjoint_dense_max_dofs=ADJOINT_MAX_DOFS, adjoint_residual_rtol=ADJOINT_RESIDUAL_RTOL))
    problem.enable_root_polishing(tolerance=ROOT_POLISH_TOLERANCE, max_steps=ROOT_POLISH_STEPS)
    problem.enable_matrix_free(**MATRIXFREE, refresh_horizon=LU_REFRESH_HORIZON, refresh_max_steps=args.steps,
                               dense_derivatives=DENSE_DERIVATIVES)
    if NEWTON_STEPS:
        problem.enable_newton_correction(max_steps=NEWTON_STEPS)

    coil_rows = coil_limits.constraint(chart.coils_from_x)
    for method in ("fun", "jac"):
        def timed(x, _call=getattr(coil_rows, method), _name=f"coil_constraint_{method}"):
            start = time.monotonic()
            value = _call(x)
            record(_name, seconds=time.monotonic() - start)
            return value
        setattr(coil_rows, method, timed)
    from scipy.optimize import LinearConstraint
    tie = np.zeros((max(len(current_dofs) - 1, 0), chart.size))
    for row in range(tie.shape[0]):  # equal relative currents: one common factor
        tie[row, row], tie[row, row + 1] = 1.0, -1.0
    mirror_bounds = [(-np.inf, P.MIRROR_LIMIT - P.MIRROR_MARGIN, P.MIRROR_LIMIT)] if mirror else []
    lower, upper, row_scales = zip(
        (P.IOTA_FLOOR + P.IOTA_MARGIN, np.inf, P.IOTA_FLOOR),
        (P.RADIUS_TARGET - width, P.RADIUS_TARGET + width, P.RADIUS_TOLERANCE), *mirror_bounds,
        (P.COIL_SURFACE_DISTANCE_LIMIT + P.DISTANCE_MARGIN, np.inf, P.COIL_SURFACE_DISTANCE_LIMIT),
        (aspect_lower, aspect_upper, aspect_scale))
    constraints = [LinearConstraint(tie, 0.0, 0.0)] * bool(tie.size) + [problem.nonlinear_constraint(
        list(lower), list(upper), scales=list(row_scales)), coil_rows]

    def save(tag):
        x = problem.accepted.parameters
        problem.coils_from_x(x).to_json(str(out / f"coils{tag}.json"))
        vj.write_wout(str(out / f"wout{tag}.nc"), problem.equilibrium_from_x(x).wout)

    last = dict(time=time.monotonic(), x=problem.accepted.parameters.copy())
    qa_of = jax.jit(lambda state: qa(state, problem.rt))

    def log_step():
        x, record = problem.accepted.parameters, problem.accepted
        values = list(map(float, problem.constraint_values(x)))
        iota, radius, surface, aspect_value = values[0], values[1], values[-2], values[-1]
        now = time.monotonic()
        row = dict(step=problem.accepted_step, qa=float(qa_of(record.state)), objective=problem.fun(x), min_abs_iota=iota, major_radius_m=radius,
                   aspect=aspect_value, **({'mirror_ratio': values[2]} if mirror else {}), coil_surface_distance_m=surface,
                   coil_minimum_scaled_slack=float(np.min(coil_rows.fun(x))),
                   current_factor=float(chart.base_currents_at(x)[0] / chart.currents[0]),
                   step_u_linf=float(np.max(np.abs((x - last["x"]) / problem.scales))),
                   root_residual=float(record.root_residual_norm), fedge=float(record.result.fedge),
                   step_seconds=now - last["time"], elapsed_seconds=now - started,
                   seconds={k: round(v, 3) for k, v in seconds.items()},
                   peak_gpu_gib=(jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 2**30)
        seconds.clear()
        last.update(time=now, x=x.copy())
        with open(out / "metrics.jsonl", "a") as stream:
            stream.write(json.dumps({k: (float(v) if isinstance(v, np.floating) else v) for k, v in row.items()}) + "\n")
        print(f"[step {row['step']}] {P.TARGET_NAME}={row['qa']:.6e} iota={iota:.5f} R={radius:.5f} aspect={aspect_value:.4f} "
              f"clearance={surface:.4f} coil_slack={row['coil_minimum_scaled_slack']:.4f} "
              f"{row['step_seconds']:.1f}s", flush=True)
        if row["step"] % args.save_every == 0:
            save(f".step{row['step']}")

    log_step()
    try:
        while True:  # a rejected equilibrium trial ends an SLSQP call; restart it from the accepted root
            before = problem.accepted_step
            result = opt.minimize(problem, x0=problem.accepted.parameters, method="SLSQP", constraints=constraints,
                                  callback=lambda x: log_step(),
                                  options=dict(maxiter=args.steps - before, ftol=OPTIMIZER_FTOL))
            if (result.stop_reason != "equilibrium_trial_rejected" or problem.accepted_step == before
                    or problem.accepted_step >= args.steps):
                break
        stop = dict(success=bool(result.success), message=str(result.message),
                    stop_reason=getattr(result, "stop_reason", None))
    except TimeoutError:
        stop = dict(success=False, message="wall-time budget reached", stop_reason="walltime")
    save("")
    summary = dict(accepted_steps=problem.accepted_step, **stop, solver=problem.solver_info,
                   failed_trials=problem.metadata["holder"]["failed_trials"],
                   elapsed_seconds=time.monotonic() - started)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, default=str))
    problem.close()


if __name__ == "__main__":
    main()
