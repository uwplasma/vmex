#!/usr/bin/env python
r"""Fixed-boundary single-stage counterpart of the coil-constraint benchmark.

The boundary and the coils form one variable vector, every trial solves the
fixed-boundary equilibrium, and SLSQP minimizes

    J = (1/2) |r_QS|^2 + (1/2) NORMAL_FIELD_WEIGHT rms(B.n/|B|)^2

(r_QS the constructed QI residual for ``COIL_CASE=qi``) subject to the hard
inequalities of ``parameters.py`` (minimum |iota|, aspect band, major radius,
the QI case's mirror ratio), the coil limits of ``_coil_constraints.py`` (length,
curvature, mean squared curvature, separation, plasma clearance) and the
normal-field limit. The B.n term keeps the prescribed boundary close to what
the coils produce, so the fixed-boundary QA stays meaningful for the coils.

In vacuum PHIEDGE only scales the field, so the coils may enclose any flux:
the counterpart of the free arm's shared current factor. FLUX_TOLERANCE adds
a band on the coils' toroidal flux through the boundary, the counterpart of
fixed currents and PHIEDGE. ``flux_ratio`` is logged either way.

    python single_stage_optimization.py --steps 5 --output runs/fixed

Continue a run with ``--coils <out>/coils.json --wout <out>/wout.nc``: the
boundary restarts from the WOUT and SLSQP from an identity Hessian.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import parameters as P  # noqa: E402

MAX_MODE = 8                       # boundary modes varied; RBC(0,0) stays fixed
ESS_ALPHA = 1.2
# SLSQP's first step has an identity Hessian: the QH and QI residuals start near 1,
# 50 times the QA seed's, so their boundary coordinates are scaled down.
BOUNDARY_STEP, COIL_STEP = (0.1 if P.SEED is None else 0.02), P.COIL_STEP
NORMAL_FIELD_CONSTRAINT = 0.008   # area-weighted RMS B.n/|B| limit
NORMAL_FIELD_WEIGHT = 1.0e3
FLUX_TOLERANCE = None              # e.g. 0.005: relative band on the coils' flux around PHIEDGE
OPTIMIZER_FTOL = 1e-10
NPHI, NTHETA = 37, 32


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="new run directory")
    parser.add_argument("--steps", type=int, default=100, help="SLSQP iterations")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--coils", type=Path, default=P.COILS_FILE)
    parser.add_argument("--wout", type=Path, help="restart the boundary from this WOUT's last surface")
    parser.add_argument("--save-every", type=int, default=25, help="save coils and WOUT every N steps")
    return parser.parse_args(argv)


def boundary_from_wout(inp, wout):
    """``inp`` with the boundary of ``wout``'s last surface, truncated to its resolution."""
    from dataclasses import replace
    import numpy as np

    rbc, zbs = np.zeros_like(inp.rbc), np.zeros_like(inp.zbs)
    for m, n, r, z in zip(np.asarray(wout.xm, int), np.asarray(wout.xn, int) // wout.nfp,
                          wout.rmnc[-1], wout.zmns[-1]):
        if m < rbc.shape[1] and abs(n) <= inp.ntor:
            rbc[n + inp.ntor, m], zbs[n + inp.ntor, m] = r, z
    return replace(inp, rbc=rbc, zbs=zbs)


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
    from essos.fields import BiotSavart
    from essos.surfaces import surfacerzfourier_from_boundary
    from scipy.optimize import minimize
    import vmex as vj
    from vmex import optimize as opt
    import _coil_constraints as coil_limits
    from free_boundary_single_stage_optimization import resize_coils, seed_input, target_residual

    started = time.monotonic()
    inp = seed_input(vj)
    if args.wout is not None:
        inp = boundary_from_wout(inp, vj.read_wout(args.wout))
    phiedge = abs(float(inp.phiedge))

    qs = target_residual()
    plasma_problem = opt.VmecProblem.from_tuples(
        inp, [(qs.residuals_state, 0.0, 1.0)], max_mode=MAX_MODE, use_ess=True, ess_alpha=ESS_ALPHA)

    resize_coils(Coils.from_json(str(args.coils)), P.COIL_ORDER, P.N_SEGMENTS).to_json(str(out / "coils.initial.json"))
    coils0 = Coils.from_json(str(out / "coils.initial.json"))
    x_boundary0 = plasma_problem.x0
    x_coils0 = np.asarray(coils0.curves.dofs).ravel()
    x0 = np.concatenate([x_boundary0, x_coils0])
    scales = np.concatenate([BOUNDARY_STEP * plasma_problem.scales, np.full(x_coils0.size, COIL_STEP)])
    n_boundary = x_boundary0.size

    def objects_from_x(x, nphi=NPHI, ntheta=NTHETA):
        rbc, zbs = plasma_problem.boundary_from_x(x[:n_boundary])
        surface = surfacerzfourier_from_boundary(rbc, zbs, inp.nfp, nphi=nphi, ntheta=ntheta)
        coils = coils0.with_dofs(jnp.concatenate((x[n_boundary:], coils0.dofs_currents)))
        return rbc, zbs, surface, coils

    # Toroidal flux of the coil field through the boundary's phi = 0 cross-section:
    # Gauss-Legendre in radius, uniform in angle, differentiable in boundary and coils.
    rho, rho_weights = np.polynomial.legendre.leggauss(24)
    rho, rho_weights = 0.5 * (rho + 1.0), 0.5 * rho_weights
    theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)

    def toroidal_flux(rbc, zbs, coils):
        m = np.arange(rbc.shape[1])
        cos_m, sin_m = np.cos(np.outer(theta, m)), np.sin(np.outer(theta, m))
        rbc0, zbs0 = rbc.sum(axis=0), zbs.sum(axis=0)
        r_edge, z_edge = cos_m @ rbc0, sin_m @ zbs0
        dr_edge, dz_edge = -sin_m @ (m * rbc0), cos_m @ (m * zbs0)
        r = rbc0[0] + rho[:, None] * (r_edge - rbc0[0])
        z = rho[:, None] * z_edge
        area = rho[:, None] * ((r_edge - rbc0[0]) * dz_edge - z_edge * dr_edge)
        points = jnp.stack([r, jnp.zeros_like(r), z], axis=-1).reshape(-1, 3)
        b_phi = jax.vmap(BiotSavart(coils).B)(points)[:, 1].reshape(r.shape)
        return jnp.sum(rho_weights[:, None] * b_phi * area) * (2.0 * np.pi / theta.size)

    def normal_field_rms(coils, surface):
        field = jax.vmap(BiotSavart(coils).B)(surface.gamma.reshape(-1, 3)).reshape(surface.gamma.shape)
        normal = jnp.sum(field * surface.unitnormal, axis=2) / jnp.linalg.norm(field, axis=2)
        weights = surface.area_element / jnp.sum(surface.area_element)
        return jnp.sqrt(jnp.sum(weights * normal**2))

    aspect_lower, aspect_upper = P.ASPECT_RANGE
    aspect_scale = 0.5 * (aspect_upper - aspect_lower)
    width = P.RADIUS_TOLERANCE - P.RADIUS_MARGIN

    n_plasma_rows = 5 + bool(P.MIRROR_LIMIT)

    def plasma_rows(state, ctx):
        iota, aspect, radius = opt.min_abs_iota(state, ctx), opt.aspect_ratio(state, ctx), opt.major_radius(state, ctx)
        rows = [(iota - P.IOTA_FLOOR - P.IOTA_MARGIN) / P.IOTA_FLOOR,
                (aspect - aspect_lower) / aspect_scale, (aspect_upper - aspect) / aspect_scale,
                (radius - P.RADIUS_TARGET + width) / P.RADIUS_TOLERANCE,
                (P.RADIUS_TARGET + width - radius) / P.RADIUS_TOLERANCE]
        if P.MIRROR_LIMIT:
            rows.append((P.MIRROR_LIMIT - P.MIRROR_MARGIN - opt.mirror_ratio(state, ctx)) / P.MIRROR_LIMIT)
        return jnp.stack(rows)

    def coil_rows(u):
        rbc, zbs, surface, coils = objects_from_x(jnp.asarray(x0) + jnp.asarray(scales) * u)
        clearance = coil_limits.surface_distance(coils, surface)
        rows = [coil_limits.coil_inequalities(coils),
                jnp.atleast_1d((clearance - P.COIL_SURFACE_DISTANCE_LIMIT - P.DISTANCE_MARGIN)
                               / P.COIL_SURFACE_DISTANCE_LIMIT),
                jnp.atleast_1d(1.0 - normal_field_rms(coils, surface) / NORMAL_FIELD_CONSTRAINT)]
        if FLUX_TOLERANCE:
            flux = jnp.abs(toroidal_flux(rbc, zbs, coils)) / phiedge - 1.0
            rows.append(jnp.stack([FLUX_TOLERANCE - flux, FLUX_TOLERANCE + flux]) / FLUX_TOLERANCE)
        return jnp.concatenate(rows)

    coil_rows_jit = jax.jit(coil_rows)
    plasma_rows_jit = jax.jit(lambda u: plasma_problem.jax_extra_costs_from_state(
        (jnp.asarray(x0) + jnp.asarray(scales) * u)[:n_boundary], plasma_rows, n_extra_terms=n_plasma_rows)[1])
    plasma_rows_jac = jax.jit(jax.jacrev(lambda u: plasma_problem.jax_extra_costs_from_state(
        (jnp.asarray(x0) + jnp.asarray(scales) * u)[:n_boundary], plasma_rows, n_extra_terms=n_plasma_rows)[1]))
    coil_rows_jac = jax.jit(jax.jacrev(coil_rows))
    cache = {}
    qa_value_and_grad = jax.jit(jax.value_and_grad(lambda u: plasma_problem.jax_objective_from_state(
        (jnp.asarray(x0) + jnp.asarray(scales) * u)[:n_boundary], lambda state, ctx: jnp.zeros(1),
        n_extra_terms=1)[0]))

    def normal_field_cost(u):
        _, _, surface, coils = objects_from_x(jnp.asarray(x0) + jnp.asarray(scales) * u)
        return 0.5 * NORMAL_FIELD_WEIGHT * normal_field_rms(coils, surface)**2

    normal_field_value_and_grad = jax.jit(jax.value_and_grad(normal_field_cost))

    def objective(u):
        u = np.asarray(u, dtype=float)
        if cache.get("key") != u.tobytes():
            half_qa, gradient = qa_value_and_grad(jnp.asarray(u))
            extra, extra_gradient = normal_field_value_and_grad(jnp.asarray(u))
            cache.update(key=u.tobytes(), qa=2 * float(half_qa), value=float(half_qa + extra),
                         gradient=np.asarray(gradient + extra_gradient))
        return cache["value"], cache["gradient"].copy()

    last = dict(time=time.monotonic(), step=0)

    def log_step(u):
        u = np.asarray(u, dtype=float)
        value, _ = objective(u)
        x = x0 + scales * u
        equilibrium = plasma_problem.equilibrium_from_x(x[:n_boundary])
        state, ctx = equilibrium.solution, equilibrium.runtime
        rbc, zbs, surface, coils = objects_from_x(jnp.asarray(x))
        rows = np.asarray(coil_limits.coil_inequalities(coils))
        now = time.monotonic()
        row = dict(step=last["step"], qa=cache["qa"], objective=value,
                   min_abs_iota=float(opt.min_abs_iota(state, ctx)), aspect=float(opt.aspect_ratio(state, ctx)),
                   major_radius_m=float(opt.major_radius(state, ctx)),
                   **({"mirror_ratio": float(opt.mirror_ratio(state, ctx))} if P.MIRROR_LIMIT else {}),
                   coil_surface_distance_m=float(coil_limits.surface_distance(coils, surface)),
                   coil_minimum_scaled_slack=float(np.min(rows)),
                   normal_field_rms=float(normal_field_rms(coils, surface)),
                   flux_ratio=float(abs(toroidal_flux(rbc, zbs, coils)) / phiedge),
                   step_seconds=now - last["time"], elapsed_seconds=now - started)
        last.update(time=now, step=last["step"] + 1)
        with open(out / "metrics.jsonl", "a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(f"[step {row['step']}] {P.TARGET_NAME}={row['qa']:.6e} iota={row['min_abs_iota']:.5f} "
              f"aspect={row['aspect']:.4f} R={row['major_radius_m']:.5f} flux={row['flux_ratio']:.5f} "
              f"B.n={row['normal_field_rms']:.2e} coil_slack={row['coil_minimum_scaled_slack']:.4f} "
              f"{row['step_seconds']:.1f}s", flush=True)

    def save(tag, u):
        x = x0 + scales * np.asarray(u, dtype=float)
        objects_from_x(jnp.asarray(x))[3].to_json(str(out / f"coils{tag}.json"))
        vj.write_wout(str(out / f"wout{tag}.nc"), plasma_problem.equilibrium_from_x(x[:n_boundary]).wout)

    def checkpoint(u):
        log_step(u)
        if args.save_every and (last["step"] - 1) % args.save_every == 0:
            save(f".step{last['step'] - 1}", u)

    u0 = np.zeros_like(x0)
    checkpoint(u0)
    constraints = [dict(type="ineq", fun=lambda u: np.asarray(plasma_rows_jit(jnp.asarray(u))),
                        jac=lambda u: np.asarray(plasma_rows_jac(jnp.asarray(u)))),
                   dict(type="ineq", fun=lambda u: np.asarray(coil_rows_jit(jnp.asarray(u))),
                        jac=lambda u: np.asarray(coil_rows_jac(jnp.asarray(u))))]
    result = minimize(objective, u0, jac=True, method="SLSQP", callback=checkpoint,
                      constraints=constraints, options={"maxiter": args.steps, "ftol": OPTIMIZER_FTOL})
    x = x0 + scales * result.x
    rbc, zbs, surface, coils = objects_from_x(jnp.asarray(x))
    coils.to_json(str(out / "coils.json"))
    vj.write_wout(str(out / "wout.nc"), plasma_problem.equilibrium_from_x(x[:n_boundary]).wout)
    summary = dict(iterations=int(result.nit), evaluations=int(result.nfev), success=bool(result.success),
                   message=str(result.message), elapsed_seconds=time.monotonic() - started)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
