#!/usr/bin/env python
"""Stage-two coil fit that produced the ``examples/data/ESSOS_coils_<case>.json`` files.

The starting coils of the coil-constraint single-stage examples
(``examples/optimization/single_stage_*_coil_constraints.py``). Starting from
``N_COILS`` circular coils of radius ``--radius`` per half period, centred on
R = ``RADIUS_TARGET``, L-BFGS-B minimizes

    0.5 (rms B.n/|B| / 1e-3)^2 + 0.5 WEIGHT sum_j min(c_j, 0)^2

on the case's fixed-boundary seed (``seed_input``, no equilibrium solve), with
c_j the coil rows of ``coil_inequalities`` and the plasma clearance; the case
values (``COIL_CASE``) are the examples', repeated below. The currents are then
scaled so the coils' toroidal flux through the seed's phi = 0 cross-section is
PHIEDGE. The optimization scripts rescale them again, to an edge R B_phi of
B0 R0.

    COIL_CASE=qa4-beta python benchmarks/coil_constraints_fit_coils.py --output ESSOS_coils_qa4_beta.json

The winding radii were 0.55 m for ``qa4-beta*``, 0.46 m for ``qi6-beta*`` and
0.42-0.5 m for ``qa3``, ``qh`` and ``qi`` (``--radius``; rms B.n/|B| 3.7e-3,
4.4e-3 and 5e-4), with 3000 iterations and weight 1 (100 for
``qa4-beta-tok``, whose coils otherwise exceed the length limit). The ``qa3``,
``qh`` and ``qi`` coils start up to 5 % over the length and curvature limits,
which SLSQP then enforces. The fit is fast on a GPU and slow on a loaded CPU:
the coil-coil distance over all symmetry copies dominates.
"""
import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path

os.environ["JAX_ENABLE_X64"] = "1"

# ---- cases ------------------------------------------------------------------------------------------------------
# COIL_CASE selects the case of the examples; only the values read here are repeated. The values are ellipse5's; the
# blocks after them override them per case.
DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
CASE = os.environ.get("COIL_CASE", "ellipse5")
RESOLUTION = (8, 8, 51)            # MPOL, NTOR, NS of the seed
GRID = (64, 64)                    # NTHETA, NZETA
EQUILIBRIUM_FTOL = 1e-15
INPUT_FILE = DATA / "input.rotating_ellipse_nfp2"
SEED = None                        # (nfp, aspect, b / a_eff): rotating ellipse replacing the deck's boundary
ASPECT_RANGE = (4.9, 5.1)
RADIUS_TARGET = 1.0

# Coils and their hard engineering limits.
N_COILS, COIL_ORDER, N_SEGMENTS = 3, 16, 256
COIL_STEP = 0.05                   # coordinate scale of the coil Fourier modes
LENGTH_LIMIT = 5.0                 # m, each independent coil
# Set: the limits follow the widest plasma allowed, a = R / aspect_min, and the clearance d:
# LENGTH_LIMIT = c_L 2 pi (a + d), CURVATURE_LIMIT = c_k / (a + d), MSC_LIMIT = c_m / (a + d)^2.
COIL_LIMIT_FACTORS = None          # (c_L, c_k, c_m)
CURVATURE_LIMIT = 5.0              # 1/m, everywhere along each coil
MSC_LIMIT = 5.0                    # 1/m^2, each coil
COIL_DISTANCE_LIMIT = 0.15         # m, including symmetry copies
COIL_SURFACE_DISTANCE_LIMIT = 0.20 # m, to the plasma boundary
# Interior margins of the sampled constraints.
CURVATURE_MARGIN, MSC_MARGIN, LENGTH_MARGIN, DISTANCE_MARGIN = 0.10, 0.02, 1e-5, 0.001

if CASE in ("qa3", "qh", "qi"):
    N_COILS, COIL_ORDER = 3, 8
    LENGTH_LIMIT, CURVATURE_LIMIT, MSC_LIMIT = 3.5, 8.0, 10.0
    COIL_DISTANCE_LIMIT, COIL_SURFACE_DISTANCE_LIMIT = 0.08, 0.15
    SEED, ASPECT_RANGE = {"qa3": ((3, 6.0, 0.5), (5.9, 6.1)), "qh": ((4, 6.0, 0.9), (5.9, 6.1)),
                          "qi": ((4, 8.0, 0.5), (7.9, 8.1))}[CASE]
elif CASE.removesuffix("-tok") in ("qa4-beta", "qi6-beta", "qh4-beta"):
    # (1.8, 2.5, 1.2): mid-range of Wechsung et al. (2022), Jorge et al. (2023) and Wiedman et al. (2024)
    N_COILS, COIL_ORDER, COIL_LIMIT_FACTORS = 4, 12, (1.8, 2.5, 1.2)
    if CASE.startswith("qa4-beta"):
        SEED, ASPECT_RANGE = (2, 4.0, 0.5), (3.5, 4.5)
        COIL_DISTANCE_LIMIT, COIL_SURFACE_DISTANCE_LIMIT = 0.10, 0.20
    elif CASE.startswith("qh4-beta"):
        N_COILS, SEED, ASPECT_RANGE = 3, (4, 6.0, 0.9), (5.9, 6.1)
        COIL_DISTANCE_LIMIT, COIL_SURFACE_DISTANCE_LIMIT = 0.08, 0.15
    else:
        SEED, ASPECT_RANGE = (4, 6.0, 0.7), (5.9, 6.1)
        COIL_DISTANCE_LIMIT, COIL_SURFACE_DISTANCE_LIMIT = 0.08, 0.15
elif CASE not in ("ellipse5", "ellipse5-beta7"):
    raise ValueError(f"unknown COIL_CASE {CASE!r}")
if CASE.endswith("-tok"):
    SEED = (SEED[0], SEED[1], 0.01)  # a circular tokamak with a 1% helical ripple
if COIL_LIMIT_FACTORS is not None:
    _radius = RADIUS_TARGET / ASPECT_RANGE[0] + COIL_SURFACE_DISTANCE_LIMIT
    LENGTH_LIMIT = COIL_LIMIT_FACTORS[0] * 2 * math.pi * _radius
    CURVATURE_LIMIT, MSC_LIMIT = COIL_LIMIT_FACTORS[1] / _radius, COIL_LIMIT_FACTORS[2] / _radius**2


# ---- the examples' seed and coil rows -----------------------------------------------------------------------------
def seed_input():
    """The case's fixed-boundary seed at the optimization resolution.

    With ``SEED = (nfp, aspect, ratio)`` the boundary is the rotating ellipse
    R = R0 + a cos(theta) - b cos(theta + nfp phi), Z = a sin(theta) + b sin(theta + nfp phi)
    with a^2 - b^2 = (R0 / aspect)^2, b = ratio R0 / aspect and PHIEDGE for B0 ~ 1 T;
    otherwise it is ``INPUT_FILE``'s. ``finite_beta_input`` then calibrates PHIEDGE.
    """
    import numpy as np
    import vmex as vj

    mpol, ntor, ns = RESOLUTION
    inp = vj.VmecInput.from_file(INPUT_FILE)
    if SEED is not None:
        inp = replace(inp, nfp=SEED[0])
    inp = inp.change_resolution(mpol=mpol, ntor=ntor, ntheta=GRID[0], nzeta=GRID[1])
    if SEED is not None:
        _, aspect, ratio = SEED
        minor = RADIUS_TARGET / aspect
        rbc, zbs = np.zeros_like(inp.rbc), np.zeros_like(inp.zbs)
        rbc[ntor, 0] = RADIUS_TARGET
        rbc[ntor, 1] = zbs[ntor, 1] = minor * np.sqrt(1.0 + ratio**2)
        rbc[ntor - 1, 1], zbs[ntor - 1, 1] = -ratio * minor, ratio * minor
        inp = replace(inp, rbc=rbc, zbs=zbs, phiedge=np.pi * minor**2)
    return replace(inp, ns_array=np.array([ns]), ftol_array=np.array([EQUILIBRIUM_FTOL]), lfreeb=False)


def coil_field(coils):
    """points (..., 3) -> the coils' Biot-Savart field B (..., 3)."""
    import jax
    from essos.fields import BiotSavart

    biot_savart = BiotSavart(coils)
    return lambda points: jax.vmap(biot_savart.B)(points.reshape(-1, 3)).reshape(points.shape)


def weighted_rms(weights, values):
    """sqrt(sum weights values^2): the RMS for area weights that sum to one."""
    import jax.numpy as jnp

    return jnp.sqrt(jnp.sum(weights * values**2))


def normal_field_rms(coils, surface):
    """Area-weighted RMS of the coil-only B.n/|B| on an ESSOS surface."""
    import jax.numpy as jnp

    field = coil_field(coils)(surface.gamma)
    normal = jnp.sum(field * surface.unitnormal, axis=2) / jnp.linalg.norm(field, axis=2)
    return weighted_rms(surface.area_element / jnp.sum(surface.area_element), normal)


CURVATURE_POINTS = max(512, 64*COIL_ORDER)
DISTANCE_POINTS = max(128, 16*COIL_ORDER)
SURFACE_GRID = (61, 64)
SELF_CLEARANCE = 1e-6  # numerical nonintersection guard, metres


def resampled(coils, points, *, symmetric=True):
    """ESSOS curves of ``coils`` on ``points`` quadrature points; base curves only unless ``symmetric``."""
    curves = coils.curves.copy()
    curves.n_segments = points
    if not symmetric:
        curves.nfp, curves.stellsym = 1, False
    return curves


def segment_distances(a, b):
    """All distances between two closed polygons' segments, including interiors."""
    import jax.numpy as jnp

    u, v = jnp.roll(a, -1, axis=0)-a, jnp.roll(b, -1, axis=0)-b
    w = a[:, None]-b[None, :]
    aa, bb = jnp.sum(u*u, axis=-1)[:, None], jnp.sum(v*v, axis=-1)[None, :]
    uv = jnp.einsum('ik,jk->ij', u, v)
    uw, vw = jnp.sum(u[:, None]*w, axis=-1), jnp.sum(v[None, :]*w, axis=-1)
    aa, bb = jnp.maximum(aa, 1e-30), jnp.maximum(bb, 1e-30)
    den = aa*bb-uv*uv
    safe = jnp.where(den > 1e-24, den, 1.0)
    s, t = (uv*vw-bb*uw)/safe, (aa*vw-uv*uw)/safe
    def distance(s, t):
        q = w+s[..., None]*u[:, None]-t[..., None]*v[None, :]
        return jnp.sum(q*q, axis=-1)
    candidates = [distance(jnp.zeros_like(uw), jnp.clip(vw/bb, 0, 1)),
                  distance(jnp.ones_like(uw), jnp.clip((vw+uv)/bb, 0, 1)),
                  distance(jnp.clip(-uw/aa, 0, 1), jnp.zeros_like(uw)),
                  distance(jnp.clip((uv-uw)/aa, 0, 1), jnp.ones_like(uw)),
                  jnp.where((den > 1e-24)&(s>=0)&(s<=1)&(t>=0)&(t<=1), distance(s,t), jnp.inf)]
    return jnp.sqrt(jnp.min(jnp.stack(candidates), axis=0)+1e-30)


def separations(points):
    """Minimum intercoil and nonadjacent self-segment distance, all symmetry copies."""
    import jax
    import jax.numpy as jnp

    pairs = jnp.asarray([(i,j) for i in range(len(points)) for j in range(i+1,len(points))])
    inter = jax.lax.map(lambda ij: jnp.min(segment_distances(points[ij[0]], points[ij[1]])), pairs)
    n = points.shape[1]
    diff = jnp.abs(jnp.arange(n)[:,None]-jnp.arange(n)[None,:])
    nonadjacent = jnp.minimum(diff, n-diff)>1
    own = jax.lax.map(lambda p: jnp.min(jnp.where(nonadjacent, segment_distances(p,p), jnp.inf)), points)
    return jnp.min(inter), jnp.min(own)


def coil_metrics(coils, *, curvature_points=CURVATURE_POINTS, distance_points=DISTANCE_POINTS):
    """ESSOS length, peak curvature and speed of each base coil, plus what ESSOS lacks."""
    import jax.numpy as jnp

    base = resampled(coils, curvature_points, symmetric=False)
    speed = jnp.linalg.norm(base.gamma_dash, axis=-1)
    curvature = base.curvature
    cc, own = separations(resampled(coils, distance_points).gamma)
    return dict(length=base.length, peak=jnp.max(curvature, axis=1),
                msc=jnp.sum(curvature**2*speed, axis=1)/jnp.sum(speed, axis=1),
                coil_distance=cc, self_distance=own, min_speed=jnp.min(speed, axis=1))


def coil_inequalities(coils):
    import jax.numpy as jnp

    m = coil_metrics(coils)
    return jnp.concatenate(((LENGTH_LIMIT-LENGTH_MARGIN-m['length'])/LENGTH_LIMIT,
        (CURVATURE_LIMIT-CURVATURE_MARGIN-m['peak'])/CURVATURE_LIMIT,
        (MSC_LIMIT-MSC_MARGIN-m['msc'])/MSC_LIMIT,
        jnp.atleast_1d((m['coil_distance']-COIL_DISTANCE_LIMIT-DISTANCE_MARGIN)/COIL_DISTANCE_LIMIT),
        jnp.atleast_1d((m['self_distance']-SELF_CLEARANCE)/COIL_DISTANCE_LIMIT),
        (m['min_speed']-1e-4)/LENGTH_LIMIT))


def surface_distance(coils, surface):
    """Sampled coil-to-moving-surface clearance; JAX differentiates both sides."""
    import jax
    import jax.numpy as jnp

    points = resampled(coils, DISTANCE_POINTS).gamma
    targets = surface.gamma.reshape(-1,3)
    # Mapping bounds memory and preserves exact differentiation of the active min.
    return jnp.min(jax.lax.map(lambda p: jnp.sqrt(jnp.min(jnp.sum((p-targets)**2,axis=1))+1e-30),
                               points.reshape(-1,3)))


# ---- the fit ---------------------------------------------------------------------------------------------------------
RADIUS = {"qa4-beta": 0.55, "qa4-beta-tok": 0.55, "qi6-beta": 0.46, "qi6-beta-tok": 0.46}  # m


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--radius", type=float, default=RADIUS.get(CASE),
                        help=f"winding radius of the circular start [m] (default for COIL_CASE={CASE}: "
                             f"{RADIUS.get(CASE)})")
    parser.add_argument("--maxiter", type=int, default=3000, help="L-BFGS-B iterations")
    parser.add_argument("--weight", type=float, default=1.0, help="penalty weight of the coil rows")
    parser.add_argument("--output", type=Path, default=Path(f"ESSOS_coils_{CASE.replace('-', '_')}.json"))
    args = parser.parse_args(argv)
    if args.radius is None:
        parser.error(f"COIL_CASE={CASE} has no default --radius")
    return args


def main(argv=None):
    args = parse_args(argv)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from essos.coils import Coils, CreateEquallySpacedCurves
    from essos.fields import BiotSavart
    from essos.surfaces import surfacerzfourier_from_boundary
    from scipy.optimize import minimize

    inp = seed_input()
    surface = surfacerzfourier_from_boundary(jnp.asarray(inp.rbc), jnp.asarray(inp.zbs), inp.nfp,
                                             nphi=SURFACE_GRID[0], ntheta=SURFACE_GRID[1])
    curves = CreateEquallySpacedCurves(N_COILS, COIL_ORDER, RADIUS_TARGET, args.radius,
                                       n_segments=N_SEGMENTS, nfp=inp.nfp, stellsym=True)
    coils0 = Coils(curves, np.full(N_COILS, 1.0e5))
    x0 = jnp.asarray(coils0.curves.dofs).ravel()

    def coils_from_u(u):
        return coils0.with_dofs(jnp.concatenate((x0 + COIL_STEP * u, coils0.dofs_currents)))

    def rows(coils):
        return jnp.concatenate([coil_inequalities(coils), jnp.atleast_1d(
            (surface_distance(coils, surface) - COIL_SURFACE_DISTANCE_LIMIT - DISTANCE_MARGIN)
            / COIL_SURFACE_DISTANCE_LIMIT)])

    def objective(u):
        c = coils_from_u(u)
        return 0.5 * (normal_field_rms(c, surface) / 1e-3)**2 + 0.5 * args.weight * jnp.sum(jnp.minimum(rows(c), 0.0)**2)

    value_and_grad = jax.jit(jax.value_and_grad(objective))
    print(f"circular: B.n rms {float(normal_field_rms(coils0, surface)):.3e}, "
          f"min row {float(jnp.min(rows(coils0))):.3f}", flush=True)
    fit = minimize(lambda u: tuple(map(np.asarray, value_and_grad(jnp.asarray(u)))), np.zeros(x0.size), jac=True,
                   method="L-BFGS-B", options=dict(maxiter=args.maxiter, maxcor=30, ftol=1e-15, gtol=1e-10))
    coils = coils_from_u(jnp.asarray(fit.x))

    # Toroidal flux through the seed's phi = 0 cross-section (as single_stage_optimization_coil_constraints.py).
    rho, rw = np.polynomial.legendre.leggauss(24)
    rho, rw = 0.5 * (rho + 1), 0.5 * rw
    theta = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    m = np.arange(inp.rbc.shape[1])
    rbc0, zbs0 = inp.rbc.sum(axis=0), inp.zbs.sum(axis=0)
    cm, sm = np.cos(np.outer(theta, m)), np.sin(np.outer(theta, m))
    re, ze, dre, dze = cm @ rbc0, sm @ zbs0, -sm @ (m * rbc0), cm @ (m * zbs0)
    r, z = rbc0[0] + rho[:, None] * (re - rbc0[0]), rho[:, None] * ze
    area = rho[:, None] * ((re - rbc0[0]) * dze - ze * dre)
    points = jnp.asarray(np.stack([r, 0 * r, z], -1).reshape(-1, 3))
    bphi = np.asarray(jax.vmap(BiotSavart(coils).B)(points))[:, 1].reshape(r.shape)
    flux = float(np.sum(rw[:, None] * bphi * area) * 2 * np.pi / theta.size)
    scale = abs(float(inp.phiedge)) / abs(flux)
    coils = Coils(coils.curves, coils.dofs_currents_raw * scale, currents_scale=coils.currents_scale)

    metrics = coil_metrics(coils)
    report = dict(case=CASE, radius=args.radius, nit=int(fit.nit), message=str(fit.message),
                  bn_rms=float(normal_field_rms(coils, surface)), min_row=float(jnp.min(rows(coils))),
                  current_A=float(np.asarray(coils.currents)[0]), flux_scale=scale,
                  length=np.asarray(metrics["length"]).tolist(), peak=np.asarray(metrics["peak"]).tolist(),
                  msc=np.asarray(metrics["msc"]).tolist(), coil_distance=float(metrics["coil_distance"]),
                  clearance=float(surface_distance(coils, surface)))
    print(json.dumps(report), flush=True)
    coils.to_json(str(args.output))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
