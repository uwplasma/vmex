#!/usr/bin/env python3
"""Draw the exterior-field figure from ``benchmarks/exterior_field_figure.npz``.

Four panels, each computed by VMEX from ``examples/data`` alone:

(a) accuracy against distance from the boundary. The finite-beta NFP=2 QA deck
    (``input.nfp2_QA_finite_beta``) is solved at fixed boundary and its LCFS
    sampled at 32 (the default), 64 and 96 points per period. The boundary field is replaced by
    the field of a current filament on the magnetic axis, whose exterior field
    is known in closed form, so the error of ``VmecExtender`` is an error
    against an exact answer: ``B`` and ``grad B`` along the outward normal at
    three boundary points, for the default ``near_surface="auto"`` and for the
    direct path alone.
(b) the plasma field of the same equilibrium outside the LCFS in the phi = 0
    plane at the default settings, and the region where the direct path misses
    ``10**-digits`` and the graded rule takes over.
(c) the derivative of the exterior field with respect to the boundary modes
    (``VmecProblem.exterior_field(...).B_vjp``) against central differences of
    re-solved equilibria, at three distances from the boundary.
(d) the plasma field outside a mirror plasma from its boundary and end-plane
    field (``examples/mirror/mirror_exterior_field.py``), with field lines of
    the vacuum field plus the plasma field.

Usage::

    python docs/_static/figures/sources/make_exterior_field_figure.py            # draw
    python docs/_static/figures/sources/make_exterior_field_figure.py --record   # recompute (about 40 min CPU)
    ... --record mirror                                                          # one of accuracy, map, gradient, mirror
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
INPUT = REPO / "examples" / "data" / "input.nfp2_QA_finite_beta"
RECORD = REPO / "benchmarks" / "exterior_field_figure.npz"
FIGURES = REPO / "docs" / "_static" / "figures"
DISTANCES = np.logspace(-4, 0, 17)                  # distance from the boundary [minor radii]
RESOLUTIONS = (32, 64, 96)                         # source samples per period (toroidal and poloidal)
GRADIENT_DISTANCES = (0.05, 0.2, 1.0)               # [minor radii]
GRADIENT_STEPS = (1e-3, 1e-4)                       # central-difference steps in the DOF vector norm
FILAMENT_CURRENT, FILAMENT_SEGMENTS = 1.0e6, 4096  # [A], chords of the axis filament
MIRROR = dict(center_field=1.0, half_length=1.0, ratio=1.5, radius=0.12, stations=25,
              pres_scale=2.0e4, ns=11, elements=10, launch=(0.2, 0.3, 0.4), steps=80)
BLUE, ORANGE, INK, GREY = "#2a78b8", "#eb6834", "#1b1b1b", "#8a8a8a"


def solved_qa():
    """The finite-beta QA deck, its wout and the default exterior field of its LCFS."""
    import vmex as vj

    inp = vj.VmecInput.from_file(INPUT)
    res = vj.solve_multigrid(inp)
    wout = vj.wout_from_state(inp=inp, state=res.state, fsqr=float(res.fsqr), fsqz=float(res.fsqz),
                              fsql=float(res.fsql), niter=int(res.iterations), converged=bool(res.converged))
    return inp, res, wout


def filament_field(wout):
    """``B(xyz)`` [T] of a closed current filament on the magnetic axis (exact segment Biot-Savart)."""
    import jax.numpy as jnp

    nfp = int(wout.nfp)
    phi = np.linspace(0.0, 2.0 * np.pi, FILAMENT_SEGMENTS, endpoint=False)
    n = np.arange(len(wout.raxis_cc))[:, None]
    R = (np.asarray(wout.raxis_cc)[:, None] * np.cos(n * nfp * phi)).sum(0)
    Z = (np.asarray(wout.zaxis_cs)[:, None] * np.sin(n * nfp * phi)).sum(0)
    nodes = jnp.asarray(np.stack((R * np.cos(phi), R * np.sin(phi), Z), 1))
    start, end = nodes, jnp.roll(nodes, -1, axis=0)

    def field(x):
        a, b = x - start, x - end
        ab = end - start
        cross = jnp.cross(ab, a)
        na, nb = jnp.linalg.norm(a, axis=1), jnp.linalg.norm(b, axis=1)
        factor = ((jnp.sum(ab * a, 1) / na - jnp.sum(ab * b, 1) / nb)
                  / jnp.maximum(jnp.sum(cross * cross, 1), 1e-300))
        return 1e-7 * FILAMENT_CURRENT * jnp.sum(cross * factor[:, None], 0)

    return field


def boundary_points(surface, indices, distances_m):
    """Targets at ``distances_m`` along the outward normal of the boundary samples ``indices``."""
    gamma, normal = np.asarray(surface.gamma), np.asarray(surface.normal)
    out = []
    for i, j in indices:
        unit = normal[:, i, j] / np.linalg.norm(normal[:, i, j])
        out.append(gamma[:, i, j][None] + np.asarray(distances_m)[:, None] * unit[None])
    return np.stack(out)   # (locations, distances, 3)


def record_accuracy(wout) -> dict:
    """Error of ``B`` and ``grad B`` against the axis filament at three source resolutions."""
    import jax
    import jax.numpy as jnp

    import vmex as vj
    from vmex.core import virtual_casing as vc

    reference = filament_field(wout)
    minor = float(wout.Aminor_p)
    out = {"acc_distance": DISTANCES, "acc_resolution": np.array(RESOLUTIONS)}
    for n in RESOLUTIONS:
        surface = vc.surface_field_data_from_wout(wout, nphi=n, ntheta=n)
        gamma = np.asarray(surface.gamma)
        on_surface = jax.vmap(reference)(jnp.asarray(gamma.reshape(3, -1).T))
        surface = replace(surface, B_total=jnp.asarray(np.asarray(on_surface).T.reshape(gamma.shape)))
        scale = float(np.sqrt(np.mean(np.sum(np.asarray(surface.B_total) ** 2, axis=0))))
        field = vj.VmecExtender.from_surface_data(surface, accuracy_check="off")
        points = boundary_points(surface, [(0, 0), (0, n // 4), (n // 2, n // 2)], DISTANCES * minor)
        flat = jnp.asarray(points.reshape(-1, 3))
        exact = np.asarray(jax.vmap(reference)(flat))
        exact_grad = np.asarray(jax.vmap(jax.jacfwd(reference))(flat))
        shape = points.shape[:2]
        for mode in ("auto", "direct"):
            field.near_surface = mode
            b = np.asarray(field.B(flat))
            out[f"acc_B_{mode}_{n}"] = (np.linalg.norm(b - exact, axis=1) / scale).reshape(shape)
            out[f"acc_estimate_{mode}_{n}"] = np.asarray(field.B_error_estimate(flat)).reshape(shape)
            if mode == "auto":
                g = np.asarray(field.gradB(flat))
                out[f"acc_grad_{mode}_{n}"] = (np.abs(g - exact_grad).max((1, 2)) * minor / scale).reshape(shape)
        print(f"accuracy {n}: auto B", out[f"acc_B_auto_{n}"].max(0), flush=True)
    return out


def record_map(wout) -> dict:
    """The plasma field of the QA equilibrium outside the LCFS in the phi = 0 plane."""
    import jax.numpy as jnp
    from matplotlib.path import Path as Outline

    import vmex as vj
    from vmex.core.plotting import surface_rz

    field = vj.VmecExtender.from_wout(wout, accuracy_check="off")
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    lcfs = surface_rz(wout, s_index=-1, theta=theta, phi=np.array([0.0]))
    lcfs = np.stack((lcfs[0][:, 0], lcfs[1][:, 0]), 1)
    minor = float(wout.Aminor_p)
    r = np.linspace(lcfs[:, 0].min() - 0.9 * minor, lcfs[:, 0].max() + 0.9 * minor, 90)
    z = np.linspace(-1.25 * minor, 1.25 * minor, 70)
    rr, zz = np.meshgrid(r, z)
    xyz = np.stack((rr.ravel(), 0 * rr.ravel(), zz.ravel()), 1)
    outside = ~Outline(lcfs).contains_points(xyz[:, [0, 2]])
    field_B = np.full(len(xyz), np.nan)
    missed = np.full(len(xyz), np.nan)
    index = np.flatnonzero(outside)
    for i in range(0, len(index), 1024):
        idx = index[i:i + 1024]
        p = jnp.asarray(xyz[idx])
        field_B[idx] = np.linalg.norm(np.asarray(field.B(p)), axis=1)
        field.near_surface = "direct"
        missed[idx] = np.asarray(field.B_error_estimate(p))
        field.near_surface = "auto"
    surface = field.plasma_field.surface_data
    scale = float(np.sqrt(np.mean(np.sum(np.asarray(surface.B_total) ** 2, axis=0))))
    print("map: graded fraction", float(np.mean(missed[outside] > 1e-6)), flush=True)
    return {"map_R": r, "map_Z": z, "map_B": (field_B / scale).reshape(rr.shape),
            "map_direct_error": missed.reshape(rr.shape), "map_lcfs": lcfs, "map_minor": minor}


def record_gradient(inp, wout) -> dict:
    """``B_vjp`` against central differences of re-solved equilibria along one boundary direction."""
    import jax.numpy as jnp

    from vmex import optimize as opt

    problem = opt.VmecProblem.from_input(inp, max_mode=1, use_ess=True, progress=False)
    x0 = np.asarray(problem.x0, dtype=float)
    direction = np.random.default_rng(0).normal(size=x0.size)
    direction /= np.linalg.norm(direction)
    equilibrium = problem.equilibrium_from_x(x0)
    equilibrium.set_points_flux([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    axis, edge = (np.asarray(p) for p in equilibrium.field.get_points_cart())
    minor = float(wout.Aminor_p)
    ray = (edge - axis) / np.linalg.norm(edge - axis)
    weights = np.random.default_rng(1).normal(size=(1, 3))

    def exterior(x, points):
        field = problem.equilibrium_from_x(x).exterior_field(accuracy_check="off")
        return field.set_points_xyz(points)

    out = {"grad_distance": np.array(GRADIENT_DISTANCES), "grad_steps": np.array(GRADIENT_STEPS)}
    ad, fd = [], []
    for d in GRADIENT_DISTANCES:
        points = (edge + d * minor * ray)[None]
        field = problem.equilibrium_from_x(x0).exterior_field(accuracy_check="off").set_points_xyz(points)
        ad.append(float(np.asarray(field.B_vjp(jnp.asarray(weights))) @ direction))
        row = []
        for h in GRADIENT_STEPS:
            plus = float((np.asarray(exterior(x0 + h * direction, points).B()) * weights).sum())
            minus = float((np.asarray(exterior(x0 - h * direction, points).B()) * weights).sum())
            row.append((plus - minus) / (2 * h))
        fd.append(row)
        print(f"gradient d={d}: AD {ad[-1]:.6e} FD {row}", flush=True)
    out["grad_ad"], out["grad_fd"] = np.array(ad), np.array(fd)
    return out


def record_mirror() -> dict:
    """Plasma field outside a fixed-boundary mirror and field lines of vacuum plus plasma field."""
    import jax
    import jax.numpy as jnp
    from scipy.interpolate import CubicSpline
    from virtual_casing_jax.mirror import MirrorExteriorField, mirror_panels

    from vmex.mirror import MirrorInput, solve_mirror
    from vmex.mirror.analytic import AxisymmetricPolynomialMirror

    m = MIRROR
    half = m["half_length"]
    vacuum = AxisymmetricPolynomialMirror(center_field=m["center_field"], half_length=half,
                                          mirror_strength=m["ratio"] - 1.0)
    zb = np.linspace(-half, half, m["stations"])
    inp = MirrorInput(ns=m["ns"], elements=m["elements"], z_min=-half, z_max=half,
                      phiedge=2.0 * np.pi * float(vacuum.poloidal_flux(m["radius"], 0.0)), zb=zb,
                      rbc=[vacuum.boundary_radius(m["radius"], zb)], pres_scale=m["pres_scale"],
                      am=[1.0, -1.0], ftol=1.0e-12, niter=1000)
    mout = solve_mirror(inp).mout()
    z = np.asarray(mout.z)

    def tabulate(x, y):
        fine = np.linspace(x[0], x[-1], 4001)
        table = jnp.asarray(CubicSpline(x, y)(fine))
        fine = jnp.asarray(fine)
        return lambda q: jnp.stack([jnp.interp(q, fine, table[:, k]) for k in range(table.shape[1])])

    def rotate(b, v):
        c, s = jnp.cos(v), jnp.sin(v)
        return jnp.stack((c * b[0] - s * b[1], s * b[0] + c * b[1], b[2]))

    def cap_trace(index):
        r = np.sqrt(np.asarray(mout.s)) * np.asarray(mout.radius_scale)[-1, 0, index]
        table = tabulate(r, np.asarray(mout.b_xyz)[:, 0, index])
        return lambda w, v: rotate(table(w * r[-1]), v)

    wall_radius = tabulate(z, np.asarray(mout.radius_scale)[-1, 0][:, None])
    wall_field = tabulate(z, np.asarray(mout.b_xyz)[-1, 0])
    surface = MirrorExteriorField(mirror_panels(
        lambda zz, v: wall_radius(zz)[0] + 0.0 * v, z[0], z[-1],
        lambda zz, v: rotate(wall_field(zz), v), cap_trace(0), cap_trace(-1)))

    zz = np.linspace(1.15 * z[0], 1.15 * z[-1], 80)
    rr = np.linspace(0.0, 0.42, 44)
    zg, rg = np.meshgrid(zz, rr)
    wall = np.array([float(wall_radius(q)[0]) for q in zz])
    outside = ~((rg <= wall[None]) & (np.abs(zg) <= z[-1]))
    xyz = jnp.asarray(np.stack((rg[outside], 0 * rg[outside], zg[outside]), 1))
    field = np.full(rg.shape, np.nan)
    field[outside] = np.linalg.norm(np.asarray(jax.jit(surface.B_plasma_xyz)(xyz)), axis=1)

    def total(x):
        return vacuum.field(x) + surface.B_plasma_xyz(x[None])[0]

    def direction(x):
        b = total(x)
        return b / jnp.linalg.norm(b)

    unit = jax.jit(jax.vmap(direction))
    h = 1.2 * half / m["steps"]
    x = np.array([[r, 0.0, 0.0] for r in m["launch"]])
    lines = [x]
    for _ in range(m["steps"]):   # RK4 in arc length along B
        k1 = np.asarray(unit(x))
        k2 = np.asarray(unit(x + 0.5 * h * k1))
        k3 = np.asarray(unit(x + 0.5 * h * k2))
        k4 = np.asarray(unit(x + h * k3))
        x = x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        lines.append(x)
    lines = np.stack(lines, 1)
    print("mirror: flux defect", float(surface.flux_defect()), flush=True)
    return {"mirror_z": zz, "mirror_r": rr, "mirror_B": field, "mirror_wall": wall,
            "mirror_lines": lines.astype(np.float32), "mirror_half": half,
            "mirror_flux_defect": float(surface.flux_defect())}


def record(names: list[str]) -> None:
    """Recompute the named parts (default all) and merge them into the record."""
    parts = names or ["accuracy", "map", "gradient", "mirror"]
    saved = dict(np.load(RECORD)) if RECORD.exists() else {}
    if set(parts) - {"mirror"}:
        inp, _, wout = solved_qa()
    for part in parts:
        new = {"accuracy": lambda: record_accuracy(wout), "map": lambda: record_map(wout),
               "gradient": lambda: record_gradient(inp, wout), "mirror": record_mirror}[part]()
        saved |= {k: (v.astype(np.float32) if isinstance(v, np.ndarray) and v.dtype == np.float64
                      and k.startswith(("map_", "mirror_")) else v) for k, v in new.items()}
        np.savez_compressed(RECORD, **saved)


def draw(d) -> None:
    from matplotlib.colors import LogNorm

    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.0))
    (a, b), (c, e) = axes

    x = d["acc_distance"]
    top = int(max(d["acc_resolution"]))
    floor = 1e-17
    for n, shade in zip(d["acc_resolution"], (0.35, 0.65, 1.0)):
        a.loglog(x, np.maximum(d[f"acc_B_auto_{int(n)}"].max(0), floor), color=BLUE, alpha=shade, lw=1.6,
                 label=f"$B$, auto, {int(n)} samples per period")
    a.loglog(x, np.maximum(d[f"acc_B_direct_{top}"].max(0), floor), color=GREY, lw=1.4, ls="--",
             label=f"$B$, direct path only, {top}")
    a.loglog(x, np.maximum(d[f"acc_grad_auto_{top}"].max(0), floor), color=ORANGE, lw=1.6,
             label=f"$a\\,\\nabla B$, auto, {top}")
    a.axhline(1e-6, color=INK, lw=0.5, ls=":")
    a.set(xlabel="distance from the boundary / minor radius", ylabel="error / RMS |B| on the boundary",
          ylim=(1e-15, 1e2), title="(a) known-answer accuracy, QA boundary")
    a.legend(fontsize=6.5, frameon=False, loc="upper right")
    a.grid(alpha=0.2, lw=0.5)

    R, Z, B = d["map_R"], d["map_Z"], d["map_B"]
    image = b.pcolormesh(R, Z, B, norm=LogNorm(vmin=1e-3, vmax=max(float(np.nanmax(B)), 2e-3)),
                         cmap="viridis", shading="auto", rasterized=True)
    b.contour(R, Z, np.nan_to_num(d["map_direct_error"], nan=0.0), levels=[1e-6], colors="white",
              linewidths=0.9, linestyles="--")
    b.plot(d["map_lcfs"][:, 0], d["map_lcfs"][:, 1], color=INK, lw=1.2)
    b.set(aspect="equal", xlabel="R [m]", ylabel="Z [m]", title="(b) plasma field outside the LCFS")
    fig.colorbar(image, ax=b, pad=0.02, shrink=0.8, label="$|B_{plasma}|$ / RMS $|B|_{LCFS}$")
    b.text(0.03, 0.04, "dashed: graded rule inside", transform=b.transAxes, color="white", fontsize=7)

    steps = d["grad_steps"]
    distance = d["grad_distance"]
    for k, h in enumerate(steps):
        error = np.abs(d["grad_ad"] - d["grad_fd"][:, k]) / np.abs(d["grad_ad"])
        c.loglog(distance, error, "o-", color=(BLUE, ORANGE)[k % 2], lw=1.2, ms=4,
                 label=f"finite-difference step {h:g}")
    c.set(xlabel="distance from the boundary / minor radius", ylabel="|AD - FD| / |AD|",
          title="(c) shape derivative vs re-solved FD", ylim=(1e-9, 1e-1))
    c.legend(fontsize=7, frameon=False, loc="upper right")
    c.grid(alpha=0.2, lw=0.5)

    zz, rr = d["mirror_z"], d["mirror_r"]
    half = float(d["mirror_half"])
    field = d["mirror_B"] * 1e3
    while np.isnan(field).any():   # nodes inside the plasma take their neighbours' values, then are painted over
        pad = np.pad(field, 1, constant_values=np.nan)
        ring = np.stack((pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]))
        count = (~np.isnan(ring)).sum(0)
        mean = np.where(np.isnan(ring), 0.0, ring).sum(0) / np.maximum(count, 1)
        field = np.where(np.isnan(field) & (count > 0), mean, field)
    levels = np.logspace(-2, np.log10(max(float(field.max()), 2e-2)), 29)
    image = e.contourf(zz, rr, field, levels=levels, norm=LogNorm(vmin=levels[0], vmax=levels[-1]),
                       cmap="viridis", extend="min")
    inside = np.abs(zz) <= half
    e.fill_between(zz[inside], 0, d["mirror_wall"][inside], color="white", zorder=3)
    e.plot(zz, d["mirror_wall"], color=INK, lw=1.2, zorder=4)
    for end in (-half, half):
        e.plot([end, end], [0, d["mirror_wall"][np.argmin(abs(zz - end))]], color=INK, lw=1.2, zorder=4)
    for line in d["mirror_lines"]:
        for sign in (1, -1):
            e.plot(sign * line[:, 2], np.hypot(line[:, 0], line[:, 1]), color=ORANGE, lw=1.0, zorder=4)
    e.set(xlabel="z [m]", ylabel="r [m]", xlim=(zz[0], zz[-1]), title="(d) mirror: plasma field outside")
    fig.colorbar(image, ax=e, pad=0.02, shrink=0.8, label="$|B_{plasma}|$ [mT]", ticks=[1e-2, 1e-1, 1.0])
    fig.tight_layout()
    fig.savefig(FIGURES / "readme_exterior_field.webp", dpi=100, pil_kwargs={"quality": 60, "method": 6})
    plt.close(fig)


def main() -> None:
    if "--record" in sys.argv:
        record([a for a in sys.argv[1:] if not a.startswith("--")])
    draw(np.load(RECORD))


if __name__ == "__main__":
    main()
