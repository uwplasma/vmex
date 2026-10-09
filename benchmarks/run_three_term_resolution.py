#!/usr/bin/env python3
"""Three-term free boundary against VMEC + NESTOR and DESC at mpol = ntor = 4 ... 12.

The case has a known answer.  The fixed-boundary QA equilibrium of
``examples/data/input.LandremanPaul2021_QA_beta2p5_bootstrap`` (``--case beta``,
the default; ``--case vacuum`` uses the vacuum ``input.LandremanPaul2021_QA_lowres``) at 12 x 12,
``ns`` 51 is the target.  Its external field is a winding-surface current at
1.2 minor radii, fitted so that the coil field cancels the plasma's own normal
field on the target (virtual casing) and carries the target's net poloidal
current.  The target is then a free boundary of that field without a sheet
current, and every method is scored against it.  Each step runs in a process of
its own, so wall time and peak GPU memory belong to that step alone::

    python benchmarks/run_three_term_resolution.py field OUT          # target and field, once
    python benchmarks/run_three_term_resolution.py target M OUT       # fixed-boundary target and start at M x M
    python benchmarks/run_three_term_resolution.py nestor M OUT       # VMEC + NESTOR free boundary
    python benchmarks/run_three_term_resolution.py three_term M OUT   # the three-term free boundary
    python benchmarks/run_three_term_resolution.py mgrid OUT          # the field as an mgrid table, once
    python benchmarks/run_three_term_resolution.py vmec2000 M OUT     # VMEC2000 + NESTOR on it (CPU, $VMEC2000_CMD)
    python benchmarks/run_three_term_resolution.py nestor_mgrid M OUT # VMEX + NESTOR on the same table
    python benchmarks/run_three_term_resolution.py score M OUT NAME WOUT   # score another code's boundary
    python benchmarks/run_three_term_resolution.py table OUT          # collect OUT/*/*.json

All methods start from the same boundary, the target's with every ``m >= 1``
coefficient scaled by ``START_SCALE``.  Every result is scored the same way: its
boundary is re-solved as a fixed-boundary equilibrium at that resolution, and
the three interface conditions (B.n, the pressure jump
(|B_out|^2 - |B_in|^2 - 2 mu0 p) / (2 |B_in|^2) and the sheet current) are evaluated with a 48 x 48 virtual-casing grid
and a fixed singular quadrature (4 nfp 48 x 96).  Other scores are the force residual, the QA
residual, iota on the axis and the edge, and the largest LCFS distance to the
12 x 12 target and to the target at the same resolution.  DESC runs with
its own scripts outside VMEX (DESC is not a VMEX dependency), and ``score``
scores its wout, re-solved with all of its modes.  VMEC2000 reads the field from an mgrid table:
301 x 301 points in (R, Z), about 4 mm apart, and 48 planes per field period.
Its ``NZETA`` (16, 24 or 48, the smallest one at or above 2 ntor + 6) divides
48, so the table is sampled on its own planes.  ``nestor_mgrid`` runs VMEX +
NESTOR on the same table, which separates the table's interpolation error from
the difference between the codes.  ``$VMEC2000_CMD`` is the command that runs
``xvmec``, for example ``mpirun -np 8 /path/to/xvmec``.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

CASES = {  # --case: the deck whose 12 x 12 fixed-boundary equilibrium is the known answer
    "beta": REPO / "examples" / "data" / "input.LandremanPaul2021_QA_beta2p5_bootstrap",  # beta 2.5%, bootstrap current
    "vacuum": REPO / "examples" / "data" / "input.LandremanPaul2021_QA_lowres",  # precise QA, no pressure or current
}
DECK = CASES["beta"]
NS = 51
TRUTH_MODES = 12
START_SCALE = 0.97
VC_GRID = 48
QA_SURFACES = np.arange(1, 11) / 10
# A score keeps these entries of the other code's own report.json (its time, memory and convergence).
REPORT_KEYS = ("seconds", "total_seconds", "peak_gpu_gib", "peak_rss_gib", "device", "nfev", "message", "converged",
               "exit_code", "free_fsq")


def deck(modes, ns=NS):
    import vmex as vj

    inp = vj.VmecInput.from_file(str(DECK)).change_resolution(
        mpol=modes, ntor=modes, ntheta=2 * modes + 6, nzeta=2 * modes + 6)
    return replace(inp, ns_array=np.array([ns]), ftol_array=np.array([1e-13]), niter_array=np.array([60000]))


def start_boundary(inp):
    """The common initial guess: every m >= 1 coefficient of ``inp``'s boundary scaled by ``START_SCALE``."""
    rbc, zbs = np.array(inp.rbc, dtype=float), np.array(inp.zbs, dtype=float)
    rbc[:, 1:] *= START_SCALE
    zbs[:, 1:] *= START_SCALE
    return replace(inp, rbc=rbc, zbs=zbs)


def with_wout_boundary(inp, w):
    """``inp`` with the LCFS (and axis) of ``w``, truncated or padded to ``inp``'s modes."""
    rbc, zbs = np.zeros_like(inp.rbc), np.zeros_like(inp.zbs)
    for m, k, r, z in zip(np.asarray(w.xm, int), np.asarray(w.xn, int) // int(w.nfp), np.asarray(w.rmnc)[-1],
                          np.asarray(w.zmns)[-1]):
        if m < inp.mpol and abs(k) <= inp.ntor:
            rbc[k + inp.ntor, m], zbs[k + inp.ntor, m] = r, z
    n = min(inp.ntor + 1, np.asarray(w.raxis_cc).size)
    raxis, zaxis = np.zeros(inp.ntor + 1), np.zeros(inp.ntor + 1)
    raxis[:n], zaxis[:n] = np.asarray(w.raxis_cc)[:n], np.asarray(w.zaxis_cs)[:n]
    return replace(inp, rbc=rbc, zbs=zbs, raxis_c=raxis, zaxis_s=zaxis)


class Meter:
    """Wall seconds, JAX compile seconds and peak GPU memory of one step."""

    def __init__(self):
        import jax

        self.compile = 0.0
        jax.monitoring.register_event_duration_secs_listener(self._listen)
        self.t0 = time.perf_counter()

    def _listen(self, event, duration, **_):
        if event.startswith("/jax/core/compile/"):
            self.compile += duration

    def __call__(self):
        import jax

        stats = jax.devices()[0].memory_stats() or {}
        return dict(seconds=round(time.perf_counter() - self.t0, 1), compile_seconds=round(self.compile, 1),
                    peak_gpu_gib=round(stats.get("peak_bytes_in_use", 0) / 2**30, 2), device=str(jax.devices()[0]))


@functools.cache
def _filament_class():
    """The filament field type, made once (JAX is imported lazily: ``table`` needs no JAX)."""
    import jax
    import jax.numpy as jnp

    @dataclasses.dataclass(frozen=True)
    class FilamentField:
        gamma: object  # (coils, points, 3)
        gamma_dash: object
        currents: object  # (coils,)

        def b_cyl(self, r, phi, z):
            xyz = jnp.stack(jnp.broadcast_arrays(r * jnp.cos(phi), r * jnp.sin(phi), z), axis=-1)
            d = xyz[..., None, None, :] - self.gamma
            dB = jnp.cross(self.gamma_dash, d) * (jnp.sum(d * d, axis=-1) ** -1.5)[..., None]
            B = 1e-7 * jnp.mean(jnp.sum(dB * self.currents[:, None, None], axis=-3), axis=-2)
            cos, sin = jnp.cos(phi), jnp.sin(phi)
            return cos * B[..., 0] + sin * B[..., 1], -sin * B[..., 0] + cos * B[..., 1], B[..., 2]

    jax.tree_util.register_dataclass(FilamentField, data_fields=["gamma", "gamma_dash", "currents"],
                                     meta_fields=[])
    return FilamentField


def filament_field(gamma, gamma_dash, currents):
    """Biot-Savart field B = 1e-7 sum_c I_c mean_p gamma'_cp x d / |d|^3 of filaments, as a pytree with ``b_cyl``.

    VMEX holds no coil code; any field with ``b_cyl(r, phi, z)`` whose arrays are pytree leaves serves the
    three-term model without recompiling.
    """
    import jax.numpy as jnp

    return _filament_class()(jnp.asarray(gamma), jnp.asarray(gamma_dash), jnp.asarray(currents))


def load_field(out):
    data = np.load(out / "field.npz")
    return filament_field(data["gamma"], data["gamma_dash"], data["currents"])


# ---- field: the 12 x 12 target and the winding-surface current fitted to it ------------------------------------
def make_field(out):
    import jax
    import jax.numpy as jnp
    import vmex as vj
    from vmex import optimize as opt
    from vmex.core import virtual_casing as vc

    meter = Meter()
    inp = deck(TRUTH_MODES)
    nfp = int(inp.nfp)
    eq = opt.solve_equilibrium(inp)
    w = eq.wout
    vj.write_wout(str(out / "truth.nc"), w)
    data = vc.surface_field_data_from_state(inp, eq.solution, runtime=eq.solver_context, nphi=VC_GRID,
                                            ntheta=VC_GRID)
    iface = vc.PlasmaVacuumInterface.from_surface_data(data)
    targets = jnp.asarray(np.moveaxis(np.asarray(data.gamma), 0, -1).reshape(-1, 3))
    normals = jnp.asarray(np.moveaxis(np.asarray(data.normal), 0, -1).reshape(-1, 3))
    bn_plasma = np.asarray(iface.Bn_plasma).reshape(-1)
    B_mean = float(np.mean(np.linalg.norm(np.asarray(data.B_total), axis=0)))
    xm, xn = np.asarray(w.xm, float), np.asarray(w.xn, float)
    rmnc, zmns = np.asarray(w.rmnc)[-1], np.asarray(w.zmns)[-1]

    def surface(theta, phi, rc, zs, mm, nn):
        a = mm * theta[..., None] - nn * phi[..., None]
        sin, cos = np.sin(a), np.cos(a)
        return ((rc * cos).sum(-1), (zs * sin).sum(-1), (-rc * mm * sin).sum(-1), (rc * nn * sin).sum(-1),
                (zs * mm * cos).sum(-1), (-zs * nn * cos).sum(-1))

    def xyz(R, Z, phi, Rt, Rp, Zt, Zp):
        c, s = np.cos(phi), np.sin(phi)
        return (np.stack([R * c, R * s, Z], -1), np.stack([Rt * c, Rt * s, Zt], -1),
                np.stack([Rp * c - R * s, Rp * s + R * c, Zp], -1))

    # Winding surface: the target offset outward by 1.2 minor radii, smoothed to m <= 8, |n| <= 8.
    th, ph = np.meshgrid(np.linspace(0, 2 * np.pi, 96, endpoint=False),
                         np.linspace(0, 2 * np.pi / nfp, 96, endpoint=False), indexing="ij")
    R, Z, Rt, Rp, Zt, Zp = surface(th, ph, rmnc, zmns, xm, xn)
    points, e_theta, e_phi = xyz(R, Z, ph, Rt, Rp, Zt, Zp)
    normal = -np.cross(e_theta, e_phi)  # VMEC's e_theta x e_phi points inward
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    offset = points + 1.2 * 0.5 * (R.max() - R.min()) * normal
    Ro, Zo = np.hypot(offset[..., 0], offset[..., 1]), offset[..., 2]
    wm, wn, rc_w, zs_w = [], [], [], []
    for m in range(9):
        for n in range(-8, 9):
            if m == 0 and n < 0:
                continue
            a = m * th - n * nfp * ph
            norm = th.size * (1.0 if m == 0 and n == 0 else 0.5)
            wm.append(m), wn.append(n * nfp), rc_w.append(np.sum(Ro * np.cos(a)) / norm)
            zs_w.append(np.sum(Zo * np.sin(a)) / norm)
    wm, wn, rc_w, zs_w = map(np.asarray, (wm, wn, rc_w, zs_w))
    # Sources: 64 x 64 points per field period on every period.
    th, ph = np.meshgrid(np.linspace(0, 2 * np.pi, 64, endpoint=False),
                         np.linspace(0, 2 * np.pi / nfp, 64, endpoint=False), indexing="ij")
    th = np.concatenate([th] * nfp, 1)
    ph = np.concatenate([ph + 2 * np.pi * k / nfp for k in range(nfp)], 1)
    R, Z, Rt, Rp, Zt, Zp = surface(th, ph, rc_w, zs_w, wm, wn)
    src, et_w, ep_w = (jnp.asarray(a.reshape(-1, 3)) for a in xyz(R, Z, ph, Rt, Rp, Zt, Zp))
    th, ph = th.reshape(-1), ph.reshape(-1)
    dA = (2 * np.pi) ** 2 / th.size

    @jax.jit
    def b_sheet(points, phi_t, phi_p):
        """Field of the surface current phi_t e_phi - phi_p e_theta (mu0 / 4 pi absorbed)."""
        K = phi_t[:, None] * ep_w - phi_p[:, None] * et_w
        d = points[:, None, :] - src[None]
        return jnp.sum(jnp.cross(K[None], d) / (jnp.sum(d * d, -1) ** 1.5)[..., None], 1) * dA / (4 * np.pi)

    zeros, ones = jnp.zeros(th.size), jnp.ones(th.size)
    # The secular term's scale sets the net poloidal current G to the target's (edge bvco).
    phi = 2 * np.pi * np.arange(720) / 720
    nax = np.arange(np.asarray(w.raxis_cc).size) * nfp
    R_ax = np.cos(np.outer(phi, nax)) @ np.asarray(w.raxis_cc)
    Z_ax = -np.sin(np.outer(phi, nax)) @ np.asarray(w.zaxis_cs)
    axis = np.stack([R_ax * np.cos(phi), R_ax * np.sin(phi), Z_ax], -1)
    G_unit = float(np.sum(np.asarray(b_sheet(jnp.asarray(axis), zeros, ones)) * np.gradient(axis, phi, axis=0))
                   * (phi[1] - phi[0]) / (2 * np.pi))
    bvco = np.asarray(w.bvco)
    scale = (1.5 * bvco[-1] - 0.5 * bvco[-2]) / G_unit

    def bn(phi_t, phi_p):
        return np.asarray(jnp.sum(b_sheet(targets, phi_t, phi_p) * normals, -1))

    modes = [(m, n) for m in range(15) for n in range(-12, 13) if not (m == 0 and n <= 0)]
    A = np.stack([bn(jnp.asarray(m * np.cos(m * th - n * nfp * ph)), jnp.asarray(-n * nfp * np.cos(m * th - n * nfp * ph)))
                  for m, n in modes], 1)
    rhs = -bn_plasma - scale * bn(zeros, ones)
    best = None
    for rel in (1e-8, 1e-10, 1e-12, 1e-14):
        lam = rel * np.linalg.norm(A, ord="fro") ** 2 / A.shape[1]
        c = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ rhs)
        rms = float(np.sqrt(np.mean((A @ c - rhs) ** 2)))
        if best is None or rms < best[1]:
            best = (c, rms)
    c, rms = best
    phi_t, phi_p = np.zeros(th.size), np.full(th.size, scale)
    for (m, n), cm in zip(modes, c):
        a = m * th - n * nfp * ph
        phi_t += cm * m * np.cos(a)
        phi_p += cm * (-n * nfp) * np.cos(a)
    K = np.asarray(jnp.asarray(phi_t)[:, None] * ep_w - jnp.asarray(phi_p)[:, None] * et_w)
    # As filaments (filament_field: B = 1e-7 sum_c I_c gamma'_c x d / |d|^3): one one-point "coil" per element.
    gamma, gamma_dash = np.asarray(src)[:, None, :], (K * dA)[:, None, :]
    currents = np.full(th.size, 1.0 / (4 * np.pi * 1e-7))
    rng = np.random.default_rng(0)
    check = np.c_[1.0 + 0.3 * rng.uniform(-1, 1, 64), 0.3 * rng.uniform(-1, 1, 64), 0.3 * rng.uniform(-1, 1, 64)]
    d = check[:, None] - gamma[None, :, 0]
    check_B = 1e-7 * np.sum(currents[None, :, None] * np.cross(gamma_dash[None, :, 0], d)
                            / (np.sum(d * d, -1) ** 1.5)[..., None], 1)
    np.savez(out / "field.npz", gamma=gamma, gamma_dash=gamma_dash, currents=currents, check_points=check,
             check_B=check_B)
    report = dict(fit_rms_bn_over_B=rms / B_mean, fit_modes=len(modes), sources=int(th.size), **meter())
    (out / "field.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report), flush=True)


# ---- scoring ------------------------------------------------------------------------------------------------------
def lcfs_mm(wa, wb, nfp):
    """Largest distance of ``wa``'s LCFS to ``wb``'s in four planes of a half period, in mm."""
    theta = np.linspace(0, 2 * np.pi, 721)
    worst = 0.0
    for phi in np.linspace(0, np.pi / nfp, 4):
        curves = []
        for w in (wa, wb):
            a = np.outer(theta, np.asarray(w.xm, float)) - np.asarray(w.xn, float) * phi
            curves.append((np.cos(a) @ np.asarray(w.rmnc)[-1], np.sin(a) @ np.asarray(w.zmns)[-1]))
        (Ra, Za), (Rb, Zb) = curves
        worst = max(worst, float(np.max(np.min(np.hypot(Ra[:, None] - Rb[None], Za[:, None] - Zb[None]), axis=1))))
    return round(1e3 * worst, 3)


def score(modes, out, name, eq, extra, inp=None):
    """Scores of a fixed-boundary equilibrium ``eq`` of ``inp`` (default ``deck(modes)``), the boundary a method found."""
    import vmex as vj
    from vmex import optimize as opt
    from vmex.core import virtual_casing as vc
    from vmex.core.freeboundary_vc import boundary_residual, summarize_boundary_residual

    inp = deck(modes) if inp is None else inp
    w = eq.wout
    surface = vc.surface_field_data_from_state(inp, eq.solution, runtime=eq.solver_context, nphi=VC_GRID,
                                               ntheta=VC_GRID)
    plan = vc.plan_vc_precision(surface, digits=4, quad_nt=4 * int(inp.nfp) * VC_GRID, quad_np=2 * VC_GRID)
    res = summarize_boundary_residual(*boundary_residual(inp, eq.solution, load_field(out),
                                                         runtime=eq.solver_context, nphi=VC_GRID,
                                                         ntheta=VC_GRID, precision=plan))
    q = np.asarray(opt.QuasisymmetryRatioResidual(QA_SURFACES, 1, 0).residuals_state(eq.solution, eq.solver_context))
    nfp = int(inp.nfp)
    row = dict(method=name, modes=modes, ns=NS, force_fsq=float(w.fsqr + w.fsqz + w.fsql), bn=res.normal,
               pressure_balance=res.pressure, sheet_current=res.sheet_current, qa=float(q @ q),
               iota_axis=float(abs(np.asarray(w.iotaf)[0])), iota_edge=float(abs(np.asarray(w.iotaf)[-1])),
               aspect=float(w.aspect), volume_m3=float(w.volume_p),
               lcfs_mm_vs_truth=lcfs_mm(w, vj.read_wout(str(out / "truth.nc")), nfp), **extra)
    target = out / f"m{modes}" / "wout_target.nc"
    if target.exists():
        row["lcfs_mm_vs_target"] = lcfs_mm(w, vj.read_wout(str(target)), nfp)
    vj.write_wout(str(out / f"m{modes}" / f"wout_{name}.nc"), w)
    (out / f"m{modes}" / f"{name}.json").write_text(json.dumps(row, indent=1) + "\n")
    print("ROW " + json.dumps(row), flush=True)
    return row


def no_equilibrium(modes, out, name, error, extra):
    """The row of a method whose final boundary has no fixed-boundary equilibrium to score."""
    row = dict(method=name, modes=modes, ns=NS, no_equilibrium=f"{type(error).__name__}: {error}"[:200], **extra)
    (out / f"m{modes}" / f"{name}.json").write_text(json.dumps(row, indent=1) + "\n")
    print("ROW " + json.dumps(row), flush=True)


# ---- methods ------------------------------------------------------------------------------------------------------
def run_target(modes, out):
    import vmex as vj
    from vmex import optimize as opt

    meter = Meter()
    inp = deck(modes)
    eq = opt.solve_equilibrium(inp)
    score(modes, out, "target", eq, meter())
    eq_start = opt.solve_equilibrium(start_boundary(inp))  # the initial guess, for DESC
    vj.write_wout(str(out / f"m{modes}" / "wout_start.nc"), eq_start.wout)
    score(modes, out, "start", eq_start, {})


def run_nestor(modes, out):
    import vmex as vj
    from vmex import optimize as opt

    meter = Meter()
    inp = start_boundary(deck(modes))
    free = replace(inp, lfreeb=True, mgrid_file="field(direct)", ns_array=np.array([11, NS]),
                   ftol_array=np.array([1e-10, 1e-13]), niter_array=np.array([20000, 40000]))
    result = vj.solve_free_boundary_multigrid(free, external_field=load_field(out), raise_on_max_iterations=False)
    cost = meter()
    w = vj.wout_from_state(inp=free, state=result.state, fsqr=float(result.fsqr), fsqz=float(result.fsqz),
                           fsql=float(result.fsql), niter=int(result.iterations), converged=bool(result.converged),
                           vacuum_output=result.vacuum)
    fixed = replace(with_wout_boundary(deck(modes), w), lfreeb=False, mgrid_file="NONE")
    eq = opt.solve_equilibrium(fixed, initial_state=result.state)
    score(modes, out, "nestor", eq, dict(free_fsq=float(result.fsqr + result.fsqz + result.fsql),
                                         iterations=int(result.iterations), converged=bool(result.converged),
                                         **cost))


def mgrid_nzeta(modes):
    """Planes per field period of the mgrid decks: a divisor of the table's 48 at or above 2 ntor + 6."""
    return next(k for k in (16, 24, 48) if k >= 2 * modes + 6)


def make_mgrid(out):
    import jax
    import vmex as vj
    from vmex.core import virtual_casing as vc
    from vmex.core.mgrid import tabulate_cartesian_field, write_mgrid

    meter = Meter()
    field = load_field(out)
    w = vj.read_wout(str(out / "truth.nc"))
    theta, phi = np.meshgrid(np.linspace(0, 2 * np.pi, 181), np.linspace(0, 2 * np.pi, 181), indexing="ij")
    a = theta[..., None] * np.asarray(w.xm, float) - phi[..., None] * np.asarray(w.xn, float)
    R, Z = np.cos(a) @ np.asarray(w.rmnc)[-1], np.sin(a) @ np.asarray(w.zmns)[-1]
    minor = 0.5 * (R.max() - R.min())
    B = jax.jit(lambda xyz: vc.external_B_cartesian(field, xyz.T[:, :, None])[:, :, 0].T)

    def bxyz(points):  # chunks of 20,000 points: 8,192 sources each
        points = np.asarray(points)
        return np.concatenate([np.asarray(B(points[i:i + 20000])) for i in range(0, len(points), 20000)])

    data = tabulate_cartesian_field(bxyz, rmin=R.min() - 0.5 * minor, rmax=R.max() + 0.5 * minor,
                                    zmin=Z.min() - 0.5 * minor, zmax=Z.max() + 0.5 * minor, ir=301, jz=301, kp=48,
                                    nfp=int(w.nfp), label="fitted_sheet")
    write_mgrid(out / "mgrid_field.nc", data)
    report = dict(ir=301, jz=301, kp=48, rmin=data.rmin, rmax=data.rmax, zmin=data.zmin, zmax=data.zmax, **meter())
    (out / "mgrid.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report), flush=True)


def mgrid_deck(modes, out):
    """The free-boundary deck on the mgrid table, from the common start: VMEX + NESTOR's ladder and tolerances."""
    inp = start_boundary(deck(modes))
    return replace(inp, lfreeb=True, mgrid_file=str(out / "mgrid_field.nc"), extcur=np.array([1.0]), nvacskip=6,
                   nzeta=mgrid_nzeta(modes), ns_array=np.array([11, NS]), ftol_array=np.array([1e-10, 1e-13]),
                   niter_array=np.array([20000, 40000]))


def run_vmec2000(modes, out):
    import shlex
    import subprocess

    run = out / f"m{modes}" / "vmec2000"
    run.mkdir(exist_ok=True)
    free = replace(mgrid_deck(modes, out), mgrid_file="mgrid_field.nc", nstep=500)
    free.to_indata(run / "input.case")
    link = run / "mgrid_field.nc"
    if not link.exists():
        link.symlink_to((out / "mgrid_field.nc").resolve())
    started = time.perf_counter()
    with open(run / "log.txt", "w") as log:
        code = subprocess.call(shlex.split(os.environ["VMEC2000_CMD"]) + ["input.case"], cwd=run, stdout=log,
                               stderr=subprocess.STDOUT)
    seconds = round(time.perf_counter() - started, 1)
    import netCDF4

    report = dict(seconds=seconds, exit_code=code, device="cpu (" + os.environ["VMEC2000_CMD"] + ")")
    with netCDF4.Dataset(run / "wout_case.nc") as nc:  # ier_flag 0: converged to FTOL
        report.update(converged=int(nc["ier_flag"][:]) == 0, free_fsq=float(nc["fsqr"][:] + nc["fsqz"][:]
                                                                             + nc["fsql"][:]))
    (run / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    run_score(modes, out, "vmec2000", str(run / "wout_case.nc"))


def run_nestor_mgrid(modes, out):
    import vmex as vj
    from vmex import optimize as opt
    from vmex.core.mgrid import MgridField

    meter = Meter()
    free = mgrid_deck(modes, out)
    result = vj.solve_free_boundary_multigrid(free, external_field=MgridField.from_file(free.mgrid_file),
                                              raise_on_max_iterations=False)
    cost = meter()
    w = vj.wout_from_state(inp=free, state=result.state, fsqr=float(result.fsqr), fsqz=float(result.fsqz),
                           fsql=float(result.fsql), niter=int(result.iterations), converged=bool(result.converged),
                           vacuum_output=result.vacuum)
    fixed = replace(with_wout_boundary(deck(modes), w), lfreeb=False, mgrid_file="NONE")
    extra = dict(free_fsq=float(result.fsqr + result.fsqz + result.fsql), iterations=int(result.iterations),
                 converged=bool(result.converged), **cost)
    try:
        eq = opt.solve_equilibrium(fixed, initial_state=result.state)
    except vj.VmecError as error:  # the free boundary it stopped at has no equilibrium: a row without scores
        no_equilibrium(modes, out, "nestor_mgrid", error, extra)
        return
    score(modes, out, "nestor_mgrid", eq, extra)


def run_three_term(modes, out, chunk):
    from vmex.core.freeboundary_vc import solve_free_boundary_three_term

    meter = Meter()
    inp = replace(deck(modes), lfreeb=True, mgrid_file="field(direct)")
    fit = solve_free_boundary_three_term(inp, external_field=load_field(out), initial_boundary=start_boundary(inp),
                                         nphi=VC_GRID, ntheta=VC_GRID, max_nfev=60, trial_ftol=1e-11, chunk=chunk,
                                         quadrature=(4 * int(inp.nfp) * VC_GRID, 2 * VC_GRID), verbose=2)
    score(modes, out, "three_term", fit.equilibrium, dict(nfev=int(fit.nfev), njev=int(fit.njev),
                                                          cost=float(fit.cost), **meter()))


def run_score(modes, out, name, path):
    import vmex as vj
    from vmex import optimize as opt
    from vmex.core.restart import restart_state

    w = vj.read_wout(path)
    inp = deck(modes)
    mpol = int(np.max(w.xm)) + 1  # DESC's M includes m = M: keep every mode, not VMEC's m < mpol = M
    if mpol > inp.mpol:
        inp = inp.change_resolution(mpol=mpol, ntor=inp.ntor, ntheta=2 * mpol + 6, nzeta=inp.nzeta)
    inp = with_wout_boundary(inp, w)
    extra = {}
    report = Path(path).with_name("report.json")
    if report.exists():  # the other code's own time and memory
        data = json.loads(report.read_text())
        extra = {k: data[k] for k in REPORT_KEYS if k in data}
        if "block_rms" in data:  # the code's own interface residuals, for reference
            extra["own_residual_rms"] = data["block_rms"]
    try:
        state = restart_state(path, inp)
    except Exception:  # noqa: BLE001  (a wout another code wrote may not restart; a cold solve still scores it)
        state = None
    try:
        eq = opt.solve_equilibrium(inp, initial_state=state)
    except vj.VmecError as error:
        no_equilibrium(modes, out, name, error, extra)
        return
    score(modes, out, name, eq, extra, inp)


def table(out):
    import types

    import netCDF4

    def boundary(path):  # the LCFS arrays lcfs_mm reads, without importing VMEX
        with netCDF4.Dataset(path) as nc:
            return types.SimpleNamespace(**{k: np.asarray(nc[k][:]) for k in ("xm", "xn", "rmnc", "zmns", "nfp")})

    rows = [json.loads(p.read_text()) for p in sorted(out.glob("m*/*.json"))]
    rows.sort(key=lambda r: (r["modes"], r["method"]))
    for r in rows:  # a method that finished before its target: the distance from the saved wouts
        target, own = (out / f"m{r['modes']}" / f"wout_{k}.nc" for k in ("target", r["method"]))
        if "lcfs_mm_vs_target" not in r and target.exists() and own.exists():
            w = boundary(own)
            r["lcfs_mm_vs_target"] = lcfs_mm(w, boundary(target), int(w.nfp))
    field = json.loads((out / "field.json").read_text())
    (out / "three_term_resolution.json").write_text(json.dumps(dict(
        case=str(DECK.relative_to(REPO)), ns=NS, truth_modes=TRUTH_MODES, start_scale=START_SCALE,
        vc_grid=VC_GRID, field=field, rows=rows), indent=1) + "\n")
    columns = {"modes": "M = N", "method": "method", "seconds": "wall s", "peak_gpu_gib": "peak GiB",
               "force_fsq": "force fsq", "bn": "B.n / |B|", "pressure_balance": "pressure jump",
               "sheet_current": "sheet current mu0 K / |B|", "qa": "QA", "iota_axis": "iota axis",
               "iota_edge": "iota edge", "lcfs_mm_vs_truth": "LCFS to 12x12 target, mm",
               "lcfs_mm_vs_target": "LCFS to target at M, mm"}
    keys = tuple(columns)
    print("| " + " | ".join(columns.values()) + " |")
    print("|" + " --- |" * len(keys))
    for r in rows:
        print("| " + " | ".join(_fmt(r.get(k)) for k in keys) + " |")


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) >= 1e-2 or v == 0 else f"{v:.2e}"
    return "" if v is None else str(v)


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=("field", "mgrid", "target", "nestor", "nestor_mgrid", "vmec2000", "three_term",
                                     "score", "table"))
    p.add_argument("args", nargs="+", help="[M] OUT [NAME WOUT]")
    p.add_argument("--chunk", type=int, help="Jacobian columns per batch (default 8, 4 above 8 modes, 2 at 12)")
    p.add_argument("--case", choices=tuple(CASES), help="the deck (set by the field step and kept in OUT/case.txt)")
    a = p.parse_args(argv)
    modes = None if a.step in ("field", "mgrid", "table") else int(a.args[0])
    out = Path(a.args[0 if modes is None else 1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    global DECK
    saved = out / "case.txt"
    case = saved.read_text().strip() if saved.exists() else (a.case or "beta")
    if a.case and a.case != case:
        raise SystemExit(f"{out} holds the {case} case, not {a.case}")
    if a.step == "field":
        saved.write_text(case + "\n")
    DECK = CASES[case]
    if modes is not None:
        (out / f"m{modes}").mkdir(exist_ok=True)
    if a.step == "field":
        make_field(out)
    elif a.step == "mgrid":
        make_mgrid(out)
    elif a.step == "vmec2000":
        run_vmec2000(modes, out)
    elif a.step == "nestor_mgrid":
        run_nestor_mgrid(modes, out)
    elif a.step == "target":
        run_target(modes, out)
    elif a.step == "nestor":
        run_nestor(modes, out)
    elif a.step == "three_term":
        run_three_term(modes, out, a.chunk or (8 if modes <= 8 else 4 if modes <= 10 else 2))
    elif a.step == "score":
        run_score(modes, out, a.args[2], a.args[3])
    else:
        table(out)


if __name__ == "__main__":
    main()
