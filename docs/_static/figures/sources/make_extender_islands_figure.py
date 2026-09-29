#!/usr/bin/env python3
"""Draw the extended-field Poincare figures from ``benchmarks/extender_islands_sections.npz``.

Everything comes from VMEX and the inputs shipped in ``examples/data``: the
Landreman-Paul QA deck (``input.LandremanPaul2021_QA_lowres``) held by its
ESSOS coils (``ESSOS_biot_savart_LandremanPaulQA.json``), solved as a free
boundary at fixed coil currents with ``p = PRES_SCALE (1 - s)``, as in
``examples/free_boundary_essos_coils.py``. Three cases: vacuum and two finite
pressures.

The field outside the plasma is the coil Biot-Savart field plus the
``VmecExtender`` virtual-casing field of the plasma currents. The plasma part
is tabulated once on a cylindrical grid (half a period, completed by
stellarator symmetry) and read through a tricubic ``MgridField``. Nodes inside
the plasma or within ``SURFACE_GAP`` of it, and nodes where the quadrature
error estimate exceeds 1e-5, are filled by neighbour averaging, a smooth
continuation of the exterior field. Field lines launched on the phi = 0
outboard midplane from 4 mm to 4.5 cm outside the LCFS are traced through coils
+ table; the VMEX flux surfaces fill the plasma.

Usage::

    python docs/_static/figures/sources/make_extender_islands_figure.py            # draw
    python docs/_static/figures/sources/make_extender_islands_figure.py --record   # recompute (ESSOS, about 1 h CPU)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
DATA = REPO / "examples" / "data"
RECORD = REPO / "benchmarks" / "extender_islands_sections.npz"
FIGURES = REPO / "docs" / "_static" / "figures"
CASES = {"vacuum": 0.0, "beta1": 1.0 / 1.45e-3, "beta2": 2.0 / 1.45e-3}   # PRES_SCALE [Pa]
MPOL, NTOR, NS, PHIEDGE = 5, 5, 51, -0.025
COIL_GRID = dict(rmin=0.45, rmax=1.55, zmin=-0.6, zmax=0.6, ir=96, jz=96, kp=32)
N_OUTSIDE, OUTSIDE_SPAN = 24, 0.045                # seeds between 4 mm and this far [m] past the LCFS
LENGTH, SAMPLES = 1500.0, 40000                    # arclength [m], about 240 transits
GRID_SPACING, GRID_MARGIN, KP = 0.01, 0.12, 24     # table spacing [m], margin [m], planes/period
SURFACE_GAP = 0.003                                # table nodes closer to the LCFS are continued
SURFACES = (0.04, 0.16, 0.36, 0.64, 1.0)


def solve(pres_scale: float):
    """Free-boundary LP-QA at fixed coil currents; returns (input, result, wout, coils)."""
    from dataclasses import replace

    import vmex as vj
    from essos.coils import Coils

    coils = Coils.from_json(str(DATA / "ESSOS_biot_savart_LandremanPaulQA.json"))
    inp = vj.VmecInput.from_file(DATA / "input.LandremanPaul2021_QA_lowres")
    k = inp.ntor - NTOR
    inp = replace(
        inp, lfreeb=True, mgrid_file="essos_coils(direct)", mpol=MPOL, ntor=NTOR, nzeta=16,
        rbc=inp.rbc[k:k + 2 * NTOR + 1, :MPOL], zbs=inp.zbs[k:k + 2 * NTOR + 1, :MPOL],
        rbs=inp.rbs[k:k + 2 * NTOR + 1, :MPOL], zbc=inp.zbc[k:k + 2 * NTOR + 1, :MPOL],
        raxis_c=inp.raxis_c[:NTOR + 1], zaxis_s=inp.zaxis_s[:NTOR + 1],
        raxis_s=inp.raxis_s[:NTOR + 1], zaxis_c=inp.zaxis_c[:NTOR + 1],
        phiedge=PHIEDGE, ns_array=[NS], niter_array=[20000], ftol_array=[1e-12],
        pmass_type="power_series", am=[1.0, -1.0] + [0.0] * 19, pres_scale=pres_scale)
    res = vj.solve_free_boundary(inp, external_field=vj.MgridField.from_coils(coils, **COIL_GRID))
    wout = vj.wout_from_state(inp=inp, state=res.state, fsqr=float(res.fsqr), fsqz=float(res.fsqz),
                              fsql=float(res.fsql), niter=int(res.iterations),
                              converged=bool(res.converged), vacuum_output=res.vacuum)
    return inp, res, wout, coils


def boundary_input(inp, wout):
    """``inp`` with RBC/ZBS set to the solved LCFS."""
    from dataclasses import replace

    rbc, zbs = np.zeros_like(inp.rbc), np.zeros_like(inp.zbs)
    n = (np.asarray(wout.xn, dtype=float) / float(wout.nfp)).astype(int)
    for j, (m, nn) in enumerate(zip(np.asarray(wout.xm, dtype=int), n)):
        if m < inp.mpol and abs(nn) <= inp.ntor:
            rbc[nn + inp.ntor, m] = np.asarray(wout.rmnc)[-1][j]
            zbs[nn + inp.ntor, m] = np.asarray(wout.zmns)[-1][j]
    return replace(inp, rbc=rbc, zbs=zbs)


def record_case(name: str) -> dict:
    """Solve, build the extended field, trace, and return the phi = 0 sections."""
    import jax
    import jax.numpy as jnp

    import vmex as vj
    from vmex.core.mgrid import MgridData, MgridField
    from vmex.core.plotting import surface_rz
    from essos.dynamics import LevelsetStoppingCriterion, trace_field_lines
    from essos.fields import BiotSavart
    from essos.surfaces import SurfaceClassifier, surfacerzfourier_from_boundary

    jax.config.update("jax_enable_x64", True)
    inp, res, wout, coil_set = solve(CASES[name])
    print(f"{name}: beta {float(wout.betatotal):.4%}, converged {bool(res.converged)}, "
          f"fsq {float(res.fsqr) + float(res.fsqz) + float(res.fsql):.1e}", flush=True)
    coils = BiotSavart(coil_set)
    coil_B = jax.jit(lambda p: jax.vmap(coils.B)(p.reshape(-1, 3)).reshape(p.shape))
    edge_inp = boundary_input(inp, wout)
    lcfs = surfacerzfourier_from_boundary(edge_inp.rbc, edge_inp.zbs, inp.nfp, nphi=64, ntheta=64)
    gamma = np.asarray(lcfs.gamma)
    R_lcfs, Z_lcfs = np.hypot(gamma[..., 0], gamma[..., 1]), gamma[..., 2]
    classifier = SurfaceClassifier(lcfs, h=0.02, padding=GRID_MARGIN + 0.05)

    if CASES[name] > 0:  # finite beta: coils + tabulated plasma-current field
        ext = vj.VmecExtender.from_state(inp, res.state, external_field=coil_B)

        def distance(xyz: np.ndarray) -> np.ndarray:  # signed distance to the LCFS, positive inside
            return np.concatenate([np.asarray(jax.vmap(classifier.evaluate_xyz)(jnp.asarray(xyz[i:i + 20000]))).ravel()
                                   for i in range(0, len(xyz), 20000)])

        def plasma_part(xyz: np.ndarray) -> np.ndarray:
            """Plasma-current field outside the LCFS; NaN inside it and wherever the quadrature misses."""
            out = np.full_like(xyz, np.nan)
            outside = np.flatnonzero(distance(xyz) < -SURFACE_GAP)
            for i in range(0, len(outside), 4096):
                idx = outside[i:i + 4096]
                p = jnp.asarray(xyz[idx])
                value = np.asarray(ext.B(p)) - np.asarray(coil_B(p))
                value[np.asarray(ext.B_error_estimate(p)) > 1e-5] = np.nan
                out[idx] = value
                print(f"  tabulated {i + len(idx)}/{len(outside)}", flush=True)
            return out

        rmin, rmax = R_lcfs.min() - GRID_MARGIN, R_lcfs.max() + GRID_MARGIN
        zmax = np.abs(Z_lcfs).max() + GRID_MARGIN
        ir, jz, nfp = int((rmax - rmin) / GRID_SPACING) + 1, int(2 * zmax / GRID_SPACING) + 1, int(inp.nfp)
        r, z = np.linspace(rmin, rmax, ir), np.linspace(-zmax, zmax, jz)
        phi = np.arange(KP // 2 + 1) * 2 * np.pi / (nfp * KP)  # half period; stellarator symmetry fills the rest
        pp, zz, rr = np.meshgrid(phi, z, r, indexing="ij")
        nodes = np.stack((rr * np.cos(pp), rr * np.sin(pp), zz), -1).reshape(-1, 3)
        cache = Path(tempfile.gettempdir()) / f"vmex_extender_table_{name}.npy"
        if cache.exists():
            b = np.load(cache)
        else:
            b = plasma_part(nodes).reshape(*pp.shape, 3)
            np.save(cache, b)
        print(f"continuing {int(np.isnan(b[..., 0]).sum())} of {b[..., 0].size} table nodes", flush=True)
        while np.isnan(b).any():
            pad = np.pad(b, ((0, 0), (1, 1), (1, 1), (0, 0)), constant_values=np.nan)
            ring = np.stack((pad[:, :-2, 1:-1], pad[:, 2:, 1:-1], pad[:, 1:-1, :-2], pad[:, 1:-1, 2:]))
            with np.errstate(all="ignore"):
                b = np.where(np.isnan(b), np.nanmean(ring, axis=0), b)
        half = (b[..., 0] * np.cos(pp) + b[..., 1] * np.sin(pp), -b[..., 0] * np.sin(pp) + b[..., 1] * np.cos(pp), b[..., 2])
        mirror = np.arange(KP // 2 - 1, 0, -1)   # plane KP - k is plane -k: (B_R, B_phi, B_Z)(R, -phi, -Z) = (-B_R, B_phi, B_Z)
        br, bp, bz = (np.concatenate((c, sign * c[mirror, ::-1])) for c, sign in zip(half, (-1, 1, 1)))
        table = MgridData(rmin=rmin, rmax=rmax, zmin=-zmax, zmax=zmax, ir=ir, jz=jz, kp=KP, nfp=nfp, nextcur=1,
                          mgrid_mode="S", coil_groups=("vmex_plasma_current",), raw_coil_cur=(1.0,),
                          br=br[None], bp=bp[None], bz=bz[None])
        plasma = MgridField.from_mgrid_data(table, order=3)

        class Total:
            def B_contravariant(self, x):
                r, phi = jnp.hypot(x[0], x[1]), jnp.arctan2(x[1], x[0])
                br, bp, bz = plasma.b_cyl(r, phi, x[2])
                c, s = jnp.cos(phi), jnp.sin(phi)
                return coils.B(x) + jnp.stack((br * c - bp * s, br * s + bp * c, bz))

            def B(self, x):
                return self.B_contravariant(x)

            def AbsB(self, x):
                return jnp.linalg.norm(self.B_contravariant(x))

            def to_xyz(self, x):
                return x

        field = Total()
    else:  # vacuum: the extended field is the coil field
        field = coils

    theta = np.linspace(0, 2 * np.pi, 256)
    R_out = float(surface_rz(wout, s_index=-1, theta=np.array([0.0]), phi=np.array([0.0]))[0][0, 0])
    offsets = np.linspace(0.004, OUTSIDE_SPAN, N_OUTSIDE)
    seeds = jnp.asarray(np.stack((R_out + offsets, 0 * offsets, 0 * offsets), 1))
    escape = LevelsetStoppingCriterion(classifier, maximum_distance=GRID_MARGIN - 0.02)
    trace = trace_field_lines(field, seeds, length=LENGTH, samples=SAMPLES, tolerance=1e-9,
                              stopping_criteria=escape, progress=False)
    sections = trace.poincare_plot(shifts=[0.0], show=False)
    plt.close("all")
    ns = np.asarray(wout.rmnc).shape[0]
    cuts = [surface_rz(wout, s_index=int(round(s * (ns - 1))), theta=theta, phi=np.array([0.0])) for s in SURFACES]
    out = {f"{name}_beta": float(wout.betatotal), f"{name}_offsets": offsets,
           f"{name}_counts": np.array([len(sec[0]) for sec in sections]),
           f"{name}_R": np.concatenate([np.asarray(sec[0]) for sec in sections]).astype(np.float32),
           f"{name}_Z": np.concatenate([np.asarray(sec[1]) for sec in sections]).astype(np.float32),
           f"{name}_surf_R": np.array([c[0][:, 0] for c in cuts], np.float32),
           f"{name}_surf_Z": np.array([c[1][:, 0] for c in cuts], np.float32)}
    print(name, "crossings", out[f"{name}_counts"].tolist(), flush=True)
    return out


def panel(ax, d, name: str, title: str) -> None:
    counts, R, Z = d[f"{name}_counts"], d[f"{name}_R"], d[f"{name}_Z"]
    start = np.concatenate(([0], np.cumsum(counts)))
    for i in np.flatnonzero(counts >= 5):  # a line that leaves within a few transits marks only its seed
        ax.scatter(R[start[i]:start[i + 1]], Z[start[i]:start[i + 1]], s=0.5, color="#eb6834", lw=0)
    for i, (r, z) in enumerate(zip(d[f"{name}_surf_R"], d[f"{name}_surf_Z"])):
        ax.plot(r, z, color="#1b1b1b", lw=0.5 + 0.7 * (i == len(SURFACES) - 1))
    ax.set(aspect="equal", xlabel="R [m]", ylabel="Z [m]", title=title)
    ax.grid(alpha=0.2, lw=0.5)


HANDLES = [plt.Line2D([], [], color="#1b1b1b", lw=1, label="VMEX flux surfaces"),
           plt.Line2D([], [], color="#eb6834", marker="o", ls="", ms=3, label="field lines launched outside")]


def draw(d) -> None:
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 4.6), sharey=True)
    for ax, name in zip(axes, ("beta1", "beta2")):
        panel(ax, d, name, f"$\\beta$ = {100 * float(d[name + '_beta']):.2f}%")
    axes[1].set_ylabel("")
    fig.suptitle("Free-boundary QA with its coils, $\\phi$ = 0", fontsize=10)
    fig.legend(handles=HANDLES, loc="lower center", ncol=2, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGURES / "readme_extender_islands.webp", dpi=110, pil_kwargs={"quality": 80, "method": 6})
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(3.2, 4.6))
    panel(ax, d, "vacuum", "Vacuum, $\\phi$ = 0")
    fig.legend(handles=HANDLES, loc="lower center", ncol=1, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(FIGURES / "extender_vacuum_islands.webp", dpi=110, pil_kwargs={"quality": 80, "method": 6})
    plt.close(fig)


def main() -> None:
    if "--record" in sys.argv:
        names = [a for a in sys.argv[1:] if a in CASES] or list(CASES)
        old = {k: v for k, v in np.load(RECORD).items() if k.split("_")[0] in CASES} if RECORD.exists() else {}
        for name in names:
            old = {k: v for k, v in old.items() if not k.startswith(name + "_")} | record_case(name)
            np.savez_compressed(RECORD, **old)
    draw(np.load(RECORD))


if __name__ == "__main__":
    main()
