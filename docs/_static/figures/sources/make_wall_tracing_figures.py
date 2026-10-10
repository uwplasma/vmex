#!/usr/bin/env python3
"""Regenerate the three figures of ``docs/howto/trace-to-the-wall.md``.

- ``wall_tracing_reentry.webp``: signed distance to the LCFS against time for
  one orbit that stays inside, one that crosses and comes back, and one that
  strikes the wall.
- ``wall_tracing_strikes.webp``: where the strikes land on the wall, unrolled
  in toroidal and poloidal angle, coloured by strike time.
- ``wall_tracing_interpolation.webp``: error of the tabulated exterior field
  between the LCFS and the wall against the grid size, and the cost of a trace
  through each table against the direct field.

The setup is ``examples/vmex_essos_tracing_to_wall.py``: the vacuum
Landreman-Paul QA equilibrium, its ESSOS coils, 3 keV protons and a wall 3 cm
outside the LCFS.  About 10 minutes on 8 CPU cores.

Usage::

    python docs/_static/figures/sources/make_wall_tracing_figures.py
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import vmex as vj
from vmex import optimize as opt
from essos.coils import Coils
from essos.constants import ELEMENTARY_CHARGE, ONE_EV, PROTON_MASS
from essos.dynamics import Particles, Tracing
from essos.fields import ExternalField, InterpolatedField
from essos.surfaces import SurfaceClassifier, SurfaceRZFourier

REPO = Path(__file__).resolve().parents[4]
FIGURES = REPO / "docs" / "_static" / "figures"
DATA = REPO / "examples" / "data"
WALL, ENERGY_EV, TMAX, NPARTICLES = 0.03, 3e3, 4e-5, 256
GRIDS = (16, 24, 32, 48, 64)
FATE_COLORS = {"stays inside": "#3b6fb6", "crosses and returns": "#e08a1e", "strikes the wall": "#c0392b"}


def _save(fig, name: str) -> None:
    fig.canvas.draw()
    Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB").quantize(64).save(
        FIGURES / name, lossless=True, method=6)
    plt.close(fig)
    print(f"wrote {FIGURES / name}")


def _particles(n: int, seed: int = 0) -> Particles:
    k = jax.random.split(jax.random.key(seed), 4)
    births = jnp.stack([jax.random.uniform(k[0], (n,), minval=0.7, maxval=0.95),
                        jax.random.uniform(k[1], (n,), maxval=2 * jnp.pi),
                        jax.random.uniform(k[2], (n,), maxval=2 * jnp.pi)], 1)
    return Particles(initial_xyz=births, mass=PROTON_MASS, charge=ELEMENTARY_CHARGE, energy=ENERGY_EV * ONE_EV,
                     initial_vparallel_over_v=jax.random.uniform(k[3], (n,), minval=-1, maxval=1))


def _trace(setup: dict, particles: Particles) -> tuple[Tracing, float]:
    start = perf_counter()
    trace = Tracing(**setup, model="GuidingCenterAdaptative", particles=particles, maxtime=TMAX,
                    times_to_trace=1000, atol=1e-9, rtol=1e-9)
    trace.trajectories.block_until_ready()
    return trace, perf_counter() - start


def main() -> None:
    inp = vj.VmecInput.from_file(DATA / "input.LandremanPaul2021_QA_lowres")
    inp = replace(inp, phiedge=-0.025, ns_array=np.array([51]), ftol_array=np.array([1e-12]),
                  niter_array=np.array([8000]))
    wout = opt.solve_equilibrium(inp, verbose=False).wout
    coils = Coils.from_json(str(DATA / "ESSOS_biot_savart_LandremanPaulQA.json"))
    direct = vj.essos_tracing_fields(wout, coils, wall=WALL, n=None, plasma="vacuum")
    vmec, wall = direct["field"], direct["wall"]
    lcfs = SurfaceRZFourier.from_vmec(vmec, ntheta=64, nphi=128)

    # Interpolation error between the LCFS and the wall, and trace cost
    exterior = ExternalField(direct["exterior_field"])
    shells = [np.asarray(SurfaceRZFourier.from_vmec(vmec, ntheta=24, nphi=48, offset=d).gamma).reshape(-1, 3)
              for d in np.linspace(0.003, WALL, 6)]
    points = jnp.asarray(np.concatenate(shells))
    exact = jax.vmap(exterior.B)(points)
    errors, tabulate, timing = [], [], []
    few = _particles(32, seed=1)
    _, direct_time = _trace(direct, few)
    _, direct_time = _trace(direct, few)  # warm
    for n in GRIDS:
        start = perf_counter()
        table = InterpolatedField.around(direct["exterior_field"], wall, n=n, stellsym=True)
        jax.block_until_ready(table.coefficients)
        tabulate.append(perf_counter() - start)
        rel = np.linalg.norm(jax.vmap(table.B)(points) - exact, axis=1) / np.linalg.norm(exact, axis=1)
        errors.append((np.median(rel), rel.max()))
        setup = dict(direct, exterior_field=table)
        _trace(setup, few)
        timing.append(_trace(setup, few)[1])
        print(f"n = {n}: median {errors[-1][0]:.1e}, max {errors[-1][1]:.1e}, tabulation {tabulate[-1]:.1f} s, "
              f"trace {timing[-1]:.1f} s (direct {direct_time:.1f} s)")
    errors = np.array(errors)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    ax0.loglog(GRIDS, errors[:, 1], "o-", color="#c0392b", label="max")
    ax0.loglog(GRIDS, errors[:, 0], "s-", color="#3b6fb6", label="median")
    ax0.set(xlabel="grid n (n x n x 2n nodes)", ylabel=r"$|\Delta B| / |B|$",
            title="tabulated vs direct, LCFS to wall")
    ax1.plot(GRIDS, timing, "o-", color="#3b6fb6", label="tabulated")
    ax1.axhline(direct_time, color="k", ls="--", lw=1, label="direct (VmecExtender)")
    ax1.set(xlabel="grid n", ylabel="trace wall time [s]", ylim=(0, None),
            title=f"32 protons, {TMAX * 1e6:.0f} $\\mu$s, after compilation")
    for ax in (ax0, ax1):
        ax.set_xticks(GRIDS, [str(n) for n in GRIDS])
        ax.minorticks_off()
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, "wall_tracing_interpolation.webp")

    # One ensemble through the n = 48 table
    trace, _ = _trace(vj.essos_tracing_fields(wout, coils, wall=WALL, n=48, plasma="vacuum"),
                      _particles(NPARTICLES))
    crossed, struck = np.isfinite(trace.lcfs_times), np.asarray(trace.wall_hits)
    fates = {"stays inside": ~crossed, "crosses and returns": crossed & ~struck & (trace.returns > 0),
             "strikes the wall": struck}
    print({k: int(v.sum()) for k, v in fates.items()}, "of", NPARTICLES)

    # Signed distance to the LCFS along three orbits
    distance = SurfaceClassifier(lcfs, h=0.01, padding=0.06).evaluate_xyz
    times = np.asarray(trace.times) * 1e6
    xyz = np.asarray(trace.trajectories_xyz)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for label, mask in fates.items():
        i = np.flatnonzero(mask)[np.argmax(trace.returns[mask])] if label == "crosses and returns" \
            else np.flatnonzero(mask)[0]
        valid = np.isfinite(xyz[i, :, 0])
        d = -np.asarray(jax.vmap(distance)(jnp.asarray(xyz[i, valid]))) * 100
        ax.plot(times[valid], d, lw=1, color=FATE_COLORS[label], label=label)
        if label == "strikes the wall":
            ax.plot(times[valid][-1], d[-1], "kx", ms=7, mew=1.5)
    ax.axhline(0, color="k", lw=1)
    ax.axhline(WALL * 100, color="k", lw=1, ls="--")
    ax.text(times[-1], 0.15, "LCFS", ha="right", va="bottom", fontsize=8)
    ax.text(times[-1], WALL * 100 + 0.15, "wall", ha="right", va="bottom", fontsize=8)
    ax.set(xlabel=r"t [$\mu$s]", ylabel="distance outside the LCFS [cm]", xlim=(0, times[-1]))
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, "wall_tracing_reentry.webp")

    # Strike points on the wall, unrolled
    hits = np.asarray(trace.wall_positions)[struck]
    phi = np.mod(np.arctan2(hits[:, 1], hits[:, 0]), 2 * np.pi)
    g = np.asarray(wall.gamma)
    wall_phi = np.mod(np.arctan2(g[:, 0, 1], g[:, 0, 0]), 2 * np.pi)
    centre_r = np.hypot(g[..., 0], g[..., 1]).mean(axis=1)
    k = np.argmin(np.abs(np.angle(np.exp(1j * (phi[:, None] - wall_phi[None, :])))), axis=1)
    theta = np.mod(np.arctan2(hits[:, 2], np.hypot(hits[:, 0], hits[:, 1]) - centre_r[k]), 2 * np.pi)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    sc = ax.scatter(phi, theta, c=np.asarray(trace.wall_times)[struck] * 1e6, s=14, cmap="viridis")
    ax.set(xlabel=r"toroidal angle $\phi$", ylabel=r"poloidal angle (geometric)", xlim=(0, 2 * np.pi),
           ylim=(0, 2 * np.pi))
    ticks = np.linspace(0, 2 * np.pi, 5)
    labels = ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)
    ax.set_title(f"{struck.sum()} of {NPARTICLES} protons strike the wall in {TMAX * 1e6:.0f} $\\mu$s",
                 fontsize=9)
    fig.colorbar(sc, ax=ax, label=r"strike time [$\mu$s]")
    fig.tight_layout()
    _save(fig, "wall_tracing_strikes.webp")


if __name__ == "__main__":
    main()
