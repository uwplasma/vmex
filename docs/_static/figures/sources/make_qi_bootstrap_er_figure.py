#!/usr/bin/env python3
"""Regenerate the two README figures of the QI bootstrap-at-ambipolar-E_r example.

Runs ``examples/optimization/QI_optimization_bootstrap_er.py`` at its default
settings in a scratch directory, then draws from the objects it leaves behind:

* ``readme_qi_bootstrap_er_shape.webp``: the plasma boundary coloured by |B| and
  the Boozer |B| contours of the outermost surface, before (the seed with its
  DKX-consistent current) and after the optimization;
* ``readme_qi_bootstrap_er_current.webp``: the ambipolar E_r, the equilibrium,
  DKX and Redl <j.B>, and the objective history.

Output is lossy WebP (quality 50). Needs ``pip install "vmex[optimizers,neoclassical]"``.

Usage::

    python docs/_static/figures/sources/make_qi_bootstrap_er_figure.py
"""

from __future__ import annotations

import io
import os
import runpy
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageChops  # noqa: E402

import vmex as vj  # noqa: E402
from vmex.core.boozer import run_booz_xform  # noqa: E402
from vmex.core.plotting import boozer_modB_on_surface, plot_boundary_3d  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
EXAMPLE = REPO / "examples" / "optimization" / "QI_optimization_bootstrap_er.py"
FIGURES = REPO / "docs" / "_static" / "figures"
QUALITY = 45

CASES = ("before", "after")
plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "legend.fontsize": 9,
                     "axes.grid": True, "grid.alpha": 0.3})


def save_webp(fig, name: str, width: int) -> None:
    """Rasterize ``fig`` and write it as a ``width`` px WebP."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    image = Image.open(buf).convert("RGB")
    image = image.resize((width, round(width * image.height / image.width)), Image.LANCZOS)
    image.save(FIGURES / name, "WEBP", quality=QUALITY, method=6)
    print(f"wrote {FIGURES / name}")


def shape_figure(wouts: dict, tmp: Path) -> None:
    """Boundary coloured by |B| and Boozer |B| contours of the outermost surface, before and after."""
    boundary, booz = {}, {}
    for label in CASES:
        plot_boundary_3d(wouts[label], tmp / f"{label}.png", ntheta=120, nzeta=240)
        image = Image.open(tmp / f"{label}.png").convert("RGB")
        boundary[label] = np.asarray(image.crop(ImageChops.difference(
            image, Image.new("RGB", image.size, (255, 255, 255))).getbbox()))
        wout_path = vj.write_wout(tmp / f"wout_{label}.nc", wouts[label])
        booz[label] = boozer_modB_on_surface(run_booz_xform(wout_path, mbooz=24, nbooz=24),
                                             s_index=-1, ntheta=121, nphi=121)
    levels = np.linspace(min(b[2].min() for b in booz.values()),
                         max(b[2].max() for b in booz.values()), 21)
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.6), layout="constrained",
                             gridspec_kw={"height_ratios": [1.0, 0.85]})
    for column, label in enumerate(CASES):
        axes[0, column].imshow(boundary[label])
        axes[0, column].axis("off")
        axes[0, column].set_title(f"{label}: boundary, colour = $|B|$")
        theta, phi, B = booz[label]
        axes[1, column].contour(
            phi * wouts[label].nfp / (2 * np.pi), theta / (2 * np.pi), B,
            levels=levels, cmap="jet", linewidths=0.9)
        axes[1, column].set(xlabel="Boozer toroidal angle [field periods]",
                            ylabel=r"$\theta_B/2\pi$",
                            title=f"{label}: $|B|$ in Boozer coordinates, $s = 1$")
    fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(levels[0], levels[-1]), cmap="jet"),
                 ax=axes[1, :], label=r"$|B|$ [T]", pad=0.02)
    save_webp(fig, "readme_qi_bootstrap_er_shape.webp", 900)


def current_figure(rows: dict, monitor) -> None:
    """E_r, <j.B> of the equilibrium, DKX and Redl, and the objective history.

    ``rows[label]`` is the ``(s, VMEX, Redl, DKX, E_r)`` of ``kinetic_rows``.
    """
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6), layout="constrained")
    for label, (s, j_vmex, j_redl, j_dkx, e_r) in rows.items():
        after = label == "after"
        line, hollow = ("-" if after else "--"), (None if after else "none")
        ax[0].plot(s, e_r, "o" + line, color="C0", mfc=hollow, label=label)
        ax[1].plot(s, j_vmex, "s" + line, color="C0", mfc=hollow, label=f"equilibrium, {label}")
        ax[1].plot(s, j_dkx, "o" + line, color="C3", mfc=hollow, label=f"DKX at $E_r$, {label}")
        ax[2].semilogy(s, np.abs(j_dkx), "o" + line, color="C3", mfc=hollow, label=f"DKX, {label}")
        ax[2].semilogy(s, np.abs(j_redl), "^" + line, color="C2", mfc=hollow, label=f"Redl, {label}")
    ax[0].set(xlabel=r"$s$", ylabel=r"$E_r$ [kV/m]", title=r"ambipolar ion root $E_r$")
    ax[1].set(xlabel=r"$s$", ylabel=r"$\langle j\cdot B\rangle$ [MA T/m$^2$]",
              title=r"equilibrium vs DKX $\langle j\cdot B\rangle$")
    ax[2].set(xlabel=r"$s$", ylabel=r"$|\langle j\cdot B\rangle|$ [MA T/m$^2$]",
              title="DKX vs Redl (quasi-axisymmetric formula)")
    iteration = np.asarray([r.iteration for r in monitor.records])
    ax[3].plot(iteration, np.asarray(monitor.history["total"]), "o-")
    ax[3].set(xlabel="accepted evaluation", ylabel="objective (monitor total)", yscale="log",
              title="objective history")
    for a in ax[:3]:
        a.legend(fontsize=8)
    save_webp(fig, "readme_qi_bootstrap_er_current.webp", 1300)


def main() -> None:
    os.environ.pop("VMEX_EXAMPLES_CI", None)
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as scratch:
        tmp = Path(scratch)
        os.chdir(tmp)
        try:
            ns = runpy.run_path(str(EXAMPLE))
        finally:
            os.chdir(cwd)
        equilibria = {"before": ns["seed_equilibrium"], "after": ns["final_equilibrium"]}
        shape_figure({k: eq.wout for k, eq in equilibria.items()}, tmp)
        current_figure({k: ns["kinetic_rows"](eq) for k, eq in equilibria.items()}, ns["monitor"])


if __name__ == "__main__":
    main()
