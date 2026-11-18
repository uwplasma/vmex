#!/usr/bin/env python
"""Plot optimized QA/QH/QP, QI and pwO equilibria for the README.

Run the scripts in ``examples/optimization`` first, then run::

    python examples/plot_optimized_families.py

The QA/QH/QP inputs are read from the current directory. The QI comparison
uses the four validated inputs bundled in ``examples/data``. Each column shows
four equally spaced toroidal cuts over one field period, the 3-D boundary, and
``|B|`` on the LCFS in Boozer coordinates. The pwO figure compares the
``pwO_optimization.py`` seed and result and adds their ``epsilon_eff^(3/2)``
and ``|iota|`` profiles; it needs ``vmex[neoclassical]``. The ``pwo_bootstrap`` figure
compares ``pwO_optimization.py`` runs with ``BOOTSTRAP_TERM = None`` and ``"band"``
(directories ``pwO`` and ``pwO_band``): ``|B|``, the DKX bootstrap current and the alpha
losses. The DKX files ``dkx_<case>.json`` (``pwO``, ``pwO_band``)
come from ``bootstrap_bench.py --res 25,33,64,8`` run on each ``wout``.
"""

import json
from dataclasses import replace

from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np

import vmex as vj
from vmex import optimize as opt
from vmex.core.boozer import run_booz_xform
from vmex.core.plotting import boozer_modB_on_surface, surface_modB, surface_rz


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "examples" / "data"
BLUE = "#2a78d6"
CUT_COLORS = ("#a8cff7", "#6ba7e8", "#367fd0", "#174f9b")
QS_CASES = (
    ("QA", "QA  ·  nfp 2", 1, 0),
    ("QH", "QH  ·  nfp 3", 1, -1),
    ("QP", "QP  ·  nfp 2", 0, 1),
)
QI_CASES = (
    (DATA / "input.nfp1_QI", "QI  ·  nfp 1"),
    (DATA / "input.nfp2_QI", "QI  ·  nfp 2"),
    (DATA / "input.nfp3_QI_fixed_resolution_final", "QI  ·  nfp 3"),
    (DATA / "input.nfp4_QI_finite_beta", "QI  ·  nfp 4"),
)
SURFACES = np.linspace(0.1, 1.0, 8)
PLOT_DPI = 220
QS_INPUT_DIR = Path.cwd()
OUT_DIR = Path.cwd()
PLOT_ONLY = "all"  # "all", "qs", "qi", "pwo" or "pwo_bootstrap"
PWO_NFP = 3                  # the pwO_optimization.py default


def _boundary(inp, theta, phi):
    """Return boundary ``R, Z`` on a tensor grid of angles."""

    rbc, zbs = np.asarray(inp.rbc), np.asarray(inp.zbs)
    rbs = np.zeros_like(rbc) if inp.rbs is None else np.asarray(inp.rbs)
    zbc = np.zeros_like(zbs) if inp.zbc is None else np.asarray(inp.zbc)
    radius = np.zeros((theta.size, phi.size)); height = np.zeros_like(radius)
    for index in range(rbc.shape[0]):
        n = index - inp.ntor
        for m in range(rbc.shape[1]):
            angle = m * theta[:, None] - n * inp.nfp * phi[None, :]
            radius += rbc[index, m] * np.cos(angle) + rbs[index, m] * np.sin(angle)
            height += zbs[index, m] * np.sin(angle) + zbc[index, m] * np.cos(angle)
    return radius, height


def _cross_sections(axis, inp):
    theta = np.linspace(0.0, 2.0 * np.pi, 241)
    phi = np.arange(4) * np.pi / (2 * inp.nfp)
    radius, height = _boundary(inp, theta, phi)
    labels = (r"$\phi=0$", r"$\pi/(2N_{FP})$", r"$\pi/N_{FP}$", r"$3\pi/(2N_{FP})$")
    for index, (color, label) in enumerate(zip(CUT_COLORS, labels, strict=True)):
        axis.plot(radius[:, index], height[:, index], color=color, lw=1.6, label=label)
    axis.set_aspect("equal", adjustable="datalim"); axis.tick_params(labelsize=7)
    axis.spines[["top", "right"]].set_visible(False)


def _boundary_3d(figure, axis, wout):
    theta = np.linspace(0.0, 2.0 * np.pi, 180)
    phi = np.linspace(0.0, 2.0 * np.pi, min(720, max(360, 120 * int(wout.nfp))))
    radius, height = surface_rz(wout, s_index=int(wout.ns) - 1, theta=theta, phi=phi)
    mod_b = surface_modB(wout, s_index=int(wout.ns) - 1, theta=theta, phi=phi)
    phi_grid = np.meshgrid(phi, theta)[0]
    x, y = radius * np.cos(phi_grid), radius * np.sin(phi_grid)
    norm = Normalize(float(np.min(mod_b)), float(np.max(mod_b)))
    axis.plot_surface(x, y, height, facecolors=cm.jet(norm(mod_b)), linewidth=0.0,
                      antialiased=False, shade=False, rstride=1, cstride=1)
    scale = 0.55 * max(np.max(np.abs(x)), np.max(np.abs(y)))
    axis.auto_scale_xyz([-scale, scale], [-scale, scale], [-0.62 * scale, 0.62 * scale])
    axis.set_box_aspect((1, 1, 0.62), zoom=1.14); axis.view_init(elev=30, azim=-55); axis.set_axis_off()
    colorbar = figure.colorbar(cm.ScalarMappable(norm=norm, cmap="jet"), ax=axis,
                              pad=0.0, fraction=0.045, shrink=0.62)
    colorbar.ax.tick_params(labelsize=7)


def _boozer(figure, axis, wout, tag):
    with tempfile.TemporaryDirectory() as directory:
        path = vj.write_wout(Path(directory) / f"wout_{tag}.nc", wout)
        booz = run_booz_xform(path, mbooz=24, nbooz=24)
        theta, phi, mod_b = boozer_modB_on_surface(booz, s_index=-1, ntheta=90, nphi=160)
    contours = axis.contour(phi * wout.nfp / (2 * np.pi), theta / (2 * np.pi),
                            mod_b, levels=22, cmap="jet", linewidths=0.8)
    figure.colorbar(contours, ax=axis, pad=0.02, fraction=0.05).ax.tick_params(labelsize=7)
    axis.set_xlabel("Boozer toroidal angle (periods)", fontsize=8); axis.tick_params(labelsize=7)


def _pwo_seed():
    """The seed deck of pwO_optimization.py: a circular torus plus a rotating ellipse."""
    inp = vj.VmecInput.from_file(DATA / f"input.minimal_seed_nfp{PWO_NFP}")
    rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
    rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -0.05, 0.05
    return replace(inp, rbc=rbc, zbs=zbs, delt=0.5, ftol_array=np.array([1e-10]), niter_array=np.array([30000]))


def _solve(inp, ladder=(37, 45, 55, 71)):
    """Solve on the final radial grid, restarting in small steps as the example does."""
    equilibrium = None
    for ns in ladder:
        equilibrium = opt.solve_equilibrium(
            replace(inp, ns_array=np.array([ns])), verbose=False, raise_on_max_iterations=True,
            initial_state=None if equilibrium is None else equilibrium.solution)
    return equilibrium


def _draw_pwo(output):
    from vmex.core.neoclassical import screen_transport

    result = vj.VmecInput.from_file(QS_INPUT_DIR / f"input.pwO_optimized_nfp{PWO_NFP}")
    cases = ((_pwo_seed(), "Near-circular seed", "#566a79"), (result, "pwO", "#c56627"))
    figure = plt.figure(figsize=(8.1, 7.5), dpi=PLOT_DPI)
    grid = figure.add_gridspec(3, 3, height_ratios=(1.0, 1.2, 1.05), hspace=0.38, wspace=0.6)
    profile, iota = figure.add_subplot(grid[:2, 2]), figure.add_subplot(grid[2, 2])
    for column, (inp, title, color) in enumerate(cases):
        equilibrium = _solve(inp)
        wout, ns = equilibrium.wout, int(equilibrium.wout.ns)
        half = (np.arange(ns - 1) + 0.5) / (ns - 1)
        surfaces = half[np.abs(half[:, None] - np.linspace(0.1, 0.9, 8)).argmin(axis=0)]
        q = np.asarray(screen_transport(wout, surfaces))
        profile.semilogy(surfaces, q, "o-", color=color, label=f"{title}, mean {q.mean():.1e}")
        iota.plot(np.linspace(0, 1, ns), np.abs(wout.iotaf), color=color)
        cross = figure.add_subplot(grid[0, column]); _cross_sections(cross, inp)
        cross.set_title(f"{title}\nnfp {inp.nfp}", loc="left", fontsize=10)
        if column == 0:
            cross.set_ylabel("Z [m]", fontsize=8); cross.legend(fontsize=6.5, frameon=False)
        boundary = figure.add_subplot(grid[1, column], projection="3d"); _boundary_3d(figure, boundary, wout)
        boozer = figure.add_subplot(grid[2, column]); _boozer(figure, boozer, wout, f"pwo{column}")
        if column == 0:
            boozer.set_ylabel(r"Boozer poloidal angle / $2\pi$", fontsize=8)
    profile.set_xlabel("$s$", fontsize=8); profile.set_ylabel(r"$\epsilon_{\rm eff}^{3/2}$", fontsize=8)
    profile.legend(fontsize=7, frameon=False, loc="upper left"); iota.set_xlabel("$s$", fontsize=8); iota.set_ylabel(r"$|\iota|$", fontsize=8)
    for axis in (profile, iota):
        axis.tick_params(labelsize=7); axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure); print(f"wrote {output}")


def _draw_pwo_bootstrap(output):
    cases = (("pwO", QS_INPUT_DIR / "pwO" / f"input.pwO_optimized_nfp{PWO_NFP}", "pwO", "#c56627"),
             ("pwO + bootstrap control", QS_INPUT_DIR / "pwO_band" / f"input.pwO_optimized_nfp{PWO_NFP}",
              "pwO_band", "#2a78d6"))
    figure = plt.figure(figsize=(8.1, 7.5), dpi=PLOT_DPI)
    grid = figure.add_gridspec(2, 5, height_ratios=(1.2, 1.0), width_ratios=(1.4, 1.4, 0.3, 1, 1), hspace=0.3, wspace=0.75)
    current, loss = figure.add_subplot(grid[0, 3:]), figure.add_subplot(grid[1, 3:])
    for index, (title, source, tag, color) in enumerate(cases):
        equilibrium = _solve(vj.VmecInput.from_file(source))
        dkx = json.loads((QS_INPUT_DIR / f"dkx_{tag}.json").read_text())
        current.plot(dkx["surfaces"], 1e-6 * np.array(dkx["dkx_jdotb"]), "o-", color=color,
                     label=rf"{title}, $I_{{bs}}$ = {1e-3 * dkx['I_dkx']:.0f} kA")
        alphas = vj.trace_alphas(equilibrium.wout, nparticles=1000, tmax=5e-3, s=0.3, seed=2, times_to_trace=100)
        t, lost = np.asarray(alphas.times), np.asarray(alphas.loss_fractions)
        loss.plot(t[t > 0], 100 * lost[t > 0], color=color,
                  label=f"{title}, {100 * alphas.loss_fraction:.1f} %")
        boozer = figure.add_subplot(grid[0, index]); _boozer(figure, boozer, equilibrium.wout, f"bs{index}")
        boozer.set_title(title, loc="left", fontsize=9)
        _boundary_3d(figure, figure.add_subplot(grid[1, index], projection="3d"), equilibrium.wout)
        boozer.set_ylabel(r"Boozer poloidal angle / $2\pi$" if index == 0 else "", fontsize=8)
        if index:
            boozer.set(yticklabels=[], xlabel="")
    current.set(xlabel="$s$", ylabel=r"DKX $\langle j_\parallel B\rangle$ [MA T/m$^2$]")
    loss.set(xscale="log", xlabel="time [s]", ylabel="alphas lost [%]")
    for axis, corner in ((current, "center right"), (loss, "lower right")):
        axis.legend(fontsize=6.5, loc=corner, framealpha=0.9, edgecolor="none"); axis.tick_params(labelsize=7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.05, pil_kwargs={"quality": 70})
    plt.close(figure); print(f"wrote {output}")


def _find_qs_input(directory, tag):
    for name in (f"input.{tag}_optimized", f"input.{tag.lower()}_optimized"):
        path = directory / name
        if path.exists():
            return path
    raise FileNotFoundError(f"run examples/optimization/{tag}_optimization.py first; missing input.{tag}_optimized")


def _draw(cases, output, *, qi=False):
    figure = plt.figure(figsize=(2.7 * len(cases), 7.5), dpi=PLOT_DPI)
    grid = figure.add_gridspec(3, len(cases), height_ratios=(1.0, 1.2, 1.05), hspace=0.38, wspace=0.42)
    for column, case in enumerate(cases):
        path, title = case[:2]
        inp = vj.VmecInput.from_file(path)
        equilibrium = opt.solve_equilibrium(inp)
        if qi:
            score = float(opt.quasi_isodynamic_residual_from_wout(
                equilibrium.wout, surfaces=SURFACES)["total"])
            label = f"QI = {score:.2e}"
        else:
            residual = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=case[2], helicity_n=case[3])
            label = f"QS = {float(residual.total(equilibrium)):.2e}"
        cross = figure.add_subplot(grid[0, column]); _cross_sections(cross, inp)
        cross.set_title(f"{title}\n{label}", loc="left", fontsize=10)
        if column == 0:
            cross.set_ylabel("Z [m]", fontsize=8); cross.legend(fontsize=6.5, frameon=False)
        boundary = figure.add_subplot(grid[1, column], projection="3d"); _boundary_3d(figure, boundary, equilibrium.wout)
        boozer = figure.add_subplot(grid[2, column]); _boozer(figure, boozer, equilibrium.wout, title.replace(" ", "_"))
        if column == 0:
            boozer.set_ylabel(r"Boozer poloidal angle / $2\pi$", fontsize=8)
    figure.savefig(output, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.05, pil_kwargs={"quality": 55})
    plt.close(figure); print(f"wrote {output}")


if PLOT_ONLY not in ("all", "qs", "qi", "pwo", "pwo_bootstrap"):
    raise ValueError("PLOT_ONLY must be 'all', 'qs', 'qi', 'pwo' or 'pwo_bootstrap'")
OUT_DIR.mkdir(parents=True, exist_ok=True)
if PLOT_ONLY in ("all", "qs"):
    cases = [(_find_qs_input(QS_INPUT_DIR, tag), title, m, n) for tag, title, m, n in QS_CASES]
    _draw(cases, OUT_DIR / "readme_optimization.webp")
if PLOT_ONLY in ("all", "qi"):
    _draw(QI_CASES, OUT_DIR / "readme_qi.webp", qi=True)
if PLOT_ONLY == "pwo":  # not part of "all": it needs the pwO_optimization.py result
    _draw_pwo(OUT_DIR / "readme_pwo.webp")
if PLOT_ONLY == "pwo_bootstrap":  # needs the two pwO_optimization.py results and their DKX files
    _draw_pwo_bootstrap(OUT_DIR / "readme_pwo_bootstrap.webp")
