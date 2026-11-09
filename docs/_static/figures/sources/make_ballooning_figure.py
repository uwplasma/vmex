#!/usr/bin/env python3
"""Draw ``readme_ballooning.webp``: infinite-n ideal ballooning of the finite-beta QA seed.

Left: the eigenvalue lambda(s, alpha) at zeta0 = 0. Middle: lambda(s) on the
alpha = 0 line, third- and fifth-order radial stencils, against the COBRAVMEC
values frozen in ``tests/test_stability.py``. Right: eigenvalue error against
field-line resolution, raw and with the Richardson step.

Usage::

    python docs/_static/figures/sources/make_ballooning_figure.py
"""

import dataclasses
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

jax.config.update("jax_enable_x64", True)

import vmex.core.optimize as opt  # noqa: E402
import vmex.core.stability as stab  # noqa: E402
from vmex.core.input import VmecInput  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "docs" / "_static" / "figures" / "readme_ballooning.webp"
NS = 71
# COBRAVMEC (STELLOPT build) on this equilibrium, alpha = 0, converted to the
# vmex normalization (tests/test_stability.py): (row, lambda).
COBRA = ((59, 5.280e-3), (63, 1.2441e-2), (66, 8.888e-3))
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#6b6b68"

inp = dataclasses.replace(
    VmecInput.from_file(REPO / "examples" / "data" / "input.nfp2_QA_finite_beta"),
    ns_array=np.array([NS]), ftol_array=np.array([1e-14]), niter_array=np.array([20000]))
eq = opt.solve_equilibrium(inp, verbose=True)
if not eq.result.converged:
    raise RuntimeError("the equilibrium did not converge")


def lam(**kw):
    return np.asarray(stab.ballooning_lambda(eq.state, eq.runtime, **kw))


s_map, alpha_map = np.linspace(0.3, 0.95, 27), np.linspace(0.0, np.pi, 25)
print("lambda(s, alpha) map", flush=True)
grid = lam(surfaces=s_map, alphas=alpha_map)[:, :, 0]
s_line = np.linspace(0.6, 0.96, 73)
print("alpha = 0 line", flush=True)
fifth = lam(surfaces=s_line, alphas=[0.0])[:, 0, 0]
third = lam(surfaces=s_line, alphas=[0.0], radial_order=2)[:, 0, 0]
print("field-line convergence", flush=True)
turns = np.array([16, 24, 32, 48, 64, 96, 128, 192])
one = dict(surfaces=[59 / (NS - 1)], alphas=[0.0], nturns=5.0)
raw, rich = (np.array([lam(npoints=int(10 * t + 1), richardson=r, **one).item() for t in turns])
             for r in (False, True))
ref = lam(npoints=int(10 * 384 + 1), **one).item()

fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5), gridspec_kw={"width_ratios": (1.15, 1.15, 1)})
norm = TwoSlopeNorm(0.0, vmin=-0.01, vmax=0.02)
mesh = axes[0].contourf(s_map, alpha_map, np.clip(grid.T, -0.01, 0.02), levels=np.linspace(-0.01, 0.02, 31),
                        cmap="RdBu_r", norm=norm)
axes[0].contour(s_map, alpha_map, grid.T, levels=[0.0], colors="k", linewidths=0.8)
fig.colorbar(mesh, ax=axes[0], label=r"$\lambda=(\gamma a_N/v_A)^2$")
axes[0].set(xlabel="s", ylabel=r"field line $\alpha$", yticks=[0, np.pi / 2, np.pi],
            yticklabels=["0", r"$\pi/2$", r"$\pi$"], title="unstable (red) and stable (blue) lines")
axes[1].plot(s_line, fifth, color=BLUE, lw=1.4, label="vmex, 5-point radial")
axes[1].plot(s_line, third, color=GREY, lw=1.2, ls="--", label="vmex, 3-point radial")
axes[1].plot([r / (NS - 1) for r, _ in COBRA], [v for _, v in COBRA], "o", color=ORANGE, ms=6, label="COBRAVMEC")
axes[1].axhline(0.0, color="k", lw=0.6)
axes[1].set(xlabel="s", ylabel=r"$\lambda$", title=r"$\alpha=0$ line, ns = 71")
axes[2].loglog(turns, np.abs(raw / ref - 1), "s--", color=GREY, ms=4, lw=1.2, label="three-point")
axes[2].loglog(turns, np.abs(rich / ref - 1), "o-", color=BLUE, ms=4, lw=1.4, label="Richardson")
axes[2].set(xlabel="points per poloidal turn", ylabel=r"$|\lambda/\lambda_\mathrm{ref}-1|$",
            title="field-line resolution, s = 0.84")
axes[2].set_xticks([16, 32, 64, 128])
axes[2].xaxis.set_major_formatter(ScalarFormatter())
axes[2].xaxis.set_minor_formatter(plt.NullFormatter())
for ax in axes[1:]:
    ax.legend(fontsize=7, frameon=False)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#e4e4e1", lw=0.8)
fig.tight_layout()
fig.canvas.draw()
Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).quantize(256).save(OUT, lossless=True, method=6)
print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")
